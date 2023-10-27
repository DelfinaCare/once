import asyncio
import collections.abc
import enum
import functools
import threading
import time
import typing

# Before we begin, a note on the assert statements in this file:
# Why are we using assert in here, you might ask, instead of implementing "proper" error handling?
# In this case, it is actually not being done out of laziness! The assert statements here
# represent our assumptions about the state at that point in time, and are always called with locks
# held, so they **REALLY** should always hold. If the assumption behind one of these asserts fails,
# the subsequent calls are going to fail anyways, so it's not like they are making the code
# artificially brittle. However, they do make testing easer, because we can directly test our
# assumption instead of having hard-to-trace errors, and also serve as very convenient
# documentation of the assumptions.
# We are always open to suggestions if there are other ways to achieve the same functionality in
# python!


class IteratorResults:
    def __init__(self) -> None:
        self.items: list = []
        self.exception: Exception | None = None
        self.finished = False

    # NOTE: We could define a "fast_path", where we know we are already done, and it is safe to
    # directly yield results without further lock checks, by checking the following condition:
    # `self.finished and self.exception is None`. However, in practice, it does not appear to be
    # worth it.


class _IteratorAction(enum.Enum):
    # Generating the next value from the underlying iterator
    GENERATING = 1
    # Yield an already computed value
    YIELDING = 2
    # Waiting for the underlying iterator, already triggered from another call.
    WAITING = 3
    # We can return, we are done!
    RETURNING = 4


class _GeneratorWrapperBase:
    """Base class for generator wrapper.

    Even though the class stores a result, all of the methods separately take a result input.
    Why is that? Great question.

    While yielding results simultaneously from multiple tasks or threads, we want to support
    configurable outcomes for exception handling. Specifically, taking `result` as an argument is
    critical in order to support any *already executing* iterators finishing with their cached
    result and exception, but restarting new iterators from the beginning. This is done by creating
    a local reference to `result` when the iterator is started and passing that around thereafter.

    When an Exception occurs and reset_on_exception, we begin the execution again from the
    beginning for all new iterators. This is required because the received values during execution
    may be critical to the returned. For example, imagine a caller of a paginated API who chose to
    user iterator.send to propagate the next token. Given that the state of the output might depend
    on the state of what was received, to be semantically correct, we must start from the
    beginning.
    """

    def __init__(
        self, reset_on_exception: bool, func: collections.abc.Callable, *args, **kwargs
    ) -> None:
        self.callable: collections.abc.Callable | None = functools.partial(func, *args, **kwargs)
        self.generator = self.callable()
        self.result = IteratorResults()
        self.generating = False
        self.reset_on_exception = reset_on_exception

    def compute_next_action(
        self, result: IteratorResults, i: int
    ) -> typing.Tuple[_IteratorAction, typing.Any]:
        """Must be called with lock."""
        if i == len(result.items):
            if result.finished:
                if result.exception:
                    raise result.exception
                return _IteratorAction.RETURNING, None
            if self.generating:
                return _IteratorAction.WAITING, None
            else:
                # If all of these functions are called with locks, we will never have more than one
                # caller have GENERATING at any time.
                self.generating = True
                return _IteratorAction.GENERATING, None
        else:
            return _IteratorAction.YIELDING, result.items[i]

    def record_successful_completion(self, result: IteratorResults):
        """Must be called with lock."""
        result.finished = True
        self.generating = False
        self.generator = None  # Allow this to be GCed.
        self.callable = None  # Allow this to be GCed.

    def record_item(self, result: IteratorResults, item: typing.Any):
        self.generating = False
        result.items.append(item)

    def record_exception(self, result: IteratorResults, exception: Exception):
        """Must be called with lock."""
        result.finished = True
        # We need to keep track of the exception so that we can raise it in the same
        # position every time the iterator is called.
        result.exception = exception
        self.generating = False
        assert self.callable is not None
        self.generator = self.callable()  # Reset the iterator for the next call.
        if self.reset_on_exception:
            self.result = IteratorResults()


class AsyncGeneratorWrapper(_GeneratorWrapperBase):
    """Wrapper around an async generator which only runs once.

    Subsequent calls will return results from the first call, which is
    evaluated lazily.
    """

    generator: collections.abc.AsyncGenerator

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lock = asyncio.Lock()

    async def yield_results(self) -> collections.abc.AsyncGenerator:
        async with self.lock:
            result = self.result

        i = 0
        yield_value = None
        next_send = None

        while True:
            # With a lock, we figure out which action to take, and then we take it after release.
            async with self.lock:
                action, yield_value = self.compute_next_action(result, i)
            if action == _IteratorAction.RETURNING:
                return
            # This is a load bearing sleep. We're waiting for the leader to generate the result,
            # but we have control of the lock, so the async with will never yield execution to the
            # event loop, so we would loop forever. By awaiting sleep(0), we yield execution which
            # will allow us to poll for self.generating readiness.
            if action == _IteratorAction.WAITING:
                await asyncio.sleep(0)
                continue
            if action == _IteratorAction.YIELDING:
                next_send = yield yield_value
                i += 1
                continue
            assert action == _IteratorAction.GENERATING
            assert self.generator is not None
            try:
                item = await self.generator.asend(next_send)
            except StopAsyncIteration:
                async with self.lock:
                    self.record_successful_completion(result)
            except Exception as e:
                async with self.lock:
                    self.record_exception(result, e)
            else:
                async with self.lock:
                    self.record_item(result, item)


class GeneratorWrapper(_GeneratorWrapperBase):
    """Wrapper around an sync generator which only runs once.

    Subsequent calls will return results from the first call, which is
    evaluated lazily.
    """

    generator: collections.abc.Generator

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()

    def yield_results(self) -> collections.abc.Generator:
        with self.lock:
            result = self.result
        i = 0
        yield_value = None
        next_send = None
        while True:
            action: _IteratorAction | None = None
            # With a lock, we figure out which action to take, and then we take it after release.
            with self.lock:
                action, yield_value = self.compute_next_action(result, i)
            if action == _IteratorAction.RETURNING:
                return
            if action == _IteratorAction.WAITING:
                # Indicate to python that it should switch to another thread, so we do not hog the GIL.
                time.sleep(0)
                continue
            if action == _IteratorAction.YIELDING:
                next_send = yield yield_value
                i += 1
                continue
            assert action == _IteratorAction.GENERATING
            assert self.generator is not None
            try:
                item = self.generator.send(next_send)
            except StopIteration:
                with self.lock:
                    self.record_successful_completion(result)
            except Exception as e:
                with self.lock:
                    self.record_exception(result, e)
            else:
                with self.lock:
                    self.record_item(result, item)
