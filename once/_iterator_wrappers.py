import asyncio
import abc
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


class _GeneratorWrapperBase(abc.ABC):
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
        self,
        reset_on_exception: bool,
        func: collections.abc.Callable,
        allow_reset: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self.callable: collections.abc.Callable | None = functools.partial(func, *args, **kwargs)
        self.generator = self.callable()
        self.result = IteratorResults()
        self.generating = False
        self.reset_on_exception = reset_on_exception
        self.allow_reset = allow_reset

    # Why do we make the generating boolean property abstract?
    # This makes the code when the iterator state is WAITING more efficient. If this was simply
    # a boolean property, the iterator wrapper implementation would need a "sleep(0)" operation
    # whenever it was in the WAITING state. However, this lets the underlying event use a more
    # semantically appropriate event to implement the state of generating, which it can then also
    # wait on whenever it needs to handle the WAITING case. In both cases, the underlying function
    # of event.wait() is the same as the sleep(0), but the appropriate semantics make the logic
    # clearer.
    # Why would we have needed a sleep(0)
    # In the async case, the coroutine in WAITING state would be waiting for the leader (in state
    # GENERATING) to finish generating the result, but because WAITING would have control of the
    # lock, it will never yield execution to the event loop for GENERATING to execute,causing an
    # infinite loop. An await sleep(0) would trigger this yielding of execution, allowing the event
    # loop to loop through to the GENERATING so it can complete and unblock the WAITING coroutine.
    # In the sync case, we also need a sleep(0) for a different reason. While a thread is in the
    # WAITING state, it would otherwise hog the Global Interpreter Lock in its while loop until
    # the interprer decides to switch threads. With N threads, N - 1 of them would be in the
    # WAITING state, and a round-robin interpreter would give each of their while loops the
    # thread switching time in execution before reaching the single thread in the GENERATING state.
    # Theoretically, this could result in the GIL only working on the thread in the GENERATING
    # state 1/Nth of the time, without a time.sleep(0) to manually trigger a GIL switch. In
    # practice, removing the time.sleep(0) call would result in large drops in performance when
    # running the multithreaded unit tests.
    @property
    @abc.abstractmethod
    def generating(self) -> bool:
        raise NotImplementedError()

    @generating.setter
    @abc.abstractmethod
    def generating(self, val: bool):
        raise NotImplementedError()

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
        if not self.allow_reset:
            # Allow this to be GCed as long as we know we'll never need it again.
            self.callable = None

    def record_item(self, result: IteratorResults, item: typing.Any):
        """Must be called with lock."""
        self.generating = False
        result.items.append(item)

    def reset(self):
        """Must be called with lock."""
        self.result = IteratorResults()
        assert self.callable is not None
        self.generator = self.callable()  # Reset the iterator for the next call.

    def record_exception(self, result: IteratorResults, exception: Exception):
        """Must be called with lock."""
        result.finished = True
        # We need to keep track of the exception so that we can raise it in the same
        # position every time the iterator is called.
        result.exception = exception
        self.generating = False
        if self.reset_on_exception:
            self.reset()
        else:
            self.generator = None  # allow this to be GCed


class AsyncGeneratorWrapper(_GeneratorWrapperBase):
    """Wrapper around an async generator which only runs once.

    Subsequent calls will return results from the first call, which is
    evaluated lazily.
    """

    generator: collections.abc.AsyncGenerator

    def __init__(self, *args, **kwargs) -> None:
        # Must be called before super init so its self.generating setter succeeds.
        self.active_generation_completed = asyncio.Event()
        super().__init__(*args, **kwargs)
        self.lock = asyncio.Lock()

    @property
    def generating(self):
        return not self.active_generation_completed.is_set()

    @generating.setter
    def generating(self, val: bool):
        if val:
            self.active_generation_completed.clear()
        else:
            self.active_generation_completed.set()

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
            if action == _IteratorAction.WAITING:
                await self.active_generation_completed.wait()
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
        # Must be called before super init so its self.generating setter succeeds.
        self.active_generation_completed = threading.Event()
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()

    @property
    def generating(self):
        return not self.active_generation_completed.is_set()

    @generating.setter
    def generating(self, val: bool):
        if val:
            self.active_generation_completed.clear()
        else:
            self.active_generation_completed.set()

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
                self.active_generation_completed.wait()
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
