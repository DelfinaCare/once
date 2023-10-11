import asyncio
import collections.abc
import enum
import threading

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


class AsyncGeneratorWrapper:
    """Wrapper around an async generator which only runs once.

    Subsequent calls will return results from the first call, which is
    evaluated lazily.
    """

    def __init__(self, func, *args, **kwargs) -> None:
        self.generator: collections.abc.AsyncGenerator | None = func(*args, **kwargs)
        self.finished = False
        self.results: list = []
        self.generating = False
        self.lock = asyncio.Lock()

    async def yield_results(self) -> collections.abc.AsyncGenerator:
        i = 0
        send = None
        next_val = None

        # A copy of self.generating that we can access outside of the lock.
        generating = None

        # Indicates that we're tied for the head generator, but someone started generating the next
        # result first, so we should just poll until the result is available.
        waiting_for_generating = False

        while True:
            if waiting_for_generating:
                # This is a load bearing sleep. We're waiting for the leader to generate the result, but
                # we have control of the lock, so the async with will never yield execution to the event loop,
                # so we would loop forever. By awaiting sleep(0), we yield execution which will allow us to
                # poll for self.generating readiness.
                await asyncio.sleep(0)
                waiting_for_generating = False
            async with self.lock:
                if i == len(self.results) and not self.finished:
                    if self.generating:
                        # We're at the lead, but someone else is generating the next value
                        # so we just hop back onto the next iteration of the loop
                        # until it's ready.
                        waiting_for_generating = True
                        continue
                    # We're at the lead and no one else is generating, so we need to increment
                    # the iterator. We just store the value in self.results so that
                    # we can later yield it outside of the lock.
                    assert self.generator is not None
                    # TODO(matt): Is the fact that we have to suppress typing here a bug?
                    self.generating = self.generator.asend(send)  # type: ignore
                    generating = self.generating
                elif i == len(self.results) and self.finished:
                    # All done.
                    return
                else:
                    # We already have the correct result, so we grab it here to
                    # yield it outside the lock.
                    next_val = self.results[i]

            if generating:
                try:
                    next_val = await generating
                except StopAsyncIteration:
                    async with self.lock:
                        self.generator = None  # Allow this to be GCed.
                        self.finished = True
                        self.generating = None
                        generating = None
                        return
                async with self.lock:
                    self.results.append(next_val)
                    generating = None
                    self.generating = None

            send = yield next_val
            i += 1


class _IteratorAction(enum.Enum):
    # Generating the next value from the underlying iterator
    GENERATING = 1
    # Yield an already computed value
    YIELDING = 2
    # Waiting for the underlying iterator, already triggered from another call.
    WAITING = 3


class GeneratorWrapper:
    """Wrapper around an sync generator which only runs once.

    Subsequent calls will return results from the first call, which is
    evaluated lazily.
    """

    def __init__(self, func, *args, **kwargs) -> None:
        self.generator: collections.abc.Generator | None = func(*args, **kwargs)
        self.finished = False
        self.results: list = []
        self.generating = False
        self.lock = threading.Lock()
        self.next_send = None

    def yield_results(self) -> collections.abc.Generator:
        # Fast path for subsequent repeated call:
        with self.lock:
            finished = self.finished
        if finished:
            yield from self.results
            return
        i = 0
        yield_value = None
        next_send = None
        # Fast path for subsequent calls will not require a lock
        while True:
            action: _IteratorAction | None = None
            # With a lock, we figure out which action to take, and then we take it after release.
            with self.lock:
                if i == len(self.results):
                    if self.finished:
                        return
                    if self.generating:
                        action = _IteratorAction.WAITING
                    else:
                        action = _IteratorAction.GENERATING
                        next_send = self.next_send
                        self.generating = True
                else:
                    action = _IteratorAction.YIELDING
                    yield_value = self.results[i]
            if action == _IteratorAction.WAITING:
                continue
            if action == _IteratorAction.YIELDING:
                next_send = yield yield_value
                i += 1
                # If we have just sent the last element and we have not yet kicked off the next
                # iteration, we need to record the next send value.
                with self.lock:
                    if i == len(self.results) and not self.generating:
                        self.next_send = next_send
                continue
            if action == _IteratorAction.GENERATING:
                assert self.generator is not None
                try:
                    result = self.generator.send(next_send)
                except StopIteration:
                    # This lock should be unnecessary, which by definition means there should be no
                    # contention on it, so we use it to preserve our assumptions about variables which
                    # are modified under lock.
                    with self.lock:
                        self.finished = True
                        self.generator = None  # Allow this to be GCed.
                        self.generating = False
                else:
                    with self.lock:
                        self.generating = False
                        self.results.append(result)
