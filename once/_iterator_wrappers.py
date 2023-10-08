import asyncio
import collections.abc
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

    async def _yield_results(self) -> collections.abc.AsyncGenerator:
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

    def _yield_results(self) -> collections.abc.Generator:
        i = 0
        # Fast path for subsequent calls will not require a lock
        while True:
            if i < len(self.results):
                yield self.results[i]
                i += 1
                continue
            if self.finished:
                return

            # Initial calls, and concurrent calls before completion will require the lock.
            with self.lock:
                if i < len(self.results):
                    yield self.results[i]
                    i += 1
                    continue
                # Because we hold a lock, this should never be violated.
                # If it does, something has gone seriously wrong!
                assert i == len(self.results)
                if self.finished:
                    return
                # The generator should never be garbage collected while self.finished is False
                # and the lock is held.
                assert self.generator is not None
                try:
                    self.results.append(self.generator.send(self.next_send))
                except StopIteration:
                    self.finished = True
                    self.generator = None  # Allow this to be GCed.
                    return
                else:
                    i += 1
                    self.next_send = yield self.results[-1]
