"""The internal _CallState class holds the call state and return value.

This is a simple data class, and could have been implemented as a tuple.
However, it has a lock, and ensures its properties are called while they are
held, and also defines a reset method.
"""

import asyncio
import typing
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


class _CallState:

    def __init__(self, is_async: bool) -> None:
        self.is_async = is_async
        # We are going to be extra pedantic about these next two variables only being read or set
        # with a lock by defining getters and setters which enforce that the lock is held. If this
        # was C++, we would use something like the ABSL_GUARDED_BY macro for compile-time checking
        # (https://github.com/abseil/abseil-cpp/blob/master/absl/base/thread_annotations.h), but
        # this is python :)
        self._called = False
        self._return_value: typing.Any = None
        if self.is_async:
            self.async_lock = asyncio.Lock()
        else:
            self.lock = threading.Lock()

    def _locked(self) -> bool:
        return self.async_lock.locked() if self.is_async else self.lock.locked()

    @property
    def called(self) -> bool:
        """Indicates if the function has been called."""
        assert self._locked()
        return self._called

    @called.setter
    def called(self, state: bool) -> None:
        assert self._locked()
        self._called = state

    @property
    def return_value(self) -> typing.Any:
        """Stores the returned value of the function."""
        assert self._locked()
        return self._return_value

    @return_value.setter
    def return_value(self, value: typing.Any) -> None:
        assert self._locked()
        self._return_value = value

    def reset(self):
        """Resets the state back to uncalled."""
        assert self._locked()
        self._called = False
        self._return_value = None
