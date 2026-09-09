import collections.abc

from . import _iterator_wrappers
from . import _state


class _CachedException:
    def __init__(self, exception: Exception):
        self.exception = exception


class _OnceCallableBase:

    def __init__(
        self,
        func: collections.abc.Callable,
        state_factory: collections.abc.Callable[[], _state._CallState],
        retry_exceptions: bool,
    ):
        self._func = func
        self._state_factory = state_factory
        self._retry_exceptions = retry_exceptions


class OnceCallableSyncBase(_OnceCallableBase):
    def reset(self):
        with self._state_factory().lock:
            self._state_factory().reset()


class OnceCallableAsyncBase(_OnceCallableBase):
    async def reset(self):
        async with self._state_factory().async_lock:
            self._state_factory().reset()


class OnceCallableSyncFunction(OnceCallableSyncBase):
    def __call__(self, *args, **kwargs):
        call_state = self._state_factory()
        with call_state.lock:
            if not call_state.called:
                try:
                    call_state.return_value = self._func(*args, **kwargs)
                except Exception as exception:
                    if self._retry_exceptions:
                        raise exception
                    call_state.return_value = _CachedException(exception)
                call_state.called = True
            return_value = call_state.return_value
        if isinstance(return_value, _CachedException):
            raise return_value.exception
        return return_value


class OnceCallableSyncGenerator(OnceCallableSyncBase):
    def __call__(self, *args, **kwargs):
        call_state = self._state_factory()
        with call_state.lock:
            if not call_state.called:
                call_state.return_value = _iterator_wrappers.GeneratorWrapper(
                    self._retry_exceptions, self._func, *args, **kwargs
                )
                call_state.called = True
            iterator = call_state.return_value
        yield from iterator.yield_results()


class OnceCallableAsyncFunction(OnceCallableAsyncBase):
    async def __call__(self, *args, **kwargs):
        call_state = self._state_factory()
        async with call_state.async_lock:
            if not call_state.called:
                try:
                    call_state.return_value = await self._func(*args, **kwargs)
                except Exception as exception:
                    if self._retry_exceptions:
                        raise exception
                    call_state.return_value = _CachedException(exception)
                call_state.called = True
            return_value = call_state.return_value
        if isinstance(return_value, _CachedException):
            raise return_value.exception
        return return_value


class OnceCallableAsyncGenerator(OnceCallableAsyncBase):
    async def __call__(self, *args, **kwargs):
        call_state = self._state_factory()
        async with call_state.async_lock:
            if not call_state.called:
                call_state.return_value = _iterator_wrappers.AsyncGeneratorWrapper(
                    self._retry_exceptions, self._func, *args, **kwargs
                )
                call_state.called = True
            return_value = call_state.return_value
        next_value = None
        iterator = return_value.yield_results()
        while True:
            try:
                next_value = yield await iterator.asend(next_value)
            except StopAsyncIteration:
                return
