"""Utility for initialization ensuring functions are called only once."""
import abc
import asyncio
import collections.abc
import enum
import functools
import inspect
import time
import threading
import typing
import weakref

from . import _iterator_wrappers


def _is_method(func: collections.abc.Callable):
    """Determine if a function is a method on a class."""
    if isinstance(func, (classmethod, staticmethod)):
        return True
    sig = inspect.signature(func)
    return "self" in sig.parameters


class _WrappedFunctionType(enum.Enum):
    UNSUPPORTED = 0
    SYNC_FUNCTION = 1
    ASYNC_FUNCTION = 2
    SYNC_GENERATOR = 3
    ASYNC_GENERATOR = 4


def _wrapped_function_type(func: collections.abc.Callable) -> _WrappedFunctionType:
    # The function inspect.isawaitable is a bit of a misnomer - it refers
    # to the awaitable result of an async function, not the async function
    # itself.
    while isinstance(func, functools.partial):
        # Work around inspect not functioning properly in python < 3.10 for partial functions.
        func = func.func
    if inspect.isasyncgenfunction(func):
        return _WrappedFunctionType.ASYNC_GENERATOR
    if inspect.isgeneratorfunction(func):
        return _WrappedFunctionType.SYNC_GENERATOR
    if inspect.iscoroutinefunction(func):
        return _WrappedFunctionType.ASYNC_FUNCTION
    # We assume it is a callable sync function if it is callable.
    if not callable(func):
        return _WrappedFunctionType.UNSUPPORTED
    return _WrappedFunctionType.SYNC_FUNCTION


class _ExecutionState(enum.Enum):
    UNCALLED = 0
    WAITING = 1
    COMPLETED = 2


class _OnceBase(abc.ABC):
    """Abstract Base Class for once function decorators."""

    def __init__(self, func: collections.abc.Callable) -> None:
        functools.update_wrapper(self, func)
        self.func = self._inspect_function(func)
        self.call_state = _ExecutionState.UNCALLED
        self.return_value: typing.Any = None
        self.fn_type = _wrapped_function_type(self.func)
        if self.fn_type == _WrappedFunctionType.UNSUPPORTED:
            raise SyntaxError(f"Unable to wrap a {type(func)}")
        if (
            self.fn_type == _WrappedFunctionType.ASYNC_FUNCTION
            or self.fn_type == _WrappedFunctionType.ASYNC_GENERATOR
        ):
            self.async_lock = asyncio.Lock()
        else:
            self.lock = threading.Lock()

    @abc.abstractmethod
    def _inspect_function(self, func: collections.abc.Callable) -> collections.abc.Callable:
        """Inspect the passed-in function to ensure it can be wrapped.

        This function should raise a SyntaxError if the passed-in function is
        not suitable.

        It should return the function which should be executed once.
        """

    def _callable(self, func: collections.abc.Callable):
        """Generate a wrapped function appropriate to the function type.

        This wrapped function will call the correct _execute_call_once function.
        """
        if self.fn_type == _WrappedFunctionType.ASYNC_GENERATOR:

            async def wrapped(*args, **kwargs):
                next_value = None
                iterator = self._execute_call_once_async_iter(func, *args, **kwargs)
                while True:
                    try:
                        next_value = yield await iterator.asend(next_value)
                    except StopAsyncIteration:
                        return

        elif self.fn_type == _WrappedFunctionType.ASYNC_FUNCTION:

            async def wrapped(*args, **kwargs):
                return await self._execute_call_once_async(func, *args, **kwargs)

        elif self.fn_type == _WrappedFunctionType.SYNC_FUNCTION:

            def wrapped(*args, **kwargs):
                return self._execute_call_once_sync(func, *args, **kwargs)

        else:
            assert self.fn_type == _WrappedFunctionType.SYNC_GENERATOR

            def wrapped(*args, **kwargs):
                yield from self._execute_call_once_sync_iter(func, *args, **kwargs)

        functools.update_wrapper(wrapped, func)
        return wrapped

    async def _execute_call_once_async(self, func: collections.abc.Callable, *args, **kwargs):
        async with self.async_lock:
            call_state = self.call_state
        while call_state != _ExecutionState.COMPLETED:
            if call_state == _ExecutionState.WAITING:
                # Allow another thread to grab the GIL.
                await asyncio.sleep(0)
            async with self.async_lock:
                call_state = self.call_state
                if call_state == _ExecutionState.UNCALLED:
                    self.call_state = _ExecutionState.WAITING
            # Only one thread will be allowed into this state.
            if call_state == _ExecutionState.UNCALLED:
                try:
                    return_value = await func(*args, **kwargs)
                except Exception as exc:
                    async with self.async_lock:
                        self.call_state = _ExecutionState.UNCALLED
                    raise exc
                async with self.async_lock:
                    self.return_value = return_value
                    self.call_state = _ExecutionState.COMPLETED
        return self.return_value

    async def _execute_call_once_async_iter(self, func: collections.abc.Callable, *args, **kwargs):
        async with self.async_lock:
            if self.call_state == _ExecutionState.UNCALLED:
                self.return_value = _iterator_wrappers.AsyncGeneratorWrapper(func, *args, **kwargs)
                self.call_state = _ExecutionState.COMPLETED
        next_value = None
        iterator = self.return_value.yield_results()
        while True:
            try:
                next_value = yield await iterator.asend(next_value)
            except StopAsyncIteration:
                return

    def _execute_call_once_sync(self, func: collections.abc.Callable, *args, **kwargs):
        with self.lock:
            call_state = self.call_state
        while call_state != _ExecutionState.COMPLETED:
            # We only hit this state in multi-threded code. To reduce contention, we invoke
            # time.sleep so another thread an pick up the GIL.
            if call_state == _ExecutionState.WAITING:
                time.sleep(0)
            with self.lock:
                call_state = self.call_state
                if call_state == _ExecutionState.UNCALLED:
                    self.call_state = _ExecutionState.WAITING
            # Only one thread will be allowed into this state.
            if call_state == _ExecutionState.UNCALLED:
                try:
                    return_value = func(*args, **kwargs)
                except Exception as exc:
                    with self.lock:
                        self.call_state = _ExecutionState.UNCALLED
                    raise exc
                else:
                    with self.lock:
                        self.return_value = return_value
                        self.call_state = _ExecutionState.COMPLETED
        return self.return_value

    def _execute_call_once_sync_iter(self, func: collections.abc.Callable, *args, **kwargs):
        with self.lock:
            if self.call_state == _ExecutionState.UNCALLED:
                self.return_value = _iterator_wrappers.GeneratorWrapper(func, *args, **kwargs)
                self.call_state = _ExecutionState.COMPLETED
        yield from self.return_value.yield_results()


class _OnceFn(_OnceBase):
    def _inspect_function(self, func: collections.abc.Callable):
        if _is_method(func):
            raise SyntaxError(
                "Attempting to use @once.once decorator on method "
                "instead of @once.once_per_class or @once.once_per_instance"
            )
        return func


def once(func: collections.abc.Callable):
    """Decorator to ensure a function is only called once.

    The restriction of only one call also holds across threads. However, this
    restriction does not apply to unsuccessful function calls. If the function
    raises an exception, the next call will invoke a new call to the function,
    unless it is in iterator, in which case the failure will be cached.
    If the function is called with multiple arguments, it will still only be
    called only once.

    This decorator will fail for methods defined on a class. Use
    once_per_class or once_per_instance for methods on a class instead.

    Please note that because the value returned by the decorated function is
    stored to return for subsequent calls, it will not be eligible for garbage
    collection until after the decorated function itself has been deleted. For
    module and class level functions (i.e. non-closures), this means the return
    value will never be deleted.
    """
    once_obj = _OnceFn(func)
    return once_obj._callable(func)


class once_per_class(_OnceBase):  # pylint: disable=invalid-name
    """A version of once for class methods which runs once across all instances."""

    is_classmethod: bool
    is_staticmethod: bool

    def _inspect_function(self, func: collections.abc.Callable):
        if not _is_method(func):
            raise SyntaxError(
                "Attempting to use @once.once_per_class method-only decorator "
                "instead of @once.once"
            )
        if isinstance(func, classmethod):
            self.is_classmethod = True
            self.is_staticmethod = False
            return func.__func__
        if isinstance(func, staticmethod):
            self.is_classmethod = False
            self.is_staticmethod = True
            return func.__func__
        self.is_classmethod = False
        self.is_staticmethod = False
        return func

    # This is needed for a decorator on a class method to return a
    # bound version of the function to the object or class.
    def __get__(self, obj, cls):
        if self.is_classmethod:
            return self._callable(functools.partial(self.func, cls))
        if self.is_staticmethod:
            return self._callable(self.func)
        return self._callable(functools.partial(self.func, obj))


class once_per_instance(_OnceBase):  # pylint: disable=invalid-name
    """A version of once for class methods which runs once per instance."""

    def __init__(self, func: collections.abc.Callable) -> None:
        super().__init__(func)
        self.return_value: weakref.WeakKeyDictionary[
            typing.Any, typing.Any
        ] = weakref.WeakKeyDictionary()
        self.inflight_lock: typing.Dict[typing.Any, threading.Lock] = {}

    def _inspect_function(self, func: collections.abc.Callable):
        if inspect.isasyncgenfunction(func) or inspect.iscoroutinefunction(func):
            raise SyntaxError("once_per_instance not (yet) supported for async")
        if isinstance(func, (classmethod, staticmethod)):
            raise SyntaxError("Must use @once.once_per_class on classmethod and staticmethod")
        if not _is_method(func):
            raise SyntaxError(
                "Attempting to use @once.once_per_instance method-only decorator "
                "instead of @once.once"
            )
        return func

    # This is needed for a decorator on a class method to return a
    # bound version of the function to the object.
    def __get__(self, obj, cls):
        del cls
        return functools.partial(self._execute_call_once_per_instance, obj)

    def _execute_call_once_per_instance(self, obj, *args, **kwargs):
        # We only append to the call history, and do not overwrite or remove keys.
        # Therefore, we can check the call history without a lock for an early
        # exit.
        # Another concern might be the weakref dictionary for return_value
        # getting garbage collected without a lock. However, because
        # user_function references whichever key it matches, it cannot be
        # garbage collected during this call.
        if obj in self.return_value:
            return self.return_value[obj]
        with self.lock:
            if obj in self.return_value:
                return self.return_value[obj]
            if obj in self.inflight_lock:
                inflight_lock = self.inflight_lock[obj]
            else:
                inflight_lock = threading.Lock()
                self.inflight_lock[obj] = inflight_lock
        # Now we have a per-object lock. This means that we will not block
        # other instances. In addition to better performance, this reduces the
        # potential for deadlocks.
        with inflight_lock:
            if obj in self.return_value:
                return self.return_value[obj]
            result = self.func(obj, *args, **kwargs)
            self.return_value[obj] = result
            # At this point, any new call will find a cache hit before
            # even grabbing a lock. It is now safe to clean up the inflight
            # lock entry from the dictionary, as all subsequent will not need
            # it. Any other previously called inflight requests already have
            # their reference to the lock object, and do not need it present
            # in this dict either.
            self.inflight_lock.pop(obj)
            return result
