"""Utility for initialization ensuring functions are called only once."""
import asyncio
import collections.abc
import enum
import functools
import inspect
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
    SYNC_FUNCTION = 0
    ASYNC_FUNCTION = 1
    SYNC_GENERATOR = 2
    ASYNC_GENERATOR = 3


def _wrapped_function_type(func: collections.abc.Callable) -> _WrappedFunctionType:
    # The function inspect.isawaitable is a bit of a misnomer - it refers
    # to the awaitable result of an async function, not the async function
    # itself.
    original_func = func
    while isinstance(func, functools.partial):
        # Work around inspect not functioning properly in python < 3.10 for partial functions.
        func = func.func
    if inspect.isasyncgenfunction(func):
        return _WrappedFunctionType.ASYNC_GENERATOR
    if inspect.isgeneratorfunction(func):
        return _WrappedFunctionType.SYNC_GENERATOR
    if inspect.iscoroutinefunction(func):
        return _WrappedFunctionType.ASYNC_FUNCTION
    if inspect.isfunction(func):
        return _WrappedFunctionType.SYNC_FUNCTION
    raise SyntaxError(f"Unable to determine function type for {repr(original_func)}")


class _OnceBase:
    def __init__(self, fn_type: _WrappedFunctionType) -> None:
        self.called = False
        self.return_value: typing.Any = None
        self.fn_type = fn_type
        if (
            self.fn_type == _WrappedFunctionType.ASYNC_FUNCTION
            or self.fn_type == _WrappedFunctionType.ASYNC_GENERATOR
        ):
            self.async_lock = asyncio.Lock()
        else:
            self.lock = threading.Lock()

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
            if not self.called:
                self.return_value = await func(*args, **kwargs)
                self.called = True
            return self.return_value

    async def _execute_call_once_async_iter(self, func: collections.abc.Callable, *args, **kwargs):
        async with self.async_lock:
            if not self.called:
                self.return_value = _iterator_wrappers.AsyncGeneratorWrapper(func, *args, **kwargs)
                self.called = True
        next_value = None
        iterator = self.return_value.yield_results()
        while True:
            try:
                next_value = yield await iterator.asend(next_value)
            except StopAsyncIteration:
                return

    def _execute_call_once_sync(self, func: collections.abc.Callable, *args, **kwargs):
        with self.lock:
            if not self.called:
                self.return_value = func(*args, **kwargs)
                self.called = True
            return self.return_value

    def _execute_call_once_sync_iter(self, func: collections.abc.Callable, *args, **kwargs):
        with self.lock:
            if not self.called:
                self.return_value = _iterator_wrappers.GeneratorWrapper(func, *args, **kwargs)
                self.called = True
            iterator = self.return_value
        yield from iterator.yield_results()


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
    if _is_method(func):
        raise SyntaxError(
            "Attempting to use @once.once decorator on method "
            "instead of @once.once_per_class or @once.once_per_instance"
        )
    once_obj = _OnceBase(_wrapped_function_type(func))
    return once_obj._callable(func)


class once_per_class(_OnceBase):  # pylint: disable=invalid-name
    """A version of once for class methods which runs once across all instances."""

    is_classmethod: bool
    is_staticmethod: bool

    def __init__(self, func: collections.abc.Callable) -> None:
        self.func = self._inspect_function(func)
        super().__init__(_wrapped_function_type(self.func))

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
        self.func = self._inspect_function(func)
        super().__init__(_wrapped_function_type(self.func))
        self.callables_lock = threading.Lock()
        self.callables: weakref.WeakKeyDictionary[
            typing.Any, collections.abc.Callable
        ] = weakref.WeakKeyDictionary()

    def _inspect_function(self, func: collections.abc.Callable):
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
        with self.callables_lock:
            if callable := self.callables.get(obj):
                return callable
            once_obj = _OnceBase(self.fn_type)
            callable = once_obj._callable(functools.partial(self.func, obj))
            self.callables[obj] = callable
        return callable
