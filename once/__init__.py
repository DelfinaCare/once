"""Utility for initialization ensuring functions are called only once."""
import abc
import asyncio
import collections.abc
import enum
import functools
import inspect
import sys
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


_ASYNC_FN_TYPES = (_WrappedFunctionType.ASYNC_FUNCTION, _WrappedFunctionType.ASYNC_GENERATOR)


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

    def locked(self) -> bool:
        return self.async_lock.locked() if self.is_async else self.lock.locked()

    @property
    def called(self) -> bool:
        assert self.locked()
        return self._called

    @called.setter
    def called(self, state: bool) -> None:
        assert self.locked()
        self._called = state

    @property
    def return_value(self) -> typing.Any:
        assert self.locked()
        return self._return_value

    @return_value.setter
    def return_value(self, value: typing.Any) -> None:
        assert self.locked()
        self._return_value = value


if sys.version_info.minor > 8:
    _ONCE_FACTORY_TYPE = collections.abc.Callable[[], _OnceBase]
else:
    _ONCE_FACTORY_TYPE = collections.abc.Callable  # type: ignore


def _wrap(
    func: collections.abc.Callable,
    once_factory: _ONCE_FACTORY_TYPE,
    fn_type: _WrappedFunctionType,
) -> collections.abc.Callable:
    """Generate a wrapped function appropriate to the function type.

    The once_factory lets us reuse logic for both once and once_per_thread.
    For once, the factory always returns the same _OnceBase object, but for
    once_per_thread, it would return a unique one for each thread.
    """
    # Theoretically, we could compute fn_type now. However, this code may be executed at runtime
    # OR at definition time (due to once_per_instance), and we want to only be doing reflection at
    # definition time, so we force the caller to pass it in. But, if we're in debug mode, why not
    # check it again?
    assert fn_type == _wrapped_function_type(func)
    wrapped: collections.abc.Callable
    if fn_type == _WrappedFunctionType.ASYNC_GENERATOR:

        async def wrapped(*args, **kwargs) -> typing.Any:
            once_base: _OnceBase = once_factory()
            async with once_base.async_lock:
                if not once_base.called:
                    once_base.return_value = _iterator_wrappers.AsyncGeneratorWrapper(
                        func, *args, **kwargs
                    )
                    once_base.called = True
                return_value = once_base.return_value
            next_value = None
            iterator = return_value.yield_results()
            while True:
                try:
                    next_value = yield await iterator.asend(next_value)
                except StopAsyncIteration:
                    return

    elif fn_type == _WrappedFunctionType.ASYNC_FUNCTION:

        async def wrapped(*args, **kwargs) -> typing.Any:
            once_base: _OnceBase = once_factory()
            async with once_base.async_lock:
                if not once_base.called:
                    once_base.return_value = await func(*args, **kwargs)
                    once_base.called = True
                return once_base.return_value

    elif fn_type == _WrappedFunctionType.SYNC_FUNCTION:

        def wrapped(*args, **kwargs) -> typing.Any:
            once_base: _OnceBase = once_factory()
            with once_base.lock:
                if not once_base.called:
                    once_base.return_value = func(*args, **kwargs)
                    once_base.called = True
                return once_base.return_value

    elif fn_type == _WrappedFunctionType.SYNC_GENERATOR:

        def wrapped(*args, **kwargs) -> typing.Any:
            once_base: _OnceBase = once_factory()
            with once_base.lock:
                if not once_base.called:
                    once_base.return_value = _iterator_wrappers.GeneratorWrapper(
                        func, *args, **kwargs
                    )
                    once_base.called = True
                iterator = once_base.return_value
            yield from iterator.yield_results()

    else:
        raise NotImplementedError()

    functools.update_wrapper(wrapped, func)
    return wrapped


class _PerItemOnceMap:
    def __init__(self, is_async: bool) -> None:
        self.is_async = is_async
        self.per_item_onces: weakref.WeakKeyDictionary[
            typing.Any, _OnceBase
        ] = weakref.WeakKeyDictionary()
        self.lock = threading.Lock()

    def __getitem__(self, obj: typing.Any) -> _OnceBase:
        with self.lock:
            if (obj_once := self.per_item_onces.get(obj)) is None:
                obj_once = _OnceBase(self.is_async)
                self.per_item_onces[obj] = obj_once
        return obj_once


def _once_factory(is_async: bool, per_thread: bool) -> _ONCE_FACTORY_TYPE:
    if per_thread:
        per_thread_onces = _PerItemOnceMap(is_async)
        return lambda: per_thread_onces[threading.current_thread()]
    singleton_once = _OnceBase(is_async)
    return lambda: singleton_once


def once(func: collections.abc.Callable):
    """Decorator to ensure a function is only called once.

    The restriction of only one call also holds across threads. However, this
    restriction does not apply to unsuccessful function calls. If the function
    raises an exception, the next call will invoke a new call to the function,
    unless it is a generator, in which case new iterators will invoke a new
    call to the function, but existing iterators will continue and all raise
    the same cached value.

    Caching is **not** argument aware, and a subsequent call with different
    arguments after a function all will not result in a new call.

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
    fn_type = _wrapped_function_type(func)
    once_factory = _once_factory(is_async=fn_type in _ASYNC_FN_TYPES, per_thread=False)
    return _wrap(func, once_factory, fn_type)


def once_per_thread(func: collections.abc.Callable):
    """A version of once which executes only once per thread."""
    fn_type = _wrapped_function_type(func)
    once_factory = _once_factory(is_async=fn_type in _ASYNC_FN_TYPES, per_thread=True)
    return _wrap(func, once_factory, fn_type)


class once_per_class:  # pylint: disable=invalid-name
    """A version of once for class methods which runs once across all instances."""

    is_classmethod: bool
    is_staticmethod: bool

    def __init__(self, func: collections.abc.Callable) -> None:
        self.func = self._inspect_function(func)
        self.fn_type = _wrapped_function_type(self.func)
        self.once_factory = _once_factory(
            is_async=self.fn_type in _ASYNC_FN_TYPES, per_thread=False
        )

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
    def __get__(self, obj, cls) -> collections.abc.Callable:
        if self.is_classmethod:
            func = functools.partial(self.func, cls)
        elif self.is_staticmethod:
            func = self.func
        else:
            func = functools.partial(self.func, obj)
        return _wrap(func, self.once_factory, self.fn_type)


class once_per_class_per_thread(once_per_class):  # pylint: disable=invalid-name
    def __init__(self, func: collections.abc.Callable) -> None:
        super().__init__(func)
        self.once_factory = _once_factory(self.fn_type in _ASYNC_FN_TYPES, per_thread=True)


class _OncePerInstanceBase(abc.ABC):
    def __init__(self, func: collections.abc.Callable) -> None:
        self.func = self._inspect_function(func)
        self.fn_type = _wrapped_function_type(self.func)
        self.is_async_fn = self.fn_type in _ASYNC_FN_TYPES
        self.callables_lock = threading.Lock()
        self.callables: weakref.WeakKeyDictionary[
            typing.Any, collections.abc.Callable
        ] = weakref.WeakKeyDictionary()

    @abc.abstractmethod
    def once_factory(self) -> _ONCE_FACTORY_TYPE:
        """Generate a new once factory.

        A once factory factory if you will.
        """
        raise NotImplementedError()

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
    def __get__(self, obj, cls) -> collections.abc.Callable:
        del cls
        with self.callables_lock:
            if (callable := self.callables.get(obj)) is None:
                bound_func = functools.partial(self.func, obj)
                callable = _wrap(bound_func, self.once_factory(), self.fn_type)
                self.callables[obj] = callable
        return callable


class once_per_instance(_OncePerInstanceBase):  # pylint: disable=invalid-name
    """A version of once for class methods which runs once per instance."""

    def once_factory(self):
        return _once_factory(self.is_async_fn, per_thread=False)


class once_per_instance_per_thread(_OncePerInstanceBase):  # pylint: disable=invalid-name
    """A version of once for class methods which runs once per instance per thread."""

    def once_factory(self):
        return _once_factory(self.is_async_fn, per_thread=True)
