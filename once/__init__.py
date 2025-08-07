"""Utility for initialization ensuring functions are called only once."""

import abc
import asyncio
import collections.abc
import enum
import functools
import inspect
import threading
import typing
import weakref

from . import _iterator_wrappers

from typing import ParamSpec


def _is_method(func: collections.abc.Callable):
    """Determine if a function is a method on a class."""
    if isinstance(func, (classmethod, staticmethod, property)):
        return True
    sig = inspect.signature(func)
    return "self" in sig.parameters


class _WrappedFunctionType(enum.Enum):
    SYNC_FUNCTION = 0
    ASYNC_FUNCTION = 1
    SYNC_GENERATOR = 2
    ASYNC_GENERATOR = 3


_ASYNC_FN_TYPES = (_WrappedFunctionType.ASYNC_FUNCTION, _WrappedFunctionType.ASYNC_GENERATOR)


_P = ParamSpec("_P")


_R = typing.TypeVar("_R")


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
    def __init__(self, is_async: bool, allow_reset: bool = False) -> None:
        self.is_async = is_async
        # We are going to be extra pedantic about these next two variables only being read or set
        # with a lock by defining getters and setters which enforce that the lock is held. If this
        # was C++, we would use something like the ABSL_GUARDED_BY macro for compile-time checking
        # (https://github.com/abseil/abseil-cpp/blob/master/absl/base/thread_annotations.h), but
        # this is python :)
        self._called = False
        self._return_value: typing.Any = None
        self.allow_reset = allow_reset
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


_ONCE_FACTORY_TYPE = collections.abc.Callable[[], _OnceBase]


class _CachedException:
    def __init__(self, exception: Exception):
        self.exception = exception


def _wrap(
    func: collections.abc.Callable[_P, _R],
    once_factory: _ONCE_FACTORY_TYPE,
    fn_type: _WrappedFunctionType,
    retry_exceptions: bool,
) -> collections.abc.Callable[_P, _R]:
    """Generate a wrapped function appropriate to the function type.

    The once_factory lets us reuse logic for both per-thread and singleton.
    For a singleton, the factory always returns the same _OnceBase object, but
    for per thread, it would return a unique one for each thread.
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
                        retry_exceptions,
                        func,
                        *args,
                        allow_reset=once_base.allow_reset,
                        **kwargs,
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
                    try:
                        once_base.return_value = await func(*args, **kwargs)  # type: ignore
                    except Exception as exception:
                        if retry_exceptions:
                            raise exception
                        once_base.return_value = _CachedException(exception)
                    once_base.called = True
                return_value = once_base.return_value
            if isinstance(return_value, _CachedException):
                raise return_value.exception
            return return_value

    elif fn_type == _WrappedFunctionType.SYNC_FUNCTION:

        def wrapped(*args, **kwargs) -> typing.Any:
            once_base: _OnceBase = once_factory()
            with once_base.lock:
                if not once_base.called:
                    try:
                        once_base.return_value = func(*args, **kwargs)
                    except Exception as exception:
                        if retry_exceptions:
                            raise exception
                        once_base.return_value = _CachedException(exception)
                    once_base.called = True
                return_value = once_base.return_value
            if isinstance(return_value, _CachedException):
                raise return_value.exception
            return return_value

    elif fn_type == _WrappedFunctionType.SYNC_GENERATOR:

        def wrapped(*args, **kwargs) -> typing.Any:
            once_base: _OnceBase = once_factory()
            with once_base.lock:
                if not once_base.called:
                    once_base.return_value = _iterator_wrappers.GeneratorWrapper(
                        retry_exceptions,
                        func,
                        *args,
                        allow_reset=once_base.allow_reset,
                        **kwargs,
                    )
                    once_base.called = True
                iterator = once_base.return_value
            yield from iterator.yield_results()

    else:
        raise NotImplementedError()

    def reset() -> None:
        once_base: _OnceBase = once_factory()
        with once_base.lock:
            if not once_base.called:
                return
            if fn_type == _WrappedFunctionType.SYNC_GENERATOR:
                iterator = once_base.return_value
                with iterator.lock:
                    iterator.reset()
            else:
                once_base.called = False

    async def async_reset() -> None:
        once_base: _OnceBase = once_factory()
        async with once_base.async_lock:
            if not once_base.called:
                return
            if fn_type == _WrappedFunctionType.ASYNC_GENERATOR:
                iterator = once_base.return_value
                async with iterator.lock:
                    iterator.reset()
            else:
                once_base.called = False

    def not_allowed_reset():
        # This doesn't need to be awaitable even in the async case because it will
        # raise the error before an `await` has a chance to do anything.
        raise RuntimeError(
            f"reset() is not allowed to be called on onced function {func}.\n"
            "Did you mean to add `allow_reset=True` to your once.once() annotation?"
        )

    # No need for the lock here since we're the only thread that could be running,
    # since we haven't even finished wrapping the func yet.
    once_base: _OnceBase = once_factory()
    if not once_base.allow_reset:
        wrapped.reset = not_allowed_reset  # type: ignore
    else:
        if once_base.is_async:
            wrapped.reset = async_reset  # type: ignore
        else:
            wrapped.reset = reset  # type: ignore

    functools.update_wrapper(wrapped, func)
    return wrapped  # type: ignore


def _once_factory(is_async: bool, per_thread: bool, allow_reset: bool) -> _ONCE_FACTORY_TYPE:
    if not per_thread:
        singleton_once = _OnceBase(is_async, allow_reset=allow_reset)
        return lambda: singleton_once

    per_thread_onces = threading.local()

    def _get_once_per_thread():
        # Read then modify is thread-safe without a lock because each thread sees its own copy of
        # copy of `per_thread_onces` thanks to `threading.local`, and each thread cannot race with
        # itself!
        if once := getattr(per_thread_onces, "once", None):
            return once
        per_thread_onces.once = _OnceBase(is_async, allow_reset=allow_reset)
        return per_thread_onces.once

    return _get_once_per_thread


def once(
    *args: collections.abc.Callable[_P, _R],
    per_thread=False,
    retry_exceptions=False,
    allow_reset=False,
) -> collections.abc.Callable[_P, _R]:
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

    per_thread:
        If true, the decorated function should be allowed to run once-per-thread
        as opposed to once per process.
    retry_exceptions:
        If true, exceptions in the onced function will allow the function to be
        called again. Otherwise, the exceptions are cached and re-raised on
        subsequent executions.
    allow_reset:
        If true, the returned wrapped function will have an attribute
        `.reset(*args, **kwargs)` which will reset the cache to allow a
        rerun of the underlying callable. This only resets the cache in the
        same scope as it would have used otherwise, e.g. resetting a callable
        wrapped in once_per_instance will reset the cache only for that instance,
        once_per_thread only for that thread, etc.
    """
    if len(args) == 1:
        func: collections.abc.Callable = args[0]
    elif len(args) > 1:
        raise ValueError("Up to 1 argument expected.")
    else:
        # This trick lets this function be a decorator directly, or be called
        # to create a decorator.
        # Both @once and @once() will function correctly.
        return typing.cast(
            collections.abc.Callable[_P, _R],
            functools.partial(
                once,
                per_thread=per_thread,
                retry_exceptions=retry_exceptions,
                allow_reset=allow_reset,
            ),
        )
    if _is_method(func):
        raise SyntaxError(
            "Attempting to use @once.once decorator on method "
            "instead of @once.once_per_class or @once.once_per_instance"
        )
    fn_type = _wrapped_function_type(func)
    once_factory = _once_factory(
        is_async=fn_type in _ASYNC_FN_TYPES,
        per_thread=per_thread,
        allow_reset=allow_reset,
    )
    return _wrap(func, once_factory, fn_type, retry_exceptions)


class once_per_class(typing.Generic[_P, _R]):  # pylint: disable=invalid-name
    """A version of once for class methods which runs once across all instances."""

    is_classmethod: bool
    is_staticmethod: bool
    func: collections.abc.Callable[_P, _R]

    @classmethod
    def with_options(cls, per_thread: bool = False, retry_exceptions=False, allow_reset=False):
        return lambda func: cls(
            func,
            per_thread=per_thread,
            retry_exceptions=retry_exceptions,
            allow_reset=allow_reset,
        )

    def __init__(
        self,
        func: collections.abc.Callable[_P, _R],
        per_thread: bool = False,
        retry_exceptions: bool = False,
        allow_reset: bool = False,
    ) -> None:
        self.func = self._inspect_function(func)
        self.fn_type = _wrapped_function_type(self.func)
        self.once_factory = _once_factory(
            is_async=self.fn_type in _ASYNC_FN_TYPES,
            per_thread=per_thread,
            allow_reset=allow_reset,
        )
        self.retry_exceptions = retry_exceptions

    def _inspect_function(
        self, func: collections.abc.Callable[_P, _R]
    ) -> collections.abc.Callable[_P, _R]:
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
    def __get__(self, obj, cls) -> collections.abc.Callable[_P, _R]:
        func = self.func
        if self.is_classmethod:
            func = functools.partial(self.func, cls)
        elif not self.is_staticmethod:
            func = functools.partial(self.func, obj)

        # Properly annotate the return type of _wrap to match Callable[P, R].
        return _wrap(func, self.once_factory, self.fn_type, self.retry_exceptions)


class once_per_instance(typing.Generic[_P, _R]):  # pylint: disable=invalid-name
    """A version of once for class methods which runs once per instance."""

    is_property: bool
    func: collections.abc.Callable[_P, _R]

    @classmethod
    def with_options(cls, per_thread: bool = False, retry_exceptions=False, allow_reset=False):
        return lambda func: cls(
            func, per_thread=per_thread, retry_exceptions=retry_exceptions, allow_reset=False
        )

    def __init__(
        self,
        func: collections.abc.Callable[_P, _R],
        per_thread: bool = False,
        retry_exceptions: bool = False,
        allow_reset: bool = False,
    ) -> None:
        self.func = self._inspect_function(func)
        self.fn_type = _wrapped_function_type(self.func)
        self.is_async_fn = self.fn_type in _ASYNC_FN_TYPES
        self.callables_lock = threading.Lock()
        self.callables: weakref.WeakKeyDictionary[typing.Any, collections.abc.Callable[_P, _R]] = (
            weakref.WeakKeyDictionary()
        )
        self.per_thread = per_thread
        self.retry_exceptions = retry_exceptions
        self.allow_reset = allow_reset

    def once_factory(self) -> _ONCE_FACTORY_TYPE:
        """Generate a new once factory.

        A once factory factory if you will.
        """
        return _once_factory(
            self.is_async_fn, per_thread=self.per_thread, allow_reset=self.allow_reset
        )

    def _inspect_function(
        self, func: collections.abc.Callable[_P, _R]
    ) -> collections.abc.Callable[_P, _R]:
        if isinstance(func, (classmethod, staticmethod)):
            raise SyntaxError("Must use @once.once_per_class on classmethod and staticmethod")
        if isinstance(func, property):
            func = func.fget
            self.is_property = True
        else:
            self.is_property = False
        if not _is_method(func):
            raise SyntaxError(
                "Attempting to use @once.once_per_instance method-only decorator "
                "instead of @once.once"
            )
        return func

    # This is needed for a decorator on a class method to return a
    # bound version of the function to the object.
    def __get__(self, obj, cls) -> collections.abc.Callable[_P, _R]:
        del cls
        if obj is None:
            # Requesting an unbound verion, so we return the function without
            # an object bound. The weakref lookup below would fail anyways,
            # and this will at least allow inspection of the function.
            # TODO: Give a better error message if the returned function is
            # called.
            return self.func
        with self.callables_lock:
            if (callable := self.callables.get(obj)) is None:
                bound_func = functools.partial(self.func, obj)
                callable = _wrap(
                    bound_func, self.once_factory(), self.fn_type, self.retry_exceptions
                )
                self.callables[obj] = callable
        if self.is_property:
            return callable()  # type: ignore
        return callable
