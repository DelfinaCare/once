"""Utility for initialization ensuring functions are called only once."""

import collections.abc
import enum
import functools
import inspect
import threading
import typing
import weakref

from . import _base
from . import _state

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


# Instead of just passing in a state, we generally use a state_factory
# function, which returns a state. This lets us implement a version which
# returns a unique state per thread to implement per_thread, or the same object
# for a globally unique once.
_STATE_FACTORY_TYPE = collections.abc.Callable[[], _state._CallState]


def _not_allow_reset():
    raise RuntimeError("function was not created with allow_reset flag.")


def _wrap(
    func: collections.abc.Callable[_P, _R],
    state_factory: _STATE_FACTORY_TYPE,
    fn_type: _WrappedFunctionType,
    retry_exceptions: bool,
    allow_reset: bool,
) -> collections.abc.Callable[_P, _R]:
    """Generate a wrapped function appropriate to the function type.

    The state_factory lets us reuse logic for both per-thread and singleton.
    For a singleton, the factory always returns the same _CallState object, but
    for per thread, it would return a unique one for each thread.
    """
    # Theoretically, we could compute fn_type now. However, this code may be executed at runtime
    # OR at definition time (due to once_per_instance), and we want to only be doing reflection at
    # definition time, so we force the caller to pass it in. But, if we're in debug mode, why not
    # check it again?
    assert fn_type == _wrapped_function_type(func)
    once_callable: _base.OnceCallableSyncBase | _base.OnceCallableAsyncBase
    if fn_type == _WrappedFunctionType.ASYNC_GENERATOR:
        once_callable = _base.OnceCallableAsyncGenerator(func, state_factory, retry_exceptions)
    elif fn_type == _WrappedFunctionType.ASYNC_FUNCTION:
        once_callable = _base.OnceCallableAsyncFunction(func, state_factory, retry_exceptions)
    elif fn_type == _WrappedFunctionType.SYNC_FUNCTION:
        once_callable = _base.OnceCallableSyncFunction(func, state_factory, retry_exceptions)
    elif fn_type == _WrappedFunctionType.SYNC_GENERATOR:
        once_callable = _base.OnceCallableSyncGenerator(func, state_factory, retry_exceptions)
    else:
        raise NotImplementedError()

    # We return the class which exposes the reset function only if resettable,
    # otherwise we just return the function.
    wrapped = functools.partial(once_callable.__class__.__call__, once_callable)  # type: ignore
    functools.update_wrapper(wrapped, func)
    if allow_reset:
        wrapped.reset = once_callable.reset  # type: ignore
    else:
        wrapped.reset = _not_allow_reset  # type: ignore
    return wrapped


def _state_factory(is_async: bool, per_thread: bool) -> _STATE_FACTORY_TYPE:
    if not per_thread:
        singleton_state = _state._CallState(is_async)
        return lambda: singleton_state

    per_thread_states = threading.local()

    def _get_once_per_thread():
        # Read then modify is thread-safe without a lock because each thread sees its own copy of
        # copy of `per_thread_states` thanks to `threading.local`, and each thread cannot race with
        # itself!
        if state := getattr(per_thread_states, "state", None):
            return state
        per_thread_states.state = _state._CallState(is_async)
        return per_thread_states.state

    return _get_once_per_thread


@typing.overload
def once(func: collections.abc.Callable[_P, _R], /) -> collections.abc.Callable[_P, _R]: ...


@typing.overload
def once(
    *, per_thread: bool = False, retry_exceptions: bool = False, allow_reset: bool = False
) -> collections.abc.Callable[
    [collections.abc.Callable[_P, _R]], collections.abc.Callable[_P, _R]
]: ...


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
    state_factory = _state_factory(is_async=fn_type in _ASYNC_FN_TYPES, per_thread=per_thread)
    return _wrap(func, state_factory, fn_type, retry_exceptions, allow_reset)


class once_per_class(typing.Generic[_P, _R]):  # pylint: disable=invalid-name
    """A version of once for class methods which runs once across all instances."""

    is_classmethod: bool
    is_staticmethod: bool
    func: collections.abc.Callable[_P, _R]

    @classmethod
    def with_options(cls, per_thread: bool = False, retry_exceptions=False, allow_reset=False):
        return lambda func: cls(
            func, per_thread=per_thread, retry_exceptions=retry_exceptions, allow_reset=allow_reset
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
        self.state_factory = _state_factory(
            is_async=self.fn_type in _ASYNC_FN_TYPES, per_thread=per_thread
        )
        self.retry_exceptions = retry_exceptions
        self.allow_reset = allow_reset

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

        return _wrap(
            func, self.state_factory, self.fn_type, self.retry_exceptions, self.allow_reset
        )


class once_per_instance(typing.Generic[_P, _R]):  # pylint: disable=invalid-name
    """A version of once for class methods which runs once per instance."""

    is_property: bool
    func: collections.abc.Callable[_P, _R]

    @classmethod
    def with_options(cls, per_thread: bool = False, retry_exceptions=False, allow_reset=False):
        return lambda func: cls(
            func, per_thread=per_thread, retry_exceptions=retry_exceptions, allow_reset=allow_reset
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

    def _state_factory(self) -> _STATE_FACTORY_TYPE:
        """Generate a new state factory.

        A state factory factory if you will.
        """
        return _state_factory(self.is_async_fn, per_thread=self.per_thread)

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
            if (bound_callable := self.callables.get(obj)) is None:
                bound_func = functools.partial(self.func, obj)
                bound_callable = _wrap(
                    bound_func,
                    self._state_factory(),
                    self.fn_type,
                    self.retry_exceptions,
                    self.allow_reset,
                )
                self.callables[obj] = bound_callable
        if self.is_property:
            # There is a type mismatch on the following line because we are not
            # passing in any arguments. However, for a property, no arguments
            # are passed into the function anyways. Therefore, we suppress the
            # error.
            return bound_callable()  # type: ignore
        return bound_callable
