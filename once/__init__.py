"""Utility for initialization ensuring functions are called only once."""
import abc
import collections.abc
import functools
import inspect
import threading
import typing
import weakref


def _new_lock() -> threading.Lock:
    return threading.Lock()


def _is_method(func: collections.abc.Callable):
    """Determine if a function is a method on a class."""
    if isinstance(func, (classmethod, staticmethod)):
        return True
    sig = inspect.signature(func)
    return "self" in sig.parameters


class _OnceBase(abc.ABC):
    """Abstract Base Class for once function decorators."""

    def __init__(self, func: collections.abc.Callable):
        self._inspect_function(func)
        functools.update_wrapper(self, func)
        self.lock = _new_lock()
        self.called = False
        self.return_value: typing.Any = None
        self.func = func

    @abc.abstractmethod
    def _inspect_function(self, func: collections.abc.Callable):
        """Inspect the passed-in function to ensure it can be wrapped.

        This function should raise a SyntaxError if the passed-in function is
        not suitable."""

    def _execute_call_once(self, func: collections.abc.Callable, *args, **kwargs):
        if self.called:
            return self.return_value
        with self.lock:
            if self.called:
                return self.return_value
            self.return_value = func(*args, **kwargs)
            self.called = True
            return self.return_value


class once(_OnceBase):  # pylint: disable=invalid-name
    """Decorator to ensure a function is only called once.

    The restriction of only one call also holds across threads. However, this
    restriction does not apply to unsuccessful function calls. If the function
    raises an exception, the next call will invoke a new call to the function.
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

    def _inspect_function(self, func: collections.abc.Callable):
        if _is_method(func):
            raise SyntaxError(
                "Attempting to use @once.once decorator on method "
                "instead of @once.once_per_class or @once.once_per_instance"
            )

    def __call__(self, *args, **kwargs):
        return self._execute_call_once(self.func, *args, **kwargs)


class once_per_class(_OnceBase):  # pylint: disable=invalid-name
    """A version of once for class methods which runs once across all instances."""

    def _inspect_function(self, func):
        if not _is_method(func):
            raise SyntaxError(
                "Attempting to use @once.once_per_class method-only decorator "
                "instead of @once.once"
            )

    # This is needed for a decorator on a class method to return a
    # bound version of the function to the object or class.
    def __get__(self, obj, cls):
        if isinstance(self.func, classmethod):
            func = functools.partial(self.func.__func__, cls)
            return functools.partial(self._execute_call_once, func)
        if isinstance(self.func, staticmethod):
            # The additional __func__ is required for python <= 3.9
            return functools.partial(self._execute_call_once, self.func.__func__)
        return functools.partial(self._execute_call_once, self.func, obj)


class once_per_instance(_OnceBase):  # pylint: disable=invalid-name
    """A version of once for class methods which runs once per instance."""

    def __init__(self, func: collections.abc.Callable):
        super().__init__(func)
        self.return_value: weakref.WeakKeyDictionary[
            typing.Any, typing.Any
        ] = weakref.WeakKeyDictionary()
        self.inflight_lock: typing.Dict[typing.Any, threading.Lock] = {}

    def _inspect_function(self, func: collections.abc.Callable):
        if isinstance(func, (classmethod, staticmethod)):
            raise SyntaxError("Must use @once.once_per_class on classmethod and staticmethod")
        if not _is_method(func):
            raise SyntaxError(
                "Attempting to use @once.once_per_instance method-only decorator "
                "instead of @once.once"
            )

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
                inflight_lock = _new_lock()
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
