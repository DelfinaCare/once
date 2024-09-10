# Once

This library provides functionality to ensure a function is called exactly
once in Python, heavily inspired by `std::call_once`.

During initialization, we often want to ensure code is run **exactly once**.
But thinking about all the different ways this constraint can be violated can
be time-consuming and complex. We don't want to have to reason about what other
callers are doing and from which thread.

Introducing a simple solution - the `once.once` decorator! Simply decorate a
function with this decorator, and this library will handle all the edge cases
to ensure it is called exactly once! The first call will invoke the function,
and all subsequent calls will return the same result. It works on both sync
and aysnc functions, and methods. Enough talking, let's cut to an example:

```python
import once

@once.once
def my_expensive_object():
    load_expensive_resource()
    load_more_expensive_resources()
    return ObjectSingletonUsingLotsOfMemory()

def caller_one():
    my_expensive_object().use_it()

def caller_two_from_a_separate_thread():
    my_expensive_object().use_it()

def optional_init_function_to_prewarm():
    my_expensive_object()

@once.once
async def slow_expensive_object():
    await load_expensive_async_resource()

@once.once(per_thread=True)
async def slow_expensive_non_threadsafe_object():
    await load_expensive_async_resource()

async def user_slow_expensive_object():
    await slow_expensive_object()

```

This module is extremely simple, with no external dependencies, and heavily
tested for races.

## Use with methods
There are two versions of the decorator for methods, `once_per_class` and
`once_per_instance`. The `once_per_class` decorator calls the function only
only once for the defined class, and the `once_per_instance` decorator once
for each separate object instance created from the class.
```python
class MyClass:

    @once.once_per_class
    def fn1(self):
        pass

    @once.once_per_instance
    @classmethod
    async def fn2(cls):
        pass

A = MyClass()
B = MyClass()

A.fn1()
B.fn1()

A.fn2()  # cached
B.fn2()  # calls again
B.fn2()  # cached
```

## Options
The behavior of the decorator is configurable with boolean options, which
default to `False`. For the function decorator, options can be specified by
simply passing them into the decorator:
```python
@once.once(per_thread=True)
def non_thread_safe_constructor():
    pass
```

For methods, pass options into the `with_options` modifier
```python

class MyClass:
    @once.once_per_class.with_options(per_thread=True)
    def non_thread_safe_constructor(self):
        pass
```

### `per_thread`
This instantiates the function once per thread, and will return a thread-local
result for each separate thread. This is extremely convenient for expensive
objects which are not thread-safe.

### `allow_reset`
This exposes a `reset` method on the function, which will force the underlying
function to be called again.
```python
@once.once(allow_reset=True)
def resettable_fn():
    pass

resetable_fn()
resetable_fn.reset()
resetable_fn()  # calls again
```

### `retry_exceptions`
This will invoke the underlying function again if it raises an unhandled
exception.
