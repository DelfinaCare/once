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
and all subsequent calls will return the same result. Enough talking, let's
cut to an example:

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

```

This module is extremely simple, with no external dependencies, and heavily
tested for races.
