"""Unit tests for once decorators."""
# pylint: disable=missing-function-docstring
import asyncio
import concurrent.futures
import functools
import gc
import inspect
import math
import sys
import threading
import time
import unittest
import weakref

import once


if sys.version_info.minor < 10:
    print(f"Redefining anext for python 3.{sys.version_info.minor}")

    async def anext(iter, default=StopAsyncIteration):
        if default != StopAsyncIteration:
            try:
                return await iter.__anext__()
            except StopAsyncIteration:
                return default
        return await iter.__anext__()


# This is a "large" number of workers to schedule function calls in parallel.
_N_WORKERS = 16


class Counter:
    """Holding object for a counter.

    If we return an integer directly, it will simply return a copy and
    will not update as the number of calls increases.

    The counter can also be paused by clearing its ready attribute, which will be convenient to
    start multiple runs to execute concurrently.
    """

    def __init__(self) -> None:
        self.value = 0
        self.ready = threading.Event()
        self.ready.set()

    def get_incremented(self) -> int:
        self.ready.wait()
        self.value += 1
        return self.value


def execute_with_barrier(*args, n_workers=None):
    """Decorator to ensure function calls do not begin until at least n_workers have started.

    This ensures that our parallel tests actually test concurrency. Without this, it is possible
    that function calls execute as they are being scheduled, and do not truly execute in parallel.

    The decorated function should receive an integer multiple of n_workers invokations.
    """
    # Trick to make the decorator accept an arugment. The first call only gets the n_workers
    # parameter, and then returns a new function with it set that then accepts the function.
    if n_workers is None:
        raise ValueError("n_workers not set")
    if len(args) == 0:
        return functools.partial(execute_with_barrier, n_workers=n_workers)
    if len(args) > 1:
        raise ValueError("Up to one argument expected.")
    func = args[0]
    barrier = threading.Barrier(n_workers)

    def wrapped(*args, **kwargs):
        barrier.wait()
        return func(*args, **kwargs)

    functools.update_wrapper(wrapped, func)
    return wrapped


def ensure_started(executor, func, *args, **kwargs):
    """Submit an execution to the executor and ensure it starts."""
    event = threading.Event()

    def run():
        event.set()
        return func(*args, **kwargs)

    future = executor.submit(run)
    event.wait()
    return future


def generate_once_counter_fn():
    """Generates a once.once decorated function which counts its calls."""

    counter = Counter()

    @once.once
    def counting_fn(*args) -> int:
        """Returns the call count, which should always be 1."""
        nonlocal counter
        del args
        return counter.get_incremented()

    return counting_fn, counter


class TestFunctionInspection(unittest.TestCase):
    """Unit tests for function inspection"""

    def sample_sync_method(self, _):
        return 1

    def test_sync_method(self):
        self.assertEqual(
            once._wrapped_function_type(TestFunctionInspection.sample_sync_method),
            once._WrappedFunctionType.SYNC_FUNCTION,
        )
        self.assertEqual(
            once._wrapped_function_type(
                functools.partial(TestFunctionInspection.sample_sync_method, 1)
            ),
            once._WrappedFunctionType.SYNC_FUNCTION,
        )

    def test_sync_function(self):
        def sample_sync_fn(_1, _2):
            return 1

        self.assertEqual(
            once._wrapped_function_type(sample_sync_fn), once._WrappedFunctionType.SYNC_FUNCTION
        )
        self.assertEqual(
            once._wrapped_function_type(once.once(sample_sync_fn)),
            once._WrappedFunctionType.SYNC_FUNCTION,
        )
        self.assertEqual(
            once._wrapped_function_type(functools.partial(sample_sync_fn, 1)),
            once._WrappedFunctionType.SYNC_FUNCTION,
        )
        self.assertEqual(
            once._wrapped_function_type(functools.partial(functools.partial(sample_sync_fn, 1), 2)),
            once._WrappedFunctionType.SYNC_FUNCTION,
        )
        self.assertEqual(
            once._wrapped_function_type(lambda x: x + 1), once._WrappedFunctionType.SYNC_FUNCTION
        )

    async def sample_async_method(self, _):
        return 1

    def test_async_method(self):
        self.assertEqual(
            once._wrapped_function_type(TestFunctionInspection.sample_async_method),
            once._WrappedFunctionType.ASYNC_FUNCTION,
        )
        self.assertEqual(
            once._wrapped_function_type(
                functools.partial(TestFunctionInspection.sample_async_method, 1)
            ),
            once._WrappedFunctionType.ASYNC_FUNCTION,
        )

    def test_async_function(self):
        async def sample_async_fn(_1, _2):
            return 1

        self.assertEqual(
            once._wrapped_function_type(sample_async_fn), once._WrappedFunctionType.ASYNC_FUNCTION
        )
        self.assertEqual(
            once._wrapped_function_type(once.once(sample_async_fn)),
            once._WrappedFunctionType.ASYNC_FUNCTION,
        )
        self.assertEqual(
            once._wrapped_function_type(functools.partial(sample_async_fn, 1)),
            once._WrappedFunctionType.ASYNC_FUNCTION,
        )
        self.assertEqual(
            once._wrapped_function_type(
                functools.partial(functools.partial(sample_async_fn, 1), 2)
            ),
            once._WrappedFunctionType.ASYNC_FUNCTION,
        )

    def sample_sync_generator_method(self, _):
        yield 1

    def test_sync_generator_method(self):
        self.assertEqual(
            once._wrapped_function_type(TestFunctionInspection.sample_sync_generator_method),
            once._WrappedFunctionType.SYNC_GENERATOR,
        )
        self.assertEqual(
            once._wrapped_function_type(
                functools.partial(TestFunctionInspection.sample_sync_generator_method, 1)
            ),
            once._WrappedFunctionType.SYNC_GENERATOR,
        )

    def test_sync_generator_function(self):
        def sample_sync_generator_fn(_1, _2):
            yield 1

        self.assertEqual(
            once._wrapped_function_type(sample_sync_generator_fn),
            once._WrappedFunctionType.SYNC_GENERATOR,
        )
        self.assertEqual(
            once._wrapped_function_type(once.once(sample_sync_generator_fn)),
            once._WrappedFunctionType.SYNC_GENERATOR,
        )
        self.assertEqual(
            once._wrapped_function_type(functools.partial(sample_sync_generator_fn, 1)),
            once._WrappedFunctionType.SYNC_GENERATOR,
        )
        self.assertEqual(
            once._wrapped_function_type(
                functools.partial(functools.partial(sample_sync_generator_fn, 1), 2)
            ),
            once._WrappedFunctionType.SYNC_GENERATOR,
        )
        # The output of a sync generator is not a wrappable.
        with self.assertRaises(SyntaxError):
            once._wrapped_function_type(sample_sync_generator_fn(1, 2))

    async def sample_async_generator_method(self, _):
        yield 1

    def test_async_generator_method(self):
        self.assertEqual(
            once._wrapped_function_type(TestFunctionInspection.sample_async_generator_method),
            once._WrappedFunctionType.ASYNC_GENERATOR,
        )
        self.assertEqual(
            once._wrapped_function_type(
                functools.partial(TestFunctionInspection.sample_async_generator_method, 1)
            ),
            once._WrappedFunctionType.ASYNC_GENERATOR,
        )

    def test_async_generator_function(self):
        async def sample_async_generator_fn(_1, _2):
            yield 1

        self.assertEqual(
            once._wrapped_function_type(sample_async_generator_fn),
            once._WrappedFunctionType.ASYNC_GENERATOR,
        )
        self.assertEqual(
            once._wrapped_function_type(once.once(sample_async_generator_fn)),
            once._WrappedFunctionType.ASYNC_GENERATOR,
        )
        self.assertEqual(
            once._wrapped_function_type(functools.partial(sample_async_generator_fn, 1)),
            once._WrappedFunctionType.ASYNC_GENERATOR,
        )
        self.assertEqual(
            once._wrapped_function_type(
                functools.partial(functools.partial(sample_async_generator_fn, 1))
            ),
            once._WrappedFunctionType.ASYNC_GENERATOR,
        )
        # The output of an async generator is not a wrappable.
        with self.assertRaises(SyntaxError):
            once._wrapped_function_type(sample_async_generator_fn(1, 2))


class TestOnce(unittest.TestCase):
    """Unit tests for once decorators."""

    def test_inspect_iterator(self):
        @once.once
        def yielding_iterator():
            for i in range(3):
                yield i

        self.assertTrue(inspect.isgeneratorfunction(yielding_iterator))

    def test_counter_works(self):
        """Ensure the counter text fixture works."""
        counter = Counter()
        self.assertEqual(counter.value, 0)
        self.assertEqual(counter.get_incremented(), 1)
        self.assertEqual(counter.value, 1)
        self.assertEqual(counter.get_incremented(), 2)
        self.assertEqual(counter.value, 2)

    def test_different_args_same_result(self):
        counting_fn, counter = generate_once_counter_fn()
        self.assertEqual(counting_fn(1), 1)
        self.assertEqual(counter.value, 1)
        # Should return the same result as the first call.
        self.assertEqual(counting_fn(2), 1)
        self.assertEqual(counter.value, 1)

    def test_partial(self):
        counter = Counter()
        func = once.once(functools.partial(lambda _: counter.get_incremented(), None))
        self.assertEqual(func(), 1)
        self.assertEqual(func(), 1)

    def test_failing_function(self):
        counter = Counter()

        @once.once
        def sample_failing_fn():
            if counter.get_incremented() < 4:
                raise ValueError("expected failure")
            return 1

        with self.assertRaises(ValueError):
            sample_failing_fn()
        self.assertEqual(counter.get_incremented(), 2)
        with self.assertRaises(ValueError):
            sample_failing_fn()
        # This ensures that this was a new function call, not a cached result.
        self.assertEqual(counter.get_incremented(), 4)
        self.assertEqual(sample_failing_fn(), 1)

    def test_iterator(self):
        counter = Counter()

        @once.once
        def yielding_iterator():
            nonlocal counter
            for _ in range(3):
                yield counter.get_incremented()

        self.assertEqual(list(yielding_iterator()), [1, 2, 3])
        self.assertEqual(list(yielding_iterator()), [1, 2, 3])

    def test_failing_generator(self):
        counter = Counter()

        @once.once
        def sample_failing_fn():
            yield counter.get_incremented()
            result = counter.get_incremented()
            yield result
            if result == 2:
                raise ValueError("expected failure after 2.")

        # Both of these calls should return the same results.
        call1 = sample_failing_fn()
        call2 = sample_failing_fn()
        self.assertEqual(next(call1), 1)
        self.assertEqual(next(call2), 1)
        self.assertEqual(next(call1), 2)
        self.assertEqual(next(call2), 2)
        with self.assertRaises(ValueError):
            next(call1)
        with self.assertRaises(ValueError):
            next(call2)
        # These next 2 calls should succeed.
        call3 = sample_failing_fn()
        call4 = sample_failing_fn()
        self.assertEqual(list(call3), [3, 4])
        self.assertEqual(list(call4), [3, 4])
        # Subsequent calls should return the original value.
        self.assertEqual(list(sample_failing_fn()), [3, 4])
        self.assertEqual(list(sample_failing_fn()), [3, 4])

    def test_iterator_parallel_execution(self):
        counter = Counter()

        # Must be called over an integer multiple of _N_WORKERS
        @execute_with_barrier(n_workers=_N_WORKERS)
        @once.once
        def yielding_iterator():
            nonlocal counter
            for _ in range(3):
                yield counter.get_incremented()

        with concurrent.futures.ThreadPoolExecutor(max_workers=_N_WORKERS) as executor:
            results = executor.map(lambda _: list(yielding_iterator()), range(_N_WORKERS * 2))
        for result in results:
            self.assertEqual(result, [1, 2, 3])

    def test_iterator_lock_not_held_during_evaluation(self):
        counter = Counter()

        @once.once
        def yielding_iterator():
            nonlocal counter
            for _ in range(3):
                yield counter.get_incremented()

        gen1 = yielding_iterator()
        gen2 = yielding_iterator()
        self.assertEqual(next(gen1), 1)
        counter.ready.clear()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # By using ensure_started and self.assertFalse(updater.done()), we can ensure it is
            # definitively still running.
            gen1_updater = ensure_started(executor, next, gen1)
            self.assertFalse(gen1_updater.done())
            self.assertEqual(next(gen2), 1)
            gen2_updater = ensure_started(executor, next, gen2)
            self.assertFalse(gen1_updater.done())
            self.assertFalse(gen2_updater.done())
            counter.ready.set()
            self.assertEqual(gen1_updater.result(), 2)
            self.assertEqual(gen2_updater.result(), 2)

    def test_threaded_single_function(self):
        counting_fn, counter = generate_once_counter_fn()
        barrier_counting_fn = execute_with_barrier(counting_fn, n_workers=_N_WORKERS)
        with concurrent.futures.ThreadPoolExecutor(max_workers=_N_WORKERS) as executor:
            results_generator = executor.map(barrier_counting_fn, range(_N_WORKERS))
            results = list(results_generator)
        self.assertEqual(len(results), _N_WORKERS)
        for r in results:
            self.assertEqual(r, 1)
        self.assertEqual(counter.value, 1)

    def test_once_per_thread(self):
        counter = Counter()

        @execute_with_barrier(n_workers=_N_WORKERS)
        @once.once(per_thread=True)
        def counting_fn(*args) -> int:
            """Returns the call count, which should always be 1."""
            nonlocal counter
            del args
            return counter.get_incremented()

        with concurrent.futures.ThreadPoolExecutor(max_workers=_N_WORKERS) as executor:
            results = list(executor.map(counting_fn, range(_N_WORKERS * 4)))
        self.assertEqual(min(results), 1)
        self.assertEqual(max(results), _N_WORKERS)

    def test_threaded_multiple_functions(self):
        counters = []
        fns = []

        for _ in range(4):
            cfn, counter = generate_once_counter_fn()
            counters.append(counter)
            fns.append(execute_with_barrier(cfn, n_workers=_N_WORKERS))

        promises = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=_N_WORKERS) as executor:
            for cfn in fns:
                for _ in range(_N_WORKERS):
                    promises.append(executor.submit(cfn))
            del cfn
            fns.clear()
            for promise in promises:
                self.assertEqual(promise.result(), 1)
        for counter in counters:
            self.assertEqual(counter.value, 1)

    def test_different_fn_do_not_deadlock(self):
        """Ensure different functions use different locks to avoid deadlock."""

        fn1_called = threading.Event()
        fn2_called = threading.Event()

        # If fn1 is called first, these functions will deadlock unless they can
        # run in parallel.
        @once.once
        def fn1():
            fn1_called.set()
            if not fn2_called.wait(5.0):
                self.fail("Function fn1 deadlocked for 5 seconds.")

        @once.once
        def fn2():
            fn2_called.set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            fn1_promise = ensure_started(executor, fn1)
            fn1_called.wait()
            fn2()
            fn1_promise.result()

    def test_closure_gc(self):
        """Tests that closure function is not cached indefinitely"""

        class EphemeralObject:
            """Object which should get GC'ed"""

        def create_closure():
            ephemeral = EphemeralObject()
            ephemeral_ref = weakref.ref(ephemeral)

            @once.once
            def closure():
                return ephemeral

            return closure, ephemeral_ref

        closure, ephemeral_ref = create_closure()

        # Cannot yet be garbage collected because kept alive in the closure.
        self.assertIsNotNone(ephemeral_ref())
        self.assertIsInstance(closure(), EphemeralObject)
        self.assertIsNotNone(ephemeral_ref())
        self.assertIsInstance(closure(), EphemeralObject)
        del closure
        # Can now be garbage collected.
        # In CPython this call technically should not be needed, because
        # garbage collection should have happened automatically. However, that
        # is an implementation detail which does not hold on all platforms,
        # such as for example pypy. Therefore, we manually trigger a garbage
        # collection cycle.
        gc.collect()
        self.assertIsNone(ephemeral_ref())

    def test_function_signature_preserved(self):
        @once.once
        def type_annotated_fn(arg: float) -> int:
            """Very descriptive docstring."""
            del arg
            return 1

        sig = inspect.signature(type_annotated_fn)
        self.assertIs(sig.parameters["arg"].annotation, float)
        self.assertIs(sig.return_annotation, int)
        self.assertEqual(type_annotated_fn.__doc__, "Very descriptive docstring.")

    def test_once_per_class(self):
        class _CallOnceClass(Counter):
            @once.once_per_class
            def once_fn(self):
                return self.get_incremented()

        a = _CallOnceClass()  # pylint: disable=invalid-name
        b = _CallOnceClass()  # pylint: disable=invalid-name

        self.assertEqual(a.once_fn(), 1)
        self.assertEqual(a.once_fn(), 1)
        self.assertEqual(b.once_fn(), 1)
        self.assertEqual(b.once_fn(), 1)

    def test_once_per_class_parallel(self):
        class _CallOnceClass(Counter):
            @once.once_per_class
            def once_fn(self):
                return self.get_incremented()

        once_obj = _CallOnceClass()

        @execute_with_barrier(n_workers=_N_WORKERS)
        def execute(_):
            return once_obj.once_fn()

        with concurrent.futures.ThreadPoolExecutor(max_workers=_N_WORKERS) as executor:
            results = list(executor.map(execute, range(_N_WORKERS * 4)))
        self.assertEqual(min(results), 1)
        self.assertEqual(max(results), 1)

    def test_once_per_class_per_thread(self):
        class _CallOnceClass(Counter):
            @once.once_per_class.with_options(per_thread=True)
            def once_fn(self):
                return self.get_incremented()

        once_obj = _CallOnceClass()

        @execute_with_barrier(n_workers=_N_WORKERS)
        def execute(_):
            return once_obj.once_fn()

        with concurrent.futures.ThreadPoolExecutor(max_workers=_N_WORKERS) as executor:
            results = list(executor.map(execute, range(_N_WORKERS * 4)))
        self.assertEqual(min(results), 1)
        self.assertEqual(max(results), _N_WORKERS)

    def test_once_not_allowed_on_method(self):
        with self.assertRaises(SyntaxError):

            class _InvalidClass:  # pylint: disable=unused-variable
                @once.once
                def once_method(self):
                    pass

    def test_once_per_instance_not_allowed_on_function(self):
        with self.assertRaises(SyntaxError):

            @once.once_per_instance
            def once_fn():
                pass

    def test_once_per_class_not_allowed_on_classmethod(self):
        with self.assertRaises(SyntaxError):

            class _InvalidClass:  # pylint: disable=unused-variable
                @once.once_per_instance
                @classmethod
                def once_method(cls):
                    pass

    def test_once_per_class_not_allowed_on_staticmethod(self):
        with self.assertRaises(SyntaxError):

            class _InvalidClass:  # pylint: disable=unused-variable
                @once.once_per_instance
                @staticmethod
                def once_method():
                    pass

    def test_once_per_instance(self):
        class _CallOnceClass:
            def __init__(self, value: str, test: unittest.TestCase):
                self._value = value
                self.called = False
                self.test = test

            @once.once_per_instance
            def value(self):  # pylint: disable=inconsistent-return-statements
                if not self.called:
                    self.called = True
                    return self._value
                if self.called:
                    self.test.fail(f"Method on {self.value} called a second time.")

        a = _CallOnceClass("a", self)  # pylint: disable=invalid-name
        b = _CallOnceClass("b", self)  # pylint: disable=invalid-name

        with concurrent.futures.ThreadPoolExecutor(max_workers=_N_WORKERS) as executor:
            a_jobs = [executor.submit(a.value) for _ in range(_N_WORKERS // 2)]
            b_jobs = [executor.submit(b.value) for _ in range(_N_WORKERS // 2)]
            for a_job in a_jobs:
                self.assertEqual(a_job.result(), "a")
            for b_job in b_jobs:
                self.assertEqual(b_job.result(), "b")

        self.assertEqual(a.value(), "a")
        self.assertEqual(a.value(), "a")
        self.assertEqual(b.value(), "b")
        self.assertEqual(b.value(), "b")

    def test_once_per_instance_do_not_block_each_other(self):
        class _BlockableClass:
            def __init__(self, test: unittest.TestCase):
                self.test = test
                self.started = threading.Event()
                self.counter = Counter()

            @once.once_per_instance
            def run(self) -> int:
                self.started.set()
                return self.counter.get_incremented()

        a = _BlockableClass(self)
        a.counter.ready.clear()
        b = _BlockableClass(self)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            a_job = executor.submit(a.run)
            a.started.wait()
            # At this point, the A job has started. However, it cannot
            # complete while paused. Despite this, we want to ensure
            # that B can still run.
            b_job = executor.submit(b.run)
            # The b_job will deadlock and this will fail if different
            # object executions block each other.
            self.assertEqual(b_job.result(timeout=5), 1)
            a.counter.ready.set()
            self.assertEqual(a_job.result(timeout=5), 1)

    def test_once_per_instance_parallel(self):
        class _CallOnceClass(Counter):
            @once.once_per_instance
            def once_fn(self):
                return self.get_incremented()

        once_objs = [_CallOnceClass(), _CallOnceClass(), _CallOnceClass(), _CallOnceClass()]

        @execute_with_barrier(n_workers=_N_WORKERS)
        def execute(i):
            return once_objs[i % 4].once_fn()

        with concurrent.futures.ThreadPoolExecutor(max_workers=_N_WORKERS) as executor:
            results = list(executor.map(execute, range(_N_WORKERS * 4)))
        self.assertEqual(min(results), 1)
        self.assertEqual(max(results), 1)

    def test_once_per_instance_per_thread(self):
        class _CallOnceClass(Counter):
            @once.once_per_instance.with_options(per_thread=True)
            def once_fn(self):
                return self.get_incremented()

        once_objs = [_CallOnceClass(), _CallOnceClass(), _CallOnceClass(), _CallOnceClass()]

        @execute_with_barrier(n_workers=_N_WORKERS)
        def execute(i):
            return once_objs[i % 4].once_fn()

        with concurrent.futures.ThreadPoolExecutor(max_workers=_N_WORKERS) as executor:
            results = list(executor.map(execute, range(_N_WORKERS)))
        self.assertEqual(min(results), 1)
        self.assertEqual(max(results), math.ceil(_N_WORKERS / 4))

    def test_once_per_class_classmethod(self):
        counter = Counter()

        class _CallOnceClass:
            @once.once_per_class
            @classmethod
            def value(cls):
                nonlocal counter
                return counter.get_incremented()

        self.assertEqual(_CallOnceClass.value(), 1)
        self.assertEqual(_CallOnceClass.value(), 1)

    def test_once_per_class_staticmethod(self):
        counter = Counter()

        class _CallOnceClass:
            @once.once_per_class
            @staticmethod
            def value():
                nonlocal counter
                return counter.get_incremented()

        self.assertEqual(_CallOnceClass.value(), 1)
        self.assertEqual(_CallOnceClass.value(), 1)

    def test_receiving_iterator(self):
        @once.once
        def receiving_iterator():
            next = yield 0
            while next is not None:
                next = yield next

        gen_1 = receiving_iterator()
        gen_2 = receiving_iterator()
        self.assertEqual(gen_1.send(None), 0)
        self.assertEqual(gen_1.send(1), 1)
        self.assertEqual(gen_1.send(2), 2)
        self.assertEqual(gen_2.send(None), 0)
        self.assertEqual(gen_2.send(-1), 1)
        self.assertEqual(gen_2.send(-1), 2)
        self.assertEqual(gen_2.send(5), 5)
        self.assertEqual(next(gen_2, None), None)
        self.assertEqual(gen_1.send(None), 5)
        self.assertEqual(next(gen_1, None), None)
        self.assertEqual(list(receiving_iterator()), [0, 1, 2, 5])

    def test_receiving_iterator_parallel_execution(self):
        @once.once
        def receiving_iterator():
            next = yield 0
            while next is not None:
                next = yield next

        barrier = threading.Barrier(_N_WORKERS)

        def call_iterator(_):
            gen = receiving_iterator()
            result = []
            barrier.wait()
            result.append(gen.send(None))
            for i in range(1, _N_WORKERS * 4):
                result.append(gen.send(i))
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=_N_WORKERS) as executor:
            results = executor.map(call_iterator, range(_N_WORKERS))
        for result in results:
            self.assertEqual(result, list(range(_N_WORKERS * 4)))

    def test_receiving_iterator_parallel_execution_halting(self):
        @once.once
        def receiving_iterator():
            next = yield 0
            while next is not None:
                next = yield next

        barrier = threading.Barrier(_N_WORKERS)

        def call_iterator(n):
            """Call the iterator but end early"""
            gen = receiving_iterator()
            result = []
            barrier.wait()
            result.append(gen.send(None))
            for i in range(1, n):
                result.append(gen.send(i))
            return result

        # Unlike the previous test, each execution should yield lists of different lengths.
        # This ensures that the iterator does not hang, even if not exhausted
        with concurrent.futures.ThreadPoolExecutor(max_workers=_N_WORKERS) as executor:
            results = executor.map(call_iterator, range(1, _N_WORKERS + 1))
        for i, result in enumerate(results):
            self.assertEqual(result, list(range(i + 1)))


class TestOnceAsync(unittest.IsolatedAsyncioTestCase):
    async def test_fn_called_once(self):
        counter1 = Counter()

        @once.once
        async def counting_fn1():
            return counter1.get_incremented()

        counter2 = Counter()
        # We should get a different value than the previous function
        counter2.get_incremented()

        @once.once
        async def counting_fn2():
            return counter2.get_incremented()

        self.assertEqual(await counting_fn1(), 1)
        self.assertEqual(await counting_fn1(), 1)
        self.assertEqual(await counting_fn2(), 2)
        self.assertEqual(await counting_fn2(), 2)

    async def test_once_per_thread(self):
        counter = Counter()

        @once.once(per_thread=True)
        async def counting_fn(*args) -> int:
            """Returns the call count, which should always be 1."""
            nonlocal counter
            del args
            return counter.get_incremented()

        @execute_with_barrier(n_workers=_N_WORKERS)
        def execute():
            result = counting_fn()
            return asyncio.run(result)

        threads = [threading.Thread(target=execute) for i in range(_N_WORKERS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(await counting_fn(), _N_WORKERS + 1)
        self.assertEqual(await counting_fn(), _N_WORKERS + 1)

    async def test_failing_function(self):
        counter = Counter()

        @once.once
        async def sample_failing_fn():
            if counter.get_incremented() < 4:
                raise ValueError("expected failure")
            return 1

        with self.assertRaises(ValueError):
            await sample_failing_fn()
        self.assertEqual(counter.get_incremented(), 2)
        with self.assertRaises(ValueError):
            await sample_failing_fn()
        # This ensures that this was a new function call, not a cached result.
        self.assertEqual(counter.get_incremented(), 4)
        self.assertEqual(await sample_failing_fn(), 1)

    async def test_inspect_func(self):
        @once.once
        async def async_func():
            return True

        self.assertFalse(inspect.isasyncgenfunction(async_func))
        self.assertTrue(inspect.iscoroutinefunction(async_func))

        coroutine = async_func()
        self.assertTrue(inspect.iscoroutine(coroutine))
        self.assertTrue(inspect.isawaitable(coroutine))
        self.assertFalse(inspect.isasyncgen(coroutine))

        # Just for cleanup.
        await coroutine

    async def test_inspect_iterator(self):
        @once.once
        async def async_yielding_iterator():
            for i in range(3):
                yield i

        self.assertTrue(inspect.isasyncgenfunction(async_yielding_iterator))
        self.assertFalse(inspect.iscoroutinefunction(async_yielding_iterator))

        coroutine = async_yielding_iterator()
        self.assertFalse(inspect.iscoroutine(coroutine))
        self.assertFalse(inspect.isawaitable(coroutine))
        self.assertTrue(inspect.isasyncgen(coroutine))

        # Just for cleanup.
        async for i in coroutine:
            pass

    async def test_iterator(self):
        counter = Counter()

        @once.once
        async def async_yielding_iterator():
            for i in range(3):
                yield counter.get_incremented()

        self.assertEqual([i async for i in async_yielding_iterator()], [1, 2, 3])
        self.assertEqual([i async for i in async_yielding_iterator()], [1, 2, 3])

    @unittest.skip("This currently hangs and needs to be fixed, GitHub Issue #12")
    async def test_failing_generator(self):
        counter = Counter()

        @once.once
        async def sample_failing_fn():
            yield counter.get_incremented()
            raise ValueError("expected failure")

        with self.assertRaises(ValueError):
            [i async for i in sample_failing_fn()]
        with self.assertRaises(ValueError):
            [i async for i in sample_failing_fn()]
        self.assertEqual(await anext(sample_failing_fn()), 1)
        self.assertEqual(await anext(sample_failing_fn()), 1)

    async def test_iterator_is_lazily_evaluted(self):
        counter = Counter()

        @once.once
        async def async_yielding_iterator():
            for i in range(3):
                yield counter.get_incremented()

        gen_1 = async_yielding_iterator()
        gen_2 = async_yielding_iterator()
        gen_3 = async_yielding_iterator()

        self.assertEqual(counter.value, 0)
        self.assertEqual(await anext(gen_1), 1)
        self.assertEqual(await anext(gen_2), 1)
        self.assertEqual(await anext(gen_2), 2)
        self.assertEqual(await anext(gen_2), 3)
        self.assertEqual(await anext(gen_1), 2)
        self.assertEqual(await anext(gen_3), 1)
        self.assertEqual(await anext(gen_3), 2)
        self.assertEqual(await anext(gen_3), 3)
        self.assertEqual(await anext(gen_3, None), None)
        self.assertEqual(await anext(gen_2, None), None)
        self.assertEqual(await anext(gen_1), 3)
        self.assertEqual(await anext(gen_2, None), None)

    async def test_receiving_iterator(self):
        @once.once
        async def async_receiving_iterator():
            next = yield 0
            while next is not None:
                next = yield next

        gen_1 = async_receiving_iterator()
        gen_2 = async_receiving_iterator()
        self.assertEqual(await gen_1.asend(None), 0)
        self.assertEqual(await gen_1.asend(1), 1)
        self.assertEqual(await gen_1.asend(2), 2)
        self.assertEqual(await gen_2.asend(None), 0)
        self.assertEqual(await gen_2.asend(None), 1)
        self.assertEqual(await gen_2.asend(None), 2)
        self.assertEqual(await gen_2.asend(5), 5)
        self.assertEqual(await anext(gen_2, None), None)
        self.assertEqual(await gen_1.asend(None), 5)
        self.assertEqual(await anext(gen_1, None), None)
        self.assertEqual([i async for i in async_receiving_iterator()], [0, 1, 2, 5])

    async def test_receiving_iterator_parallel_execution(self):
        @once.once
        async def receiving_iterator():
            next = yield 0
            while next is not None:
                next = yield next

        async def call_iterator(_):
            gen = receiving_iterator()
            result = []
            result.append(await gen.asend(None))
            for i in range(1, _N_WORKERS):
                result.append(await gen.asend(i))
            return result

        results = map(call_iterator, range(_N_WORKERS))
        for result in results:
            self.assertEqual(await result, list(range(_N_WORKERS)))

    async def test_receiving_iterator_parallel_execution_halting(self):
        @once.once
        async def receiving_iterator():
            next = yield 0
            while next is not None:
                next = yield next

        async def call_iterator(n):
            """Call the iterator but end early"""
            gen = receiving_iterator()
            result = []
            result.append(await gen.asend(None))
            for i in range(1, n):
                result.append(await gen.asend(i))
            return result

        results = map(call_iterator, range(1, _N_WORKERS))
        for i, result in enumerate(results):
            self.assertEqual(await result, list(range(i + 1)))

    @unittest.skipIf(not hasattr(asyncio, "Barrier"), "Requires Barrier to evaluate")
    async def test_iterator_lock_not_held_during_evaluation(self):
        counter = Counter()

        @once.once
        async def async_yielding_iterator():
            barrier = yield counter.get_incremented()
            while barrier is not None:
                await barrier.wait()
                barrier = yield counter.get_incremented()

        gen_1 = async_yielding_iterator()
        gen_2 = async_yielding_iterator()
        barrier = asyncio.Barrier(2)
        self.assertEqual(await gen_1.asend(None), 1)
        task1 = asyncio.create_task(gen_1.asend(barrier))

        # Loop until task1 is stuck waiting.
        while barrier.n_waiting < 1:
            await asyncio.sleep(0)

        self.assertEqual(
            await gen_2.asend(None), 1
        )  # Should return immediately even though task1 is stuck.

        # .asend("None") should be ignored because task1 has already started,
        # so task2 should still return 2 instead of ending iteration.
        task2 = asyncio.create_task(gen_2.asend(None))

        await barrier.wait()

        self.assertEqual(await task1, 2)
        self.assertEqual(await task2, 2)
        self.assertEqual(await anext(gen_1, None), None)
        self.assertEqual(await anext(gen_2, None), None)

    async def test_once_per_class(self):
        class _CallOnceClass(Counter):
            @once.once_per_class
            async def once_fn(self):
                return self.get_incremented()

        a = _CallOnceClass()  # pylint: disable=invalid-name
        b = _CallOnceClass()  # pylint: disable=invalid-name

        self.assertEqual(await a.once_fn(), 1)
        self.assertEqual(await a.once_fn(), 1)
        self.assertEqual(await b.once_fn(), 1)
        self.assertEqual(await b.once_fn(), 1)

    async def test_once_per_class_classmethod(self):
        counter = Counter()

        class _CallOnceClass:
            @once.once_per_class
            @classmethod
            async def value(cls):
                nonlocal counter
                return counter.get_incremented()

        self.assertTrue(inspect.iscoroutinefunction(_CallOnceClass.value))
        self.assertEqual(await _CallOnceClass.value(), 1)
        self.assertEqual(await _CallOnceClass.value(), 1)

    async def test_once_per_class_staticmethod(self):
        counter = Counter()

        class _CallOnceClass:
            @once.once_per_class
            @staticmethod
            async def value():
                nonlocal counter
                return counter.get_incremented()

        self.assertEqual(await _CallOnceClass.value(), 1)
        self.assertEqual(await _CallOnceClass.value(), 1)


if __name__ == "__main__":
    unittest.main()
