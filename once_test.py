"""Unit tests for once decorators."""

# pylint: disable=missing-function-docstring
import asyncio
import collections.abc
import concurrent.futures
import contextlib
import functools
import gc
import inspect
import math
import sys
import threading
import traceback
import unittest
import uuid
import weakref

import typing_extensions

import once


# This is a "large" number of workers to schedule function calls in parallel.
_N_WORKERS = 32
# The default is for thread switching to occur every 5ms. Because most of the
# threads in these tests are worst-case stress tests that have very little to
# do and are often blocked, we set the threads to switch a bit quicker just so
# tests can go a bit quicker.
if sys.getswitchinterval() > 0.0001:
    sys.setswitchinterval(0.0001)


class WrappedException:
    def __init__(self, exception):
        self.exception = exception


def parallel_map(
    test: unittest.TestCase,
    func: collections.abc.Callable,
    # would be collections.abc.Iterable[tuple] | None on py >= 3.10
    call_args=None,
    n_threads: int = _N_WORKERS,
    timeout: float = 60.0,
) -> list:
    """Run a function multiple times in parallel.

    We ensure that N parallel tasks are all launched at the "same time", which
    means all have parallel threads which are released to the GIL to execute at
    the same time.
    Why?
    We can't rely on the thread pool excector to always spin up the full list of _N_WORKERS.
    In pypy, we have observed that even with blocked tasks, the same thread executes multiple
    function calls. This lets us handle the scheduling in a predictable way for testing.
    """
    if call_args is None:
        call_args = (tuple() for _ in range(n_threads))

    batches = [[] for i in range(n_threads)]  # type: list[list[tuple[int, tuple]]]
    for i, call_args in enumerate(call_args):
        if not isinstance(call_args, tuple):
            raise TypeError("call arguments must be a tuple")
        batches[i % n_threads].append((i, call_args))
    n_calls = i + 1  # len(call_args), but it is now an exhuasted iterator.
    unset = object()
    results_lock = threading.Lock()
    results = [unset for _ in range(n_calls)]

    # This barrier is used to ensure that all calls release together, after this function has
    # completed its setup of creating them.
    start_barrier = threading.Barrier(min(n_threads, n_calls))

    def wrapped_fn(batch):
        start_barrier.wait()
        for index, args in batch:
            try:
                result = func(*args)
            except Exception as e:
                result = WrappedException(e)
            with results_lock:
                results[index] = result

    # We manually set thread names for easier debugging.
    invocation_id = str(uuid.uuid4())
    threads = [
        threading.Thread(target=wrapped_fn, args=[batch], name=f"{test.id()}-{i}-{invocation_id}")
        for i, batch in enumerate(batches)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)
    for i, result in enumerate(results):
        if result is unset:
            test.fail(f"Call {i} did not complete succesfully")
        elif isinstance(result, WrappedException):
            raise result.exception
    return results


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
        self._lock = threading.Lock()

    def get_incremented(self) -> int:
        self.ready.wait()
        with self._lock:
            self.value += 1
            return self.value


def execute_with_barrier(*args, n_workers=None, is_async=False):
    """Decorator to ensure function calls do not begin until at least n_workers have started.

    This ensures that our parallel tests actually test concurrency. Without this, it is possible
    that function calls execute as they are being scheduled, and do not truly execute in parallel.

    The decorated function should receive an integer multiple of n_workers invokations.

    Please note that calling this decorator outside of our once call will generally not change the
    semantic meaning. However, it does increase the likelihood that once executions occur in
    parallel, to increase the chance of races and therefore the chances that our tests catch a race
    condition, although this is still non-deterministic. Calling this decorator **inside** the once
    decorator however is deterministic.
    """
    # Trick to make the decorator accept an arugment. The first call only gets the n_workers
    # parameter, and then returns a new function with it set that then accepts the function.
    if n_workers is None:
        raise ValueError("n_workers not set")
    if len(args) == 0:
        return functools.partial(execute_with_barrier, n_workers=n_workers, is_async=is_async)
    if len(args) > 1:
        raise ValueError("Up to one argument expected.")
    func = args[0]
    barrier = threading.Barrier(n_workers)

    if is_async:

        async def wrapped(*args, **kwargs):
            barrier.wait()  # yes I know
            return await func(*args, **kwargs)

    else:

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


class LineCapture:
    def __init__(self):
        self.line = None

    def record_next_line(self):
        """Record the next line in the parent frame"""
        self.line = inspect.currentframe().f_back.f_lineno + 1


class ExceptionContextManager:
    exception: Exception


@contextlib.contextmanager
def assertRaisesWithLineInStackTrace(test: unittest.TestCase, exception_type, line: LineCapture):
    try:
        container = ExceptionContextManager()
        yield container
    except exception_type as exception:
        container.exception = exception
        traceback_exception = traceback.TracebackException.from_exception(exception)
        if not len(traceback_exception.stack):
            test.fail("Exception stack not preserved. Did you use the raw assertRaises by mistake?")
        locations = [(frame.filename, frame.lineno) for frame in traceback_exception.stack]
        line_number = line.line
        error_message = [
            f"Traceback for exception {repr(exception)} did not have frame on line {line_number}. Exception below\n"
        ]
        error_message.extend(traceback_exception.format())
        test.assertIn((__file__, line_number), locations, msg="".join(error_message))

    else:
        test.fail("expected exception not called")


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

    def test_inspect_iterator(self) -> None:
        @once.once
        def yielding_iterator():
            for i in range(3):
                yield i

        self.assertTrue(inspect.isgeneratorfunction(yielding_iterator))

    def test_counter_works(self) -> None:
        """Ensure the counter text fixture works."""
        counter = Counter()
        self.assertEqual(counter.value, 0)
        self.assertEqual(counter.get_incremented(), 1)
        self.assertEqual(counter.value, 1)
        self.assertEqual(counter.get_incremented(), 2)
        self.assertEqual(counter.value, 2)

    def test_different_args_same_result(self) -> None:
        counting_fn, counter = generate_once_counter_fn()
        self.assertEqual(counting_fn(1), 1)
        self.assertEqual(counter.value, 1)
        # Should return the same result as the first call.
        self.assertEqual(counting_fn(2), 1)
        self.assertEqual(counter.value, 1)

    def test_partial(self) -> None:
        counter = Counter()
        func = once.once(functools.partial(lambda _: counter.get_incremented(), None))
        self.assertEqual(func(), 1)
        self.assertEqual(func(), 1)

    def test_reset(self):
        counter = Counter()

        @once.once(allow_reset=True)
        def counting_fn():
            return counter.get_incremented()

        self.assertEqual(counting_fn(), 1)
        counting_fn.reset()
        self.assertEqual(counting_fn(), 2)
        counting_fn.reset()
        counting_fn.reset()
        self.assertEqual(counting_fn(), 3)

    def test_reset_not_allowed(self):
        counting_fn, counter = generate_once_counter_fn()
        self.assertEqual(counting_fn(None), 1)
        with self.assertRaises(RuntimeError):
            counting_fn.reset()

    def test_failing_function(self) -> None:
        counter = Counter()
        failing_line = LineCapture()

        @once.once
        def sample_failing_fn():
            nonlocal failing_line
            if counter.get_incremented() < 4:
                failing_line.record_next_line()
                raise ValueError("expected failure")
            return 1

        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            sample_failing_fn()
        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line) as cm:
            sample_failing_fn()
        self.assertEqual(cm.exception.args[0], "expected failure")
        self.assertEqual(counter.get_incremented(), 2)
        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            sample_failing_fn()
        self.assertEqual(counter.get_incremented(), 3, "Function call incremented the counter")

    def test_failing_function_retry_exceptions(self) -> None:
        counter = Counter()
        failing_line = LineCapture()

        @once.once(retry_exceptions=True)
        def sample_failing_fn():
            nonlocal failing_line
            if counter.get_incremented() < 4:
                failing_line.record_next_line()
                raise ValueError("expected failure")
            return 1

        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            sample_failing_fn()
        self.assertEqual(counter.get_incremented(), 2)
        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            sample_failing_fn()
        # This ensures that this was a new function call, not a cached result.
        self.assertEqual(counter.get_incremented(), 4)
        self.assertEqual(sample_failing_fn(), 1)

    def test_iterator(self) -> None:
        counter = Counter()

        @once.once
        def yielding_iterator():
            nonlocal counter
            for _ in range(3):
                yield counter.get_incremented()

        self.assertEqual(list(yielding_iterator()), [1, 2, 3])
        self.assertEqual(list(yielding_iterator()), [1, 2, 3])

    def test_iterator_reset(self):
        counter = Counter()

        @once.once(allow_reset=True)
        def yielding_iterator():
            nonlocal counter
            for _ in range(3):
                yield counter.get_incremented()

        self.assertEqual(list(yielding_iterator()), [1, 2, 3])
        yielding_iterator.reset()
        self.assertEqual(list(yielding_iterator()), [4, 5, 6])

    def test_iterator_reset_not_allowed(self):
        counter = Counter()

        @once.once
        def yielding_iterator():
            nonlocal counter
            for _ in range(3):
                yield counter.get_incremented()

        self.assertEqual(list(yielding_iterator()), [1, 2, 3])
        with self.assertRaises(RuntimeError):
            yielding_iterator.reset()

    def test_failing_generator(self) -> None:
        counter = Counter()
        failing_line = LineCapture()

        @once.once
        def sample_failing_fn():
            yield counter.get_incremented()
            result = counter.get_incremented()
            yield result
            if result == 2:
                failing_line.record_next_line()
                raise ValueError("expected failure after 2.")

        # Both of these calls should return the same results.
        call1 = sample_failing_fn()
        call2 = sample_failing_fn()
        self.assertEqual(next(call1), 1)
        self.assertEqual(next(call2), 1)
        self.assertEqual(next(call1), 2)
        self.assertEqual(next(call2), 2)
        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            next(call1)
        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            next(call2)
        # These next 2 calls should also fail.
        call3 = sample_failing_fn()
        call4 = sample_failing_fn()
        self.assertEqual(next(call3), 1)
        self.assertEqual(next(call4), 1)
        self.assertEqual(next(call3), 2)
        self.assertEqual(next(call4), 2)
        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            next(call3)
        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            next(call4)

    def test_failing_generator_retry_exceptions(self) -> None:
        counter = Counter()
        failing_line = LineCapture()

        @once.once(retry_exceptions=True)
        def sample_failing_fn():
            yield counter.get_incremented()
            result = counter.get_incremented()
            yield result
            if result == 2:
                failing_line.record_next_line()
                raise ValueError("expected failure after 2.")

        # Both of these calls should return the same results.
        call1 = sample_failing_fn()
        call2 = sample_failing_fn()
        self.assertEqual(next(call1), 1)
        self.assertEqual(next(call2), 1)
        self.assertEqual(next(call1), 2)
        self.assertEqual(next(call2), 2)
        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            next(call1)
        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            next(call2)
        # These next 2 calls should succeed.
        call3 = sample_failing_fn()
        call4 = sample_failing_fn()
        self.assertEqual(list(call3), [3, 4])
        self.assertEqual(list(call4), [3, 4])
        # Subsequent calls should return the original value.
        self.assertEqual(list(sample_failing_fn()), [3, 4])
        self.assertEqual(list(sample_failing_fn()), [3, 4])

    def test_iterator_parallel_execution(self) -> None:
        counter = Counter()

        @once.once
        def yielding_iterator():
            nonlocal counter
            for _ in range(3):
                yield counter.get_incremented()

        results = parallel_map(
            self,
            lambda: list(yielding_iterator()),
            (tuple() for _ in range(_N_WORKERS * 2)),
        )
        for result in results:
            self.assertEqual(result, [1, 2, 3])

    def test_iterator_lock_not_held_during_evaluation(self) -> None:
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

    def test_threaded_single_function(self) -> None:
        counting_fn, counter = generate_once_counter_fn()
        results = parallel_map(self, counting_fn)
        self.assertEqual(len(results), _N_WORKERS)
        for r in results:
            self.assertEqual(r, 1)
        self.assertEqual(counter.value, 1)

    def test_once_per_thread(self) -> None:
        counter = Counter()

        @once.once(per_thread=True)
        @execute_with_barrier(n_workers=_N_WORKERS)
        def counting_fn(*args) -> int:
            """Returns the call count, which should always be 1."""
            nonlocal counter
            del args
            return counter.get_incremented()

        results = parallel_map(self, counting_fn, (tuple() for _ in range(_N_WORKERS * 4)))
        self.assertEqual(min(results), 1)
        self.assertEqual(max(results), _N_WORKERS)

    def test_threaded_multiple_functions(self) -> None:
        counters = []
        fns = []

        for _ in range(4):
            cfn, counter = generate_once_counter_fn()
            counters.append(counter)
            fns.append(cfn)

        def call_all_functions(i):
            for j in range(i, i + 4):
                self.assertEqual(fns[j % 4](), 1)

        parallel_map(self, call_all_functions, ((i,) for i in range(_N_WORKERS)))
        for counter in counters:
            self.assertEqual(counter.value, 1)

    def test_different_fn_do_not_deadlock(self) -> None:
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

    def test_closure_gc(self) -> None:
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

    def test_function_signature_preserved(self) -> None:
        def type_annotated_fn(arg: float) -> int:
            """Very descriptive docstring."""
            del arg
            return 1

        decorated_function = once.once(type_annotated_fn)
        typing_extensions.assert_type(decorated_function(1.0), int)
        original_sig = inspect.signature(type_annotated_fn)
        decorated_sig = inspect.signature(decorated_function)
        self.assertIs(original_sig.parameters["arg"].annotation, float)
        self.assertIs(decorated_sig.parameters["arg"].annotation, float)
        self.assertIs(original_sig.return_annotation, int)
        self.assertIs(decorated_sig.return_annotation, int)
        self.assertEqual(inspect.getdoc(type_annotated_fn), inspect.getdoc(decorated_function))
        if sys.flags.optimize >= 2:
            self.skipTest("docstrings get stripped with -OO")
        self.assertEqual(inspect.getdoc(type_annotated_fn), "Very descriptive docstring.")

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

    def test_once_per_class_parallel(self) -> None:
        class _CallOnceClass(Counter):
            @once.once_per_class
            def once_fn(self):
                return self.get_incremented()

        once_obj = _CallOnceClass()

        def execute():
            return once_obj.once_fn()

        results = parallel_map(self, execute, (tuple() for _ in range(_N_WORKERS * 4)))
        self.assertEqual(min(results), 1)
        self.assertEqual(max(results), 1)

    def test_once_per_class_per_thread(self) -> None:
        class _CallOnceClass(Counter):
            @once.once_per_class.with_options(per_thread=True)
            @execute_with_barrier(n_workers=_N_WORKERS)
            def once_fn(self):
                return self.get_incremented()

        once_obj = _CallOnceClass()

        def execute():
            return once_obj.once_fn()

        results = parallel_map(self, execute, (tuple() for _ in range(_N_WORKERS * 4)))
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

        def call_and_check_both(i: int):
            # Run in different order based on the call
            if i % 4 == 0:
                self.assertEqual(a.value(), "a")
                self.assertEqual(a.value(), "a")
                self.assertEqual(b.value(), "b")
                self.assertEqual(b.value(), "b")
            elif i % 4 == 1:
                self.assertEqual(a.value(), "a")
                self.assertEqual(b.value(), "b")
                self.assertEqual(a.value(), "a")
                self.assertEqual(b.value(), "b")
            elif i % 4 == 2:
                self.assertEqual(b.value(), "b")
                self.assertEqual(a.value(), "a")
                self.assertEqual(b.value(), "b")
                self.assertEqual(a.value(), "a")
            else:
                self.assertEqual(b.value(), "b")
                self.assertEqual(b.value(), "b")
                self.assertEqual(a.value(), "a")
                self.assertEqual(a.value(), "a")

        parallel_map(self, call_and_check_both, ((i,) for i in range(_N_WORKERS)))

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
            self.assertEqual(b_job.result(timeout=15), 1)
            a.counter.ready.set()
            self.assertEqual(a_job.result(timeout=15), 1)

    def test_once_per_instance_parallel(self) -> None:
        class _CallOnceClass(Counter):
            @once.once_per_instance
            @execute_with_barrier(n_workers=4)
            def once_fn(self):
                return self.get_incremented()

        once_objs = [_CallOnceClass(), _CallOnceClass(), _CallOnceClass(), _CallOnceClass()]

        def execute(i):
            return once_objs[i % 4].once_fn()

        results = parallel_map(self, execute, ((i,) for i in range(_N_WORKERS * 4)))
        self.assertEqual(min(results), 1)
        self.assertEqual(max(results), 1)

    def test_once_per_instance_per_thread(self) -> None:
        class _CallOnceClass(Counter):
            @once.once_per_instance.with_options(per_thread=True)
            @execute_with_barrier(n_workers=_N_WORKERS)
            def once_fn(self):
                return self.get_incremented()

        once_objs = [_CallOnceClass(), _CallOnceClass(), _CallOnceClass(), _CallOnceClass()]

        def execute(i):
            return once_objs[i % 4].once_fn()

        results = parallel_map(self, execute, ((i,) for i in range(_N_WORKERS)))
        self.assertEqual(min(results), 1)
        self.assertEqual(max(results), math.ceil(_N_WORKERS / 4))

    def test_once_per_instance_property(self):
        counter = Counter()

        class _CallOnceClass:
            @once.once_per_instance
            @property
            def value(self):
                nonlocal counter
                return counter.get_incremented()

        a = _CallOnceClass()
        b = _CallOnceClass()
        self.assertEqual(a.value, 1)
        self.assertEqual(b.value, 2)
        self.assertEqual(a.value, 1)
        self.assertEqual(b.value, 2)
        self.assertEqual(_CallOnceClass().value, 3)
        self.assertEqual(_CallOnceClass().value, 4)

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

    def test_once_per_class_staticmethod(self) -> None:
        counter = Counter()

        class _CallOnceClass:
            @once.once_per_class
            @staticmethod
            def value():
                nonlocal counter
                return counter.get_incremented()

        self.assertEqual(_CallOnceClass.value(), 1)
        self.assertEqual(_CallOnceClass.value(), 1)

    def test_receiving_iterator(self) -> None:
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

    def test_receiving_iterator_parallel_execution(self) -> None:
        @once.once
        def receiving_iterator():
            next = yield 0
            while next is not None:
                next = yield next

        barrier = threading.Barrier(_N_WORKERS)

        def call_iterator():
            gen = receiving_iterator()
            result = []
            barrier.wait()
            result.append(gen.send(None))
            for i in range(1, _N_WORKERS * 4):
                result.append(gen.send(i))
            return result

        results = parallel_map(self, call_iterator)
        for result in results:
            self.assertEqual(result, list(range(_N_WORKERS * 4)))

    def test_receiving_iterator_parallel_execution_halting(self) -> None:
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
        results = parallel_map(self, call_iterator, ((i,) for i in range(1, _N_WORKERS + 1)))
        for i, result in enumerate(results):
            self.assertEqual(result, list(range(i + 1)))


class TestOnceAsync(unittest.IsolatedAsyncioTestCase):
    async def test_fn_called_once(self) -> None:
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

    async def test_reset(self):
        counter = Counter()

        @once.once(allow_reset=True)
        async def counting_fn():
            return counter.get_incremented()

        self.assertEqual(await counting_fn(), 1)
        await counting_fn.reset()
        self.assertEqual(await counting_fn(), 2)

    async def test_reset_not_allowed(self):
        counter = Counter()

        @once.once
        async def counting_fn():
            return counter.get_incremented()

        self.assertEqual(await counting_fn(), 1)
        with self.assertRaises(RuntimeError):
            await counting_fn.reset()

    async def test_once_per_thread(self) -> None:
        counter = Counter()

        @once.once(per_thread=True)
        @execute_with_barrier(n_workers=_N_WORKERS, is_async=True)
        async def counting_fn(*args) -> int:
            """Returns the call count, which should always be 1."""
            nonlocal counter

            del args
            return counter.get_incremented()

        async def counting_fn_multiple_caller(*args):
            """Calls counting_fn() multiple times ensuring identical result."""
            result = await counting_fn()
            for i in range(5):
                self.assertEqual(await counting_fn(), result)
            return result

        def execute(*args):
            coro = counting_fn_multiple_caller(*args)
            return asyncio.run(coro)

        parallel_map(self, execute)

    async def test_failing_function(self) -> None:
        counter = Counter()
        failing_line = LineCapture()

        @once.once
        async def sample_failing_fn():
            if counter.get_incremented() < 4:
                failing_line.record_next_line()
                raise ValueError("expected failure")
            return 1

        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            await sample_failing_fn()
        self.assertEqual(counter.get_incremented(), 2)
        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            await sample_failing_fn()
        self.assertEqual(counter.get_incremented(), 3, "Function call incremented the counter")

    async def test_failing_function_retry_exceptions(self) -> None:
        counter = Counter()
        failing_line = LineCapture()

        @once.once(retry_exceptions=True)
        async def sample_failing_fn():
            if counter.get_incremented() < 4:
                failing_line.record_next_line()
                raise ValueError("expected failure")
            return 1

        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            await sample_failing_fn()
        self.assertEqual(counter.get_incremented(), 2)
        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            await sample_failing_fn()
        # This ensures that this was a new function call, not a cached result.
        self.assertEqual(counter.get_incremented(), 4)
        self.assertEqual(await sample_failing_fn(), 1)

    async def test_inspect_func(self) -> None:
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

    async def test_inspect_iterator(self) -> None:
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

    async def test_iterator(self) -> None:
        counter = Counter()

        @once.once
        async def async_yielding_iterator():
            for i in range(3):
                yield counter.get_incremented()

        self.assertEqual([i async for i in async_yielding_iterator()], [1, 2, 3])
        self.assertEqual([i async for i in async_yielding_iterator()], [1, 2, 3])

    async def test_iterator_reset(self):
        counter = Counter()

        @once.once(allow_reset=True)
        async def async_yielding_iterator():
            for i in range(3):
                yield counter.get_incremented()

        self.assertEqual([i async for i in async_yielding_iterator()], [1, 2, 3])
        await async_yielding_iterator.reset()
        self.assertEqual([i async for i in async_yielding_iterator()], [4, 5, 6])

    async def test_iterator_reset_not_allowed(self):
        counter = Counter()

        @once.once
        async def async_yielding_iterator():
            for i in range(3):
                yield counter.get_incremented()

        self.assertEqual([i async for i in async_yielding_iterator()], [1, 2, 3])
        with self.assertRaises(RuntimeError):
            async_yielding_iterator.reset()

    async def test_failing_generator(self) -> None:
        counter = Counter()
        failing_line = LineCapture()

        @once.once
        async def sample_failing_fn():
            yield counter.get_incremented()
            result = counter.get_incremented()
            yield result
            if result == 2:
                failing_line.record_next_line()
                raise ValueError("we raise an error when result is exactly 2")

        # Both of these calls should return the same results.
        call1 = sample_failing_fn()
        call2 = sample_failing_fn()
        self.assertEqual(await anext(call1), 1)
        self.assertEqual(await anext(call2), 1)
        self.assertEqual(await anext(call1), 2)
        self.assertEqual(await anext(call2), 2)
        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            await anext(call1)
        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            await anext(call2)
        # These next 2 calls should also fail.
        call3 = sample_failing_fn()
        call4 = sample_failing_fn()
        self.assertEqual(await anext(call3), 1)
        self.assertEqual(await anext(call4), 1)
        self.assertEqual(await anext(call3), 2)
        self.assertEqual(await anext(call4), 2)
        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            await anext(call3)
        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            await anext(call4)

    async def test_failing_generator_retry_exceptions(self) -> None:
        counter = Counter()
        failing_line = LineCapture()

        @once.once(retry_exceptions=True)
        async def sample_failing_fn():
            yield counter.get_incremented()
            result = counter.get_incremented()
            yield result
            if result == 2:
                failing_line.record_next_line()
                raise ValueError("we raise an error when result is exactly 2")

        # Both of these calls should return the same results.
        call1 = sample_failing_fn()
        call2 = sample_failing_fn()
        self.assertEqual(await anext(call1), 1)
        self.assertEqual(await anext(call2), 1)
        self.assertEqual(await anext(call1), 2)
        self.assertEqual(await anext(call2), 2)
        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            await anext(call1)
        with assertRaisesWithLineInStackTrace(self, ValueError, failing_line):
            await anext(call2)
        # These next 2 calls should succeed.
        call3 = sample_failing_fn()
        call4 = sample_failing_fn()
        self.assertEqual([i async for i in call3], [3, 4])
        self.assertEqual([i async for i in call4], [3, 4])
        # Subsequent calls should return the original value.
        self.assertEqual([i async for i in sample_failing_fn()], [3, 4])
        self.assertEqual([i async for i in sample_failing_fn()], [3, 4])

    async def test_iterator_is_lazily_evaluted(self) -> None:
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

    async def test_receiving_iterator(self) -> None:
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

    async def test_receiving_iterator_parallel_execution(self) -> None:
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

    async def test_receiving_iterator_parallel_execution_halting(self) -> None:
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
    async def test_iterator_lock_not_held_during_evaluation(self) -> None:
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

    async def test_once_per_class_staticmethod(self) -> None:
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
