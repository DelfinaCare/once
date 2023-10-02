"""Unit tests for once decorators."""
# pylint: disable=missing-function-docstring
import asyncio
import concurrent.futures
import inspect
import sys
import time
import unittest
from unittest import mock
import threading
import weakref
import gc

import once


if sys.version_info.minor < 10:
    print(f"Redefining anext for python 3.{sys.version_info.minor}")

    async def anext(iter, *args, **kwargs):
        return await iter.__anext__(*args, **kwargs)


class Counter:
    """Holding object for a counter.

    If we return an integer directly, it will simply return a copy and
    will not update as the number of calls increases.
    """

    def __init__(self) -> None:
        self.value = 0

    def get_incremented(self) -> int:
        self.value += 1
        return self.value


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


class TestOnce(unittest.TestCase):
    """Unit tests for once decorators."""

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

    def test_iterator(self):
        @once.once
        def yielding_iterator():
            for i in range(3):
                yield i

        self.assertEqual(list(yielding_iterator()), [0, 1, 2])
        self.assertEqual(list(yielding_iterator()), [0, 1, 2])

    def test_threaded_single_function(self):
        counting_fn, counter = generate_once_counter_fn()
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(counting_fn, range(32)))
        self.assertEqual(len(results), 32)
        for r in results:
            self.assertEqual(r, 1)
        self.assertEqual(counter.value, 1)

    def test_threaded_multiple_functions(self):
        counters = []
        fns = []

        for _ in range(4):
            cfn, counter = generate_once_counter_fn()
            counters.append(counter)
            fns.append(cfn)

        promises = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            for cfn in fns:
                for _ in range(16):
                    promises.append(executor.submit(cfn))
            del cfn
            fns.clear()
            for promise in promises:
                self.assertEqual(promise.result(), 1)
        for counter in counters:
            self.assertEqual(counter.value, 1)

    def test_different_fn_do_not_deadlock(self):
        """Ensure different functions use different locks to avoid deadlock."""

        fn2_called = False

        # If fn1 is called first, these functions will deadlock unless they can
        # run in parallel.
        @once.once
        def fn1():
            nonlocal fn2_called
            start = time.time()
            while not fn2_called:
                if time.time() - start > 5:
                    self.fail("Function fn1 deadlocked for 5 seconds.")
                time.sleep(0.01)

        @once.once
        def fn2():
            nonlocal fn2_called
            fn2_called = True

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fn1_promise = executor.submit(fn1)
            executor.submit(fn2)
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

    @mock.patch.object(once, "_new_lock")
    def test_lock_bypass(self, lock_mocker) -> None:
        """Test both with and without lock bypass cache lookup."""

        # We mock the lock to return our specific lock, so we can specifically
        # test behavior with it held.
        lock = threading.Lock()
        lock_mocker.return_value = lock

        counter = Counter()

        @once.once
        def sample_fn() -> int:
            nonlocal counter
            return counter.get_incremented()

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            with lock:
                potential_first_call_promises = [executor.submit(sample_fn) for i in range(32)]
                # Give the promises enough time to finish, if they were not blocked.
                # The test will still pass without this, but we wouldn't be
                # testing anything.
                time.sleep(0.01)
                # At this point, all of the promises will be waiting for the lock,
                # and none of them will have completed.
                for promise in potential_first_call_promises:
                    self.assertFalse(promise.done())
            # Now that we have released the lock, all of these should complete.
            for promise in potential_first_call_promises:
                self.assertEqual(promise.result(), 1)
            self.assertEqual(counter.value, 1)
            # Now that we know the function has already been called, we should
            # be able to get a result without waiting for a lock.
            with lock:
                bypass_lock_promises = [executor.submit(sample_fn) for i in range(32)]
                for promise in bypass_lock_promises:
                    self.assertEqual(promise.result(), 1)
            self.assertEqual(counter.value, 1)

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

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            a_jobs = [executor.submit(a.value) for _ in range(16)]
            b_jobs = [executor.submit(b.value) for _ in range(16)]
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
                self.lock = threading.Lock()
                self.test = test
                self.started = False
                self.counter = Counter()

            @once.once_per_instance
            def run(self) -> int:
                self.started = True
                with self.lock:
                    pass
                return self.counter.get_incremented()

        a = _BlockableClass(self)
        b = _BlockableClass(self)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            with a.lock:
                a_job = executor.submit(a.run)
                while not a.started:
                    pass
                # At this point, the A job has started. However, it cannot
                # complete while we hold its lock. Despite this, we want to ensure
                # that B can still run.
                b_job = executor.submit(b.run)
                # The b_job will deadlock and this will fail if different
                # object executions block each other.
                self.assertEqual(b_job.result(timeout=5), 1)
            self.assertEqual(a_job.result(timeout=5), 1)

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

    async def test_inspect_func(self):
        @once.once
        async def async_func():
            return True

        # Unfortunately these are corrupted by our @once.once.
        # self.assertFalse(inspect.isasyncgenfunction(async_func))
        # self.assertTrue(inspect.iscoroutinefunction(async_func))

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

        # Unfortunately these are corrupted by our @once.once.
        # self.assertTrue(inspect.isasyncgenfunction(async_yielding_iterator))
        # self.assertTrue(inspect.iscoroutinefunction(async_yielding_iterator))

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
            next = yield 1
            while next is not None:
                next = yield next

        gen_1 = async_receiving_iterator()
        gen_2 = async_receiving_iterator()
        self.assertEqual(await gen_1.asend(None), 1)
        self.assertEqual(await gen_1.asend(1), 1)
        self.assertEqual(await gen_1.asend(3), 3)
        self.assertEqual(await gen_2.asend(None), 1)
        self.assertEqual(await gen_2.asend(None), 1)
        self.assertEqual(await gen_2.asend(None), 3)
        self.assertEqual(await gen_2.asend(5), 5)
        self.assertEqual(await anext(gen_2, None), None)
        self.assertEqual(await gen_1.asend(None), 5)
        self.assertEqual(await anext(gen_1, None), None)

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
        
        self.assertEqual(await gen_2.asend(None), 1) # Should return immediately even though task1 is stuck.

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
