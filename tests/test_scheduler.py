"""Structural tests for the Phase-7 cross-chunk dynamic scheduler."""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from eos_trensmuther._scheduler import (
    DynamicSchedulerError,
    execute_dynamic_tasks,
)


@dataclass(frozen=True)
class _Range:
    start: int
    stop: int


@dataclass(frozen=True)
class _Result:
    start: int
    stop: int
    indices: tuple


class DynamicSchedulerTests(unittest.TestCase):
    @staticmethod
    def split(chunk):
        return tuple(
            _Range(start, min(chunk.stop, start + 2))
            for start in range(chunk.start, chunk.stop, 2)
        )

    @staticmethod
    def assemble(chunk, results):
        return _Result(
            chunk.start,
            chunk.stop,
            tuple(index for result in results for index in result.indices),
        )

    def test_out_of_order_tasks_span_chunks_and_publish_parent_owned_results(self):
        parent_thread = threading.get_ident()
        task_started = {}
        task_finished = {}
        published = []
        publication_threads = []
        lock = threading.Lock()

        def run(task):
            with lock:
                task_started[(task.start, task.stop)] = time.perf_counter()
            if task.start == 0:
                time.sleep(0.15)
            else:
                time.sleep(0.005)
            with lock:
                task_finished[(task.start, task.stop)] = time.perf_counter()
            return _Result(task.start, task.stop, tuple(range(task.start, task.stop)))

        def publish(chunk, result):
            publication_threads.append(threading.get_ident())
            published.append((chunk.start, chunk.stop, result.indices))

        chunks = (_Range(0, 4), _Range(4, 8), _Range(8, 12))
        with ThreadPoolExecutor(max_workers=3) as executor:
            report = execute_dynamic_tasks(
                executor,
                chunks,
                split_chunk=self.split,
                run_task=run,
                assemble_chunk=self.assemble,
                publish_chunk=publish,
                maximum_pending_tasks=3,
            )

        self.assertNotEqual(report.publication_order[0], (0, 4))
        self.assertEqual(report.submitted_task_count, 6)
        self.assertEqual(report.completed_task_count, 6)
        self.assertEqual(report.published_chunk_count, 3)
        self.assertLessEqual(report.maximum_pending_tasks_observed, 3)
        self.assertEqual(report.maximum_active_chunks_observed, 3)
        self.assertTrue(
            task_started[(4, 6)] < task_finished[(0, 2)],
            "a later checkpoint chunk must start before the slow first chunk drains",
        )
        self.assertEqual(publication_threads, [parent_thread] * 3)
        flattened = sorted(
            index for _, _, indices in published for index in indices
        )
        self.assertEqual(flattened, list(range(12)))
        for start, stop, indices in published:
            self.assertEqual(indices, tuple(range(start, stop)))

    def test_worker_failure_does_not_publish_incomplete_atomic_chunk(self):
        published = []

        def run(task):
            if task.start == 2:
                raise RuntimeError("synthetic worker failure")
            return _Result(task.start, task.stop, tuple(range(task.start, task.stop)))

        with ThreadPoolExecutor(max_workers=3) as executor:
            with self.assertRaisesRegex(
                DynamicSchedulerError,
                r"worker task failed for range \[2, 4\).*synthetic worker failure",
            ):
                execute_dynamic_tasks(
                    executor,
                    (_Range(0, 6),),
                    split_chunk=self.split,
                    run_task=run,
                    assemble_chunk=self.assemble,
                    publish_chunk=lambda chunk, result: published.append(result),
                    maximum_pending_tasks=3,
                )

        self.assertEqual(published, [])

    def test_invalid_partition_and_result_range_fail_closed(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            with self.assertRaisesRegex(
                DynamicSchedulerError, "do not exactly partition"
            ):
                execute_dynamic_tasks(
                    executor,
                    (_Range(0, 4),),
                    split_chunk=lambda chunk: (_Range(0, 1), _Range(2, 4)),
                    run_task=lambda task: _Result(task.start, task.stop, ()),
                    assemble_chunk=self.assemble,
                    publish_chunk=lambda chunk, result: None,
                    maximum_pending_tasks=1,
                )

        with ThreadPoolExecutor(max_workers=1) as executor:
            with self.assertRaisesRegex(
                DynamicSchedulerError, "worker returned range"
            ):
                execute_dynamic_tasks(
                    executor,
                    (_Range(0, 2),),
                    split_chunk=self.split,
                    run_task=lambda task: _Result(1, 2, (1,)),
                    assemble_chunk=self.assemble,
                    publish_chunk=lambda chunk, result: None,
                    maximum_pending_tasks=1,
                )

    def test_empty_schedule_is_a_valid_no_op(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            report = execute_dynamic_tasks(
                executor,
                (),
                split_chunk=self.split,
                run_task=lambda task: task,
                assemble_chunk=self.assemble,
                publish_chunk=lambda chunk, result: None,
                maximum_pending_tasks=1,
            )
        self.assertEqual(report.submitted_task_count, 0)
        self.assertEqual(report.published_chunk_count, 0)
        self.assertEqual(report.publication_order, ())


if __name__ == "__main__":
    unittest.main()
