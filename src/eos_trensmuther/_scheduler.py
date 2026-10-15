"""Private bounded scheduler for parent-owned atomic chunk publication.

Worker tasks are smaller than checkpoint chunks.  This module keeps tasks
from several checkpoint chunks in flight, accepts out-of-order completion,
and calls the publication callback only after one complete outer chunk has
been deterministically reassembled in the parent process.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Sequence, Tuple, Type

from concurrent.futures import FIRST_COMPLETED, wait


class DynamicSchedulerError(RuntimeError):
    """Raised when worker-task scheduling or coverage is inconsistent."""


@dataclass(frozen=True)
class DynamicScheduleReport:
    """Bounded scheduling observations for one pool lifetime."""

    submitted_task_count: int
    completed_task_count: int
    published_chunk_count: int
    maximum_pending_tasks_observed: int
    maximum_active_chunks_observed: int
    publication_order: Tuple[Tuple[int, int], ...]


@dataclass
class _ActiveChunk:
    chunk: Any
    tasks: Tuple[Any, ...]
    task_ranges: Tuple[Tuple[int, int], ...]
    next_task: int = 0
    pending_tasks: int = 0
    completed_tasks: int = 0
    results: Dict[Tuple[int, int], Any] = None

    def __post_init__(self) -> None:
        if self.results is None:
            self.results = {}


def _range_of(value: Any, name: str) -> Tuple[int, int]:
    try:
        start = value.start
        stop = value.stop
    except AttributeError as exc:
        raise DynamicSchedulerError(
            "%s must expose integer start/stop attributes" % name
        ) from exc
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or stop <= start
    ):
        raise DynamicSchedulerError(
            "%s must expose a positive half-open integer range" % name
        )
    return int(start), int(stop)


def _validate_partition(
    chunk_range: Tuple[int, int], tasks: Sequence[Any]
) -> Tuple[Tuple[int, int], ...]:
    task_ranges = tuple(_range_of(task, "worker task") for task in tasks)
    if (
        not task_ranges
        or task_ranges[0][0] != chunk_range[0]
        or task_ranges[-1][1] != chunk_range[1]
        or any(
            left[1] != right[0]
            for left, right in zip(task_ranges[:-1], task_ranges[1:])
        )
    ):
        raise DynamicSchedulerError(
            "worker tasks do not exactly partition atomic chunk [%d, %d)"
            % chunk_range
        )
    return task_ranges


def execute_dynamic_tasks(
    executor: Any,
    chunks: Iterable[Any],
    *,
    split_chunk: Callable[[Any], Sequence[Any]],
    run_task: Callable[[Any], Any],
    assemble_chunk: Callable[[Any, Sequence[Any]], Any],
    publish_chunk: Callable[[Any, Any], None],
    maximum_pending_tasks: int,
    failure_type: Type[Exception] = DynamicSchedulerError,
) -> DynamicScheduleReport:
    """Execute bounded small tasks across multiple atomic chunks.

    ``publish_chunk`` runs only in the caller after all tasks belonging to an
    outer chunk have returned and ``assemble_chunk`` has restored canonical
    range order.  A worker exception cancels all not-yet-started futures and
    aborts without publishing the incomplete chunk.
    """

    if (
        isinstance(maximum_pending_tasks, bool)
        or not isinstance(maximum_pending_tasks, int)
        or maximum_pending_tasks <= 0
    ):
        raise ValueError("maximum_pending_tasks must be a positive integer")
    if not isinstance(failure_type, type) or not issubclass(
        failure_type, Exception
    ):
        raise TypeError("failure_type must be an Exception class")

    chunk_iterator = iter(chunks)
    active: Dict[Tuple[int, int], _ActiveChunk] = {}
    seen_chunks = set()
    ready = deque()
    pending: Dict[Any, Tuple[Tuple[int, int], Tuple[int, int]]] = {}
    exhausted = False
    submitted = 0
    completed = 0
    published = 0
    maximum_pending_observed = 0
    maximum_active_observed = 0
    publication_order = []

    def cancel_pending() -> None:
        for future in tuple(pending):
            future.cancel()

    def activate_one() -> bool:
        nonlocal exhausted, maximum_active_observed
        if exhausted:
            return False
        try:
            chunk = next(chunk_iterator)
        except StopIteration:
            exhausted = True
            return False
        chunk_range = _range_of(chunk, "atomic chunk")
        if chunk_range in seen_chunks:
            raise DynamicSchedulerError(
                "atomic chunk [%d, %d) was scheduled more than once"
                % chunk_range
            )
        tasks = tuple(split_chunk(chunk))
        task_ranges = _validate_partition(chunk_range, tasks)
        active[chunk_range] = _ActiveChunk(chunk, tasks, task_ranges)
        seen_chunks.add(chunk_range)
        ready.append(chunk_range)
        maximum_active_observed = max(maximum_active_observed, len(active))
        return True

    def activate_to_bound() -> None:
        while len(active) < maximum_pending_tasks and activate_one():
            pass

    def submit_to_bound() -> None:
        nonlocal submitted, maximum_pending_observed
        while ready and len(pending) < maximum_pending_tasks:
            chunk_range = ready.popleft()
            state = active.get(chunk_range)
            if state is None or state.next_task >= len(state.tasks):
                continue
            index = state.next_task
            task = state.tasks[index]
            task_range = state.task_ranges[index]
            try:
                future = executor.submit(run_task, task)
            except Exception as exc:
                cancel_pending()
                raise failure_type(
                    "could not submit worker task [%d, %d) for atomic chunk "
                    "[%d, %d): %s"
                    % (
                        task_range[0],
                        task_range[1],
                        chunk_range[0],
                        chunk_range[1],
                        exc,
                    )
                ) from exc
            pending[future] = (chunk_range, task_range)
            state.next_task += 1
            state.pending_tasks += 1
            submitted += 1
            if state.next_task < len(state.tasks):
                ready.append(chunk_range)
            maximum_pending_observed = max(
                maximum_pending_observed, len(pending)
            )

    activate_to_bound()
    submit_to_bound()
    while pending:
        done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
        for future in sorted(done, key=lambda item: pending[item]):
            chunk_range, task_range = pending.pop(future)
            state = active.get(chunk_range)
            if state is None:
                cancel_pending()
                raise DynamicSchedulerError(
                    "worker returned for an inactive atomic chunk"
                )
            try:
                result = future.result()
            except Exception as exc:
                cancel_pending()
                raise failure_type(
                    "worker task failed for range [%d, %d) within atomic "
                    "chunk [%d, %d): %s"
                    % (
                        task_range[0],
                        task_range[1],
                        chunk_range[0],
                        chunk_range[1],
                        exc,
                    )
                ) from exc
            result_range = _range_of(result, "worker result")
            if result_range != task_range:
                cancel_pending()
                raise failure_type(
                    "worker returned range [%d, %d) for task [%d, %d)"
                    % (
                        result_range[0],
                        result_range[1],
                        task_range[0],
                        task_range[1],
                    )
                )
            if task_range in state.results:
                cancel_pending()
                raise DynamicSchedulerError(
                    "worker task [%d, %d) returned twice" % task_range
                )
            state.results[task_range] = result
            state.pending_tasks -= 1
            state.completed_tasks += 1
            completed += 1

            if state.completed_tasks == len(state.tasks):
                if (
                    state.next_task != len(state.tasks)
                    or state.pending_tasks != 0
                    or len(state.results) != len(state.tasks)
                ):
                    cancel_pending()
                    raise DynamicSchedulerError(
                        "atomic chunk completed with inconsistent task accounting"
                    )
                ordered = tuple(
                    state.results[item] for item in state.task_ranges
                )
                assembled = assemble_chunk(state.chunk, ordered)
                if _range_of(assembled, "assembled chunk") != chunk_range:
                    cancel_pending()
                    raise DynamicSchedulerError(
                        "assembled result does not match its atomic chunk"
                    )
                publish_chunk(state.chunk, assembled)
                publication_order.append(chunk_range)
                published += 1
                del active[chunk_range]

        activate_to_bound()
        submit_to_bound()

    if active or ready or submitted != completed or published != len(seen_chunks):
        raise DynamicSchedulerError(
            "dynamic scheduler stopped without complete task/chunk coverage"
        )
    return DynamicScheduleReport(
        submitted_task_count=submitted,
        completed_task_count=completed,
        published_chunk_count=published,
        maximum_pending_tasks_observed=maximum_pending_observed,
        maximum_active_chunks_observed=maximum_active_observed,
        publication_order=tuple(publication_order),
    )


__all__ = (
    "DynamicScheduleReport",
    "DynamicSchedulerError",
    "execute_dynamic_tasks",
)
