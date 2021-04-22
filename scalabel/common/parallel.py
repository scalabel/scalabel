"""Utilities for cpu parallelization."""
from __future__ import annotations

from multiprocessing import Process, Queue, cpu_count

# Disabling unused import becase we need Tuple in typing
from typing import (  # pylint: disable=unused-import
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
)

Inputs = TypeVar("Inputs")
Return = TypeVar("Return")


def run(
    func: Callable[[Inputs], Return],
    q_in: "Queue[Tuple[int, Optional[Tuple[Inputs]]]]",
    q_out: "Queue[Tuple[int, Return]]",
) -> None:
    """Run function on the inputs from the queue."""
    while True:
        i, x = q_in.get()
        if i < 0 or x is None:
            break
        q_out.put((i, func(x[0])))


def pmap(
    func: Callable[[Inputs], Return],
    inputs: Iterable[Inputs],
    nprocs: int = cpu_count(),
) -> List[Return]:
    """Parrell mapping func to arguments.

    Different from the python pool map, this function will not hang if any of
    the processes throws an exception.
    """
    q_in: "Queue[Tuple[int, Optional[Tuple[Inputs]]]]" = Queue(1)
    q_out: "Queue[Tuple[int, Return]]" = Queue()

    proc = [
        Process(target=run, args=(func, q_in, q_out)) for _ in range(nprocs)
    ]
    for p in proc:
        p.daemon = True
        p.start()

    count = 0
    for i, x in enumerate(inputs):
        q_in.put((i, (x,)))
        count += 1
    for _ in range(nprocs):
        q_in.put((-1, None))
    res = [q_out.get() for _ in range(count)]
    for p in proc:
        p.join()

    return [x for i, x in sorted(res)]
