'''How many CPU threads one process may use, given what else shares the job.

ONE rule for every entry point -- a bench shard, a training run, the pipeline:

    torch threads (this process) + DataLoader workers (this process)
        <= the job's cpus / the processes the job runs side by side

Nothing enforced it before, and the defaults broke it twice over. torch sizes
its pool to every cpu the process can see, so two shards on 8 cpus ran 16
threads and encoded 7x slower than one (36 against 251 tiles/s per process,
`diag_render_reads.py` B-bench, 2026-10-02); and a training process kept 8
threads beside 8 rendering workers, which tripled its encode (10.6 -> 33 s,
combined train, 2026-10-03). Dividing the cpus fixed both: 251 tiles/s per
shard with 4 threads each, 151 tiles/s training with 1 thread beside 8 workers.

Called by entry points only. A library module that set the thread count would
be deciding for a process it does not own.
'''
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def cpus_available() -> int:
    """The cpus this process may run on (the job's allocation, not the node's)."""
    return len(os.sched_getaffinity(0))


@dataclass(frozen=True)
class CpuBudget:
    """One process's share of the job's cpus, and how it is split.

    `share` is cpus // processes. `workers` DataLoader workers come out of it,
    and what is left -- at least one -- is the process's own torch threads. A
    worker count larger than the share is not refused: rendering workers block
    on reads and the measured best for training is 8 workers and 1 thread on
    8 cpus. It is reported, so a run says what it was given."""
    cpus: int
    processes: int
    workers: int

    @classmethod
    def for_job(cls, processes: int = 1, workers: Optional[int] = None,
                cpus: Optional[int] = None) -> 'CpuBudget':
        """`workers=None` gives the share minus one thread to the workers --
        the reader configuration measured for the window bench (2 shards on 8
        cpus: 3 workers and 1 thread each)."""
        cpus = cpus or cpus_available()
        processes = max(1, int(processes))
        share = max(1, cpus // processes)
        if workers is None:
            workers = max(0, share - 1)
        return cls(cpus=cpus, processes=processes, workers=max(0, int(workers)))

    @property
    def share(self) -> int:
        return max(1, self.cpus // self.processes)

    @property
    def threads(self) -> int:
        """torch threads for this process."""
        return max(1, self.share - self.workers)

    def apply(self) -> 'CpuBudget':
        """Set this process's torch thread pool. Returns self, to chain."""
        import torch                                                # noqa: PLC0415
        torch.set_num_threads(self.threads)
        return self

    def line(self) -> str:
        over = self.threads + self.workers > self.share
        return (f'cpu budget: {self.cpus} cpus / {self.processes} process(es) = '
                f'{self.share} each -> {self.workers} workers + {self.threads} '
                f'torch thread(s){"  (workers exceed the share)" if over else ""}')
