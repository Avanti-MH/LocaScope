'''What the job's GPUs, CPUs and memory were used for, measured from inside the job.

A big allocation that is not used is one that should be given back, so every epoch reports how busy each resource was
and what the job asked for. One `UsageMeter` per node (the process of local rank 0) samples every `interval` seconds
on a thread of its own:

    GPU      `nvidia-smi --query-gpu`: the share of the time a kernel was running (NOT how much of the card's
             arithmetic was used: a card can read 90% and be mostly waiting inside small kernels), and the memory
             in use, per card
    CPU      /proc/stat: the share of the time the CPUs this job may run on were not idle
    memory   the job's cgroup (memory.current of the job, against memory.max, page cache and /dev/shm included, which
             is what the kernel kills a job for), else the node's MemTotal - MemAvailable; and /dev/shm on its own

The samples are the program's own counters, read from files and one `nvidia-smi` call a sample; nothing else on the
node is looked at. `summary()` gives the figures since the last call, `advice()` says in words what could be handed
back.
'''
from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ══════════════════════════════════════════════════════════════════════════════
#  reading the counters
# ══════════════════════════════════════════════════════════════════════════════

def parse_gpus(text: str) -> List[Tuple[int, float, float, float]]:
    """`(index, utilisation %, memory used GB, memory total GB)` from `nvidia-smi --query-gpu=index,utilization.gpu,
    memory.used,memory.total --format=csv,noheader,nounits` (memory in MiB)."""
    out: List[Tuple[int, float, float, float]] = []
    for line in text.strip().splitlines():
        parts: List[str] = [p.strip() for p in line.split(',')]
        if len(parts) < 4:
            continue
        try:
            out.append((int(parts[0]), float(parts[1]), float(parts[2]) * 1.048576 / 1e3, float(parts[3]) * 1.048576 / 1e3))
        except ValueError:
            continue                                  # '[N/A]' on a card that does not say
    return out


def read_gpus() -> List[Tuple[int, float, float, float]]:
    """The cards this job can see, or [] when `nvidia-smi` is not there or does not answer."""
    try:
        done = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                               '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_gpus(done.stdout) if done.returncode == 0 else []


def parse_cpu_times(text: str, cpus: Sequence[int]) -> Tuple[float, float, float]:
    """`(busy, total, iowait)` jiffies of the CPUs `cpus` from the text of /proc/stat (user nice system idle iowait irq
    softirq steal): idle and iowait are not busy; iowait is a CPU waiting for a read or write to come back (a slide
    read from a slow disk, a full /dev/shm), which `busy` alone would show as an idle CPU."""
    busy = total = waiting = 0.0
    wanted = {f'cpu{c}' for c in cpus}
    for line in text.splitlines():
        fields: List[str] = line.split()
        if not fields or fields[0] not in wanted:
            continue
        v: List[float] = [float(x) for x in fields[1:9]] + [0.0] * (8 - len(fields[1:9]))
        idle: float = v[3] + v[4]
        total += sum(v)
        busy += sum(v) - idle
        waiting += v[4]
    return busy, total, waiting


def read_cpu_times(cpus: Sequence[int]) -> Tuple[float, float, float]:
    try:
        with open('/proc/stat') as f:
            return parse_cpu_times(f.read(), cpus)
    except OSError:
        return 0.0, 0.0, 0.0


def cgroup_memory() -> Optional[Tuple[float, float]]:
    """`(in use GB, limit GB)` of the job's cgroup (cgroup v2), or None when there is none or no limit is set on
    it or on a parent of it (up to four levels)."""
    try:
        with open('/proc/self/cgroup') as f:
            rel: str = next(line.strip().split(':', 2)[2] for line in f if line.startswith('0::'))
    except (OSError, StopIteration, IndexError):
        return None
    path: str = os.path.join('/sys/fs/cgroup', rel.lstrip('/'))
    for _ in range(5):
        try:
            with open(os.path.join(path, 'memory.max')) as f:
                limit: str = f.read().strip()
            if limit != 'max':
                with open(os.path.join(path, 'memory.current')) as f:
                    return int(f.read()) / 1e9, int(limit) / 1e9
        except (OSError, ValueError):
            pass
        parent: str = os.path.dirname(path)
        if parent == path or len(parent) <= len('/sys/fs/cgroup'):
            break
        path = parent
    return None


def node_memory() -> Tuple[float, float, float]:
    """`(in use GB, total GB, /dev/shm in use GB)` of the node: MemTotal - MemAvailable, and the shared-memory
    filesystem."""
    used = total = 0.0
    try:
        with open('/proc/meminfo') as f:
            info: Dict[str, float] = {line.split(':')[0]: float(line.split()[1]) for line in f if ':' in line}
        total = info.get('MemTotal', 0.0) * 1024 / 1e9
        used = total - info.get('MemAvailable', 0.0) * 1024 / 1e9
    except (OSError, ValueError, IndexError):
        pass
    shm: float = 0.0
    try:
        st = os.statvfs('/dev/shm')
        shm = (st.f_blocks - st.f_bfree) * st.f_frsize / 1e9
    except OSError:
        pass
    return used, total, shm


# ══════════════════════════════════════════════════════════════════════════════
#  the meter
# ══════════════════════════════════════════════════════════════════════════════

class UsageMeter:
    """Samples the node's GPUs, CPUs and memory on a thread. `summary()` is the figures of the samples since the last
    call (and starts a new period); `peek(since)` the figures since a time without ending the period; `stop()` ends
    the thread. With `on_report`, the figures of the last `report_every` seconds are handed to it as they pass, so a
    long first epoch is seen while it runs and not only at its end."""

    def __init__(self, interval: float = 20.0, report_every: float = 0.0,
                 on_report: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        self.interval: float = float(interval)
        self.report_every: float = float(report_every)
        self.on_report: Optional[Callable[[Dict[str, Any]], None]] = on_report
        self.cpus: List[int] = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
        self._lock: threading.Lock = threading.Lock()
        self._stop: threading.Event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._prev_cpu: Tuple[float, float, float] = read_cpu_times(self.cpus)
        self._reset()
        self.host: str = socket.gethostname()

    def _reset(self) -> None:
        self.started: float = time.time()
        #: one record per sample: time, [(card, util %, memory GB, total GB)], CPU busy %, memory GB, limit GB, shm GB
        self.records: List[Tuple[Any, ...]] = []

    def start(self) -> 'UsageMeter':
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        last: float = time.time()
        while not self._stop.wait(self.interval):
            self.sample()
            if self.on_report is not None and self.report_every > 0 and time.time() - last >= self.report_every:
                self.on_report(self.peek(last))
                last = time.time()

    def sample(self) -> None:
        gpus = read_gpus()
        busy, total, waiting = read_cpu_times(self.cpus)
        d_busy, d_total, d_wait = busy - self._prev_cpu[0], total - self._prev_cpu[1], waiting - self._prev_cpu[2]
        self._prev_cpu = (busy, total, waiting)
        cg = cgroup_memory()
        used, node_total, shm = node_memory()
        record = (time.time(), gpus, 100.0 * d_busy / d_total if d_total > 0 else float('nan'),
                  cg[0] if cg else used, cg[1] if cg else node_total, shm,
                  100.0 * d_wait / d_total if d_total > 0 else float('nan'))
        with self._lock:
            self.records.append(record)

    def _digest(self, records: Sequence[Tuple[Any, ...]],
                since: float) -> Dict[str, Any]:
        util: Dict[int, List[float]] = {}
        mem: Dict[int, List[float]] = {}
        total: Dict[int, float] = {}
        for _, gpus, *_ in records:
            for index, u, m, t in gpus:
                util.setdefault(index, []).append(u)
                mem.setdefault(index, []).append(m)
                total[index] = t
        cpu: List[float] = [r[2] for r in records if r[2] == r[2]]
        shm: List[float] = [r[5] for r in records]
        wait: List[float] = [r[6] for r in records if r[6] == r[6]]
        used: List[float] = [r[3] for r in records]
        return {
            'host': self.host, 'seconds': time.time() - since, 'samples': len(records),
            'gpus': {i: {'util_mean': sum(u) / len(u), 'util_max': max(u), 'mem_peak': max(mem[i]),
                         'mem_total': total[i]} for i, u in util.items() if u},
            'cpus': len(self.cpus), 'cpu_mean': sum(cpu) / len(cpu) if cpu else float('nan'),
            'cpu_peak': max(cpu) if cpu else float('nan'),
            'io_mean': sum(wait) / len(wait) if wait else float('nan'), 'io_peak': max(wait) if wait else float('nan'),
            'mem_peak': max(used) if used else float('nan'), 'mem_limit': records[-1][4] if records else 0.0,
            'shm_peak': max(shm) if shm else float('nan')}

    def summary(self) -> Dict[str, Any]:
        """The figures since the last call: GPUs (mean and highest utilisation %, memory peak GB, per card), the
        CPUs this job runs on (mean and peak busy %), memory (peak GB, its limit) and /dev/shm peak, and the seconds
        and number of samples they are over. Starts a new period."""
        with self._lock:
            out: Dict[str, Any] = self._digest(self.records, self.started)
            self._reset()
        return out

    def peek(self, since: float) -> Dict[str, Any]:
        """The same figures over the samples taken at or after `since` (a time), and the period goes on."""
        with self._lock:
            return self._digest([r for r in self.records if r[0] >= since], since)


def usage_lines(summary: Dict[str, Any]) -> List[str]:
    """One node's figures as log lines."""
    g: Dict[int, Dict[str, float]] = summary['gpus']
    lines: List[str] = [f'    node {summary["host"]}: {summary["samples"]} samples over {summary["seconds"]:.0f} s']
    if g:
        means: List[float] = [v['util_mean'] for v in g.values()]
        lines.append(f'      GPU    busy (a kernel running) mean {sum(means) / len(means):.0f}% '
                     f'(cards {min(means):.0f}% .. {max(means):.0f}%), memory peak '
                     f'{max(v["mem_peak"] for v in g.values()):.0f} of {max(v["mem_total"] for v in g.values()):.0f} GB '
                     f'a card, {len(g)} cards')
    else:
        lines.append('      GPU    not read (no nvidia-smi answer)')
    lines.append(f'      CPU    {summary["cpus"]} CPUs: busy mean {summary["cpu_mean"]:.0f}%, peak {summary["cpu_peak"]:.0f}%; '
                 f'waiting for a read or write mean {summary.get("io_mean", float("nan")):.0f}%, peak {summary.get("io_peak", float("nan")):.0f}%')
    lines.append(f'      memory peak {summary["mem_peak"]:.0f} of {summary["mem_limit"]:.0f} GB, '
                 f'/dev/shm peak {summary["shm_peak"]:.0f} GB')
    return lines


def usage_line(summary: Dict[str, Any]) -> str:
    """One node's figures as ONE line (what `usage_lines` says in four): GPUs busy and their memory peak, CPUs busy, the node's
    memory and /dev/shm peaks, and the time spent waiting for a read or write."""
    g: Dict[int, Dict[str, float]] = summary['gpus']
    if g:
        means: List[float] = [v['util_mean'] for v in g.values()]
        gpu: str = (f'GPU {sum(means) / len(means):.0f}% busy' + (f' ({min(means):.0f}..{max(means):.0f}% over {len(g)} cards)' if len(g) > 1 else '')
                    + f', memory peak {max(v["mem_peak"] for v in g.values()):.0f}/{max(v["mem_total"] for v in g.values()):.0f} GB')
    else:
        gpu = 'GPU not read'
    return (f'{summary["host"]} · {gpu} · CPU {summary["cpu_mean"]:.0f}% busy (peak {summary["cpu_peak"]:.0f}%, '
            f'io wait {summary.get("io_mean", float("nan")):.0f}%) · RAM peak {summary["mem_peak"]:.0f}/{summary["mem_limit"]:.0f} GB · '
            f'/dev/shm {summary["shm_peak"]:.0f} GB')


def advice(summary: Dict[str, Any]) -> List[str]:
    """What this node's figures say could be handed back. Over one epoch's samples, which include the first epoch's
    encoding, validation and waiting: read it over several epochs before cutting."""
    out: List[str] = []
    g: Dict[int, Dict[str, float]] = summary['gpus']
    if g:
        mean: float = sum(v['util_mean'] for v in g.values()) / len(g)
        peak: float = max(v['mem_peak'] for v in g.values())
        if mean < 40.0:
            out.append(f'GPUs are busy {mean:.0f}% of the time: the job waits for data (see "waited for a visit" '
                       f'in the time line); fewer cards would take about as long')
        if g and peak < 0.3 * max(v['mem_total'] for v in g.values()):
            out.append(f'GPU memory peak {peak:.0f} GB of {max(v["mem_total"] for v in g.values()):.0f}: a smaller '
                       f'card would do (not asked for here), or a larger step')
    io_mean: float = summary.get('io_mean', float('nan'))
    if io_mean == io_mean and io_mean >= 10.0:
        out.append(f'the CPUs wait for a read or write {io_mean:.0f}% of the time: slow reads (the slides\' storage, '
                   f'other jobs reading it) hold the job up, not the cards or the CPUs -- more of either would not help')
    cpu_peak: float = summary['cpu_peak']
    if cpu_peak == cpu_peak and cpu_peak < 50.0:
        need: int = max(int(summary['cpus'] * cpu_peak / 100.0 * 1.5) + 1, 1)
        out.append(f'CPUs peak at {cpu_peak:.0f}% of {summary["cpus"]}: about {need} would do')
    limit, peak_mem = summary['mem_limit'], summary['mem_peak']
    if limit and peak_mem == peak_mem and peak_mem < 0.4 * limit:
        out.append(f'memory peak {peak_mem:.0f} of {limit:.0f} GB: about {int(peak_mem * 1.5) + 1}G would do')
    return out


__all__ = ['UsageMeter', 'advice', 'cgroup_memory', 'node_memory', 'parse_cpu_times', 'parse_gpus', 'read_gpus',
           'usage_line', 'usage_lines']
