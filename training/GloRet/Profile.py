'''`--profile`: where one pass of the training spends its seconds. The first pool choice (scoring the items) and the first
optimiser step of each head are run under `torch.profiler` ONCE per job, and `<out>/profile/<label>.txt` says what the card did in
that time: the share of the wall seconds a kernel was running, the copies between host and card, and the operations that took
the most of the card's and of the CPU's time. The SECOND pool choice of a head is timed by part instead (`<label>-modules.txt`:
the network's layers one by one, each by the card's clock and the CPU's), without the profiler's own cost, so that the share of
each part of the network can be read. Only the process that is given a directory profiles.
'''
from __future__ import annotations

import os
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch

#: the directory to write to (None: off), how many times each label was met, and the notes of the pass being measured
STATE: Dict[str, Any] = {'dir': None, 'count': {}, 'notes': None}

#: the methods of `LogLoRetNet` that are the parts of a score (`section` of what they call), and of its `cross`, timed as such
SECTIONS: Tuple[str, ...] = ('window_side', 'query_side', 'pair_side')
CROSS_METHODS: Tuple[str, ...] = ('ask', 'answer', 'score', 'read')


def start(directory: Optional[str]) -> None:
    """Profile into `directory` (None: not at all)."""
    STATE['dir'] = directory
    STATE['count'] = {}
    STATE['notes'] = None


def note(text: str) -> None:
    """A line for the report of the pass being measured (nothing when none is)."""
    if STATE['notes'] is not None:
        STATE['notes'].append(text)


def profiled(label: str, net: Optional[torch.nn.Module] = None) -> Any:
    """A context manager that profiles what runs in it the first time `label` is met and, given the network, times its parts
    the second time."""
    if STATE['dir'] is None:
        return nullcontext()
    seen: int = STATE['count'].get(label, 0)
    STATE['count'][label] = seen + 1
    if seen == 0:
        return _profile(label)
    if seen == 1 and net is not None:
        return _module_times(label, net)
    return nullcontext()


def _device_us(event: Any) -> float:
    """The microseconds a profiled event kept the card busy (itself, not what it called)."""
    for name in ('self_device_time_total', 'self_cuda_time_total'):
        if hasattr(event, name):
            return float(getattr(event, name))
    return 0.0


def _card_seconds(events: Any, copies_only: bool = False) -> float:
    """The card's busy seconds in the profile. An operation on the host carries the device time of the kernels it launched and
    the kernels are events of their own: counting both counts every kernel twice, so only the host operations' are summed
    (what the profiler's own `Self CUDA time total` is)."""
    try:
        from torch.autograd import DeviceType                              # noqa: PLC0415
        host: List[Any] = [e for e in events if getattr(e, 'device_type', None) == DeviceType.CPU]
    except ImportError:
        host = []
    chosen: List[Any] = host if host else list(events)
    if copies_only:
        chosen = [e for e in chosen if 'Memcpy' in e.key or 'Memset' in e.key]
    return sum(_device_us(e) for e in chosen) / 1e6


@contextmanager
def _profile(label: str) -> Iterator[None]:
    from torch.profiler import ProfilerActivity, profile                 # noqa: PLC0415
    on_card: bool = torch.cuda.is_available()
    activities: List[Any] = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if on_card else [])
    if on_card:
        torch.cuda.synchronize()
    STATE['notes'] = []
    started: float = time.perf_counter()
    with profile(activities=activities) as prof:
        yield
        if on_card:
            torch.cuda.synchronize()
    wall: float = time.perf_counter() - started
    notes: List[str] = STATE['notes']
    STATE['notes'] = None
    events = prof.key_averages()
    kernels: float = _card_seconds(events)
    copies: float = _card_seconds(events, copies_only=True)
    out_dir: str = os.path.join(str(STATE['dir']), 'profile')
    os.makedirs(out_dir, exist_ok=True)
    sort_card: str = 'self_cuda_time_total' if on_card else 'self_cpu_time_total'
    with open(os.path.join(out_dir, f'{label}.txt'), 'w') as f:
        f.write(f'{label}: {wall:.2f} s of wall time (with the profiler on: slower than the same pass without it)\n')
        f.write(f'  the card ran kernels and copies for {kernels:.2f} s = {100 * kernels / max(wall, 1e-9):.0f}% of it '
                f'(the rest it waited: for the CPU to launch, for a copy, for another stream)\n')
        f.write(f'  of that, copies and memsets (host <-> card, card <-> card) {copies:.2f} s\n')
        for line in notes:
            f.write(f'  {line}\n')
        f.write('\nby the card\'s own time\n' + events.table(sort_by=sort_card, row_limit=25, max_name_column_width=70) + '\n\n')
        f.write('by the CPU\'s own time\n' + events.table(sort_by='self_cpu_time_total', row_limit=25, max_name_column_width=70) + '\n')


class Timer:
    """The calls of named parts, each timed by the card's clock (events on the stream: what the card did between the start and
    the end of the call, its waits for the host's launches inside it included) and by the host's (what the call took to issue)."""

    def __init__(self, on_card: bool) -> None:
        self.on_card: bool = on_card
        self.section: str = ''
        self.calls: List[Tuple[str, Any, Any, float]] = []
        self._stack: List[Tuple[Any, float]] = []

    def begin(self) -> None:
        mark: Any = None
        if self.on_card:
            mark = torch.cuda.Event(enable_timing=True)
            mark.record()
        self._stack.append((mark, time.perf_counter()))

    def end(self, name: str) -> None:
        mark, t0 = self._stack.pop()
        stop: Any = None
        if self.on_card:
            stop = torch.cuda.Event(enable_timing=True)
            stop.record()
        self.calls.append((name, mark, stop, time.perf_counter() - t0))

    def wrap(self, name: str, fn: Callable[..., Any], section: Optional[str] = None) -> Callable[..., Any]:
        """`fn` that is timed under `name` (prefixed by the section it runs in); with `section`, what it calls is timed under it."""
        def run(*args: Any, **kwargs: Any) -> Any:
            outer: str = self.section
            label: str = name if section is not None else (f'{outer}/{name}' if outer else name)
            if section is not None:
                self.section = section
            self.begin()
            try:
                return fn(*args, **kwargs)
            finally:
                self.end(label)
                self.section = outer
        return run

    def table(self) -> List[str]:
        if self.on_card:
            torch.cuda.synchronize()
        totals: Dict[str, List[float]] = {}
        for name, mark, stop, host in self.calls:
            row: List[float] = totals.setdefault(name, [0.0, 0.0, 0.0])
            row[0] += 1
            row[1] += (mark.elapsed_time(stop) / 1e3) if mark is not None else 0.0
            row[2] += host
        order: List[str] = sorted(totals, key=lambda n: (SECTIONS.index(n.split('/')[0]) if n.split('/')[0] in SECTIONS else 9,
                                                         n.count('/'), n))
        width: int = max([len(n) for n in order] + [4])
        out: List[str] = [f'{"part":<{width}}  {"calls":>7}  {"card clock s":>12}  {"host s":>8}']
        out += [f'{n:<{width}}  {totals[n][0]:>7.0f}  {totals[n][1]:>12.2f}  {totals[n][2]:>8.2f}' for n in order]
        return out


@contextmanager
def _module_times(label: str, net: torch.nn.Module) -> Iterator[None]:
    """Time the parts of `net` while the block runs: the three parts of a pair scorer's score (`SECTIONS`) and, inside each, the
    layers it calls (named `section/layer`) and the methods of its cross-attention."""
    on_card: bool = torch.cuda.is_available() and next(net.parameters()).is_cuda
    if on_card:
        torch.cuda.synchronize()
    timer: Timer = Timer(on_card)
    STATE['notes'] = []
    handles: List[Any] = []
    patched: List[Tuple[Any, str]] = []
    sectioned: bool = all(hasattr(net, s) for s in SECTIONS)

    def hook_pair(name: str) -> Tuple[Callable[..., None], Callable[..., None]]:
        return (lambda module, args: timer.begin(),
                lambda module, args, out: timer.end(f'{timer.section}/{name}' if timer.section else name))

    for name, child in net.named_children():
        pre, post = hook_pair(name)
        handles += [child.register_forward_pre_hook(pre), child.register_forward_hook(post)]
    if sectioned:
        for s in SECTIONS:
            setattr(net, s, timer.wrap(s, getattr(net, s), section=s))
            patched.append((net, s))
        for m in CROSS_METHODS:
            setattr(net.cross, m, timer.wrap(f'cross.{m}', getattr(net.cross, m)))
            patched.append((net.cross, m))
    started: float = time.perf_counter()
    try:
        yield
    finally:
        wall: float = time.perf_counter() - started
        for h in handles:
            h.remove()
        for obj, name in patched:
            delattr(obj, name)
        lines: List[str] = timer.table()
        notes: List[str] = STATE['notes'] or []
        STATE['notes'] = None
        out_dir: str = os.path.join(str(STATE['dir']), 'profile')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f'{label}-modules.txt'), 'w') as f:
            f.write(f'{label}: the parts of the network, one pass of {wall:.2f} s of wall time (no profiler on)\n')
            f.write('  card clock: the card\'s time between the start and the end of the part, the gaps in which it waited for the host\n'
                    '  included; host: the seconds the host spent issuing it. A section line (window_side, ...) holds the layers under it.\n')
            for line in notes:
                f.write(f'  {line}\n')
            f.write('\n' + '\n'.join(lines) + '\n')


__all__ = ['SECTIONS', 'Timer', 'note', 'profiled', 'start']
