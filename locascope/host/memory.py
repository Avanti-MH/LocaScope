"""Report how much memory this job is allowed and how much it is using.

On SLURM, /proc/meminfo is a trap: it reports the NODE (1.9 TB on these
nodes), not the job's share. The OOM killer acts on the cgroup, so the cgroup
is the only "used / total" that predicts being killed. The cgroup also keeps
an exact high-water mark, where sacct's MaxRSS is sampled and can miss a
spike -- which matters when the thing being measured is one allocation that
lives for a few seconds.

    from _memprobe import mem_line
    print(mem_line('after base mask'), flush=True)

Everything here degrades to a shorter line rather than raising: this is
instrumentation, and instrumentation that can take the run down is worse than
no instrumentation.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

_GIB = float(1 << 30)


def _read_int(path: str) -> Optional[int]:
    """First whitespace-separated integer in a file, or None if unreadable."""
    try:
        with open(path) as f:
            return int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _node_total_bytes() -> Optional[int]:
    """MemTotal from /proc/meminfo. Only used to recognise 'no limit'."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


_CG_V1_ROOT = '/sys/fs/cgroup/memory'


def _cgroup_v1_chain() -> list:
    """This process's cgroup v1 memory directory, then every ancestor.

    The chain matters because SLURM does not necessarily set the limit where
    the process sits. On these nodes /proc/self/cgroup points at the step
    cgroup, whose memory.limit_in_bytes is the unlimited sentinel, while the
    real --mem constraint lives on an ancestor. Reading only the step made a
    job that was about to be OOM killed report "unlimited".
    """
    try:
        with open('/proc/self/cgroup') as f:
            rel = next(
                (p[2] for p in (l.strip().split(':', 2) for l in f)
                 if len(p) == 3 and 'memory' in p[1].split(',')),
                None,
            )
    except OSError:
        return []
    if rel is None:
        return []

    chain, cur = [], os.path.join(_CG_V1_ROOT, rel.lstrip('/'))
    while cur.startswith(_CG_V1_ROOT):
        if os.path.isdir(cur):
            chain.append(cur)
        if cur == _CG_V1_ROOT:
            break
        cur = os.path.dirname(cur)
    return chain


def cgroup_mem() -> Optional[Tuple[Optional[int], Optional[int], Optional[int]]]:
    """(used, limit, peak) in bytes for the cgroup that actually constrains us.

    Walks up from this process's own cgroup and reports the first ancestor
    with a finite limit -- that is the one the OOM killer enforces, so its
    used/limit is the pair worth watching. Falls back to the innermost cgroup
    when nothing in the chain is limited.

    A v1 limit reads as a near-2**63 sentinel when unlimited; anything at or
    above the node's own MemTotal counts as no limit rather than as a number
    that would make every usage look comfortably safe.
    """
    chain = _cgroup_v1_chain()
    if not chain:
        # cgroup v2 layout, in case this ever runs somewhere else.
        used = _read_int('/sys/fs/cgroup/memory.current')
        if used is None:
            return None
        lim = _read_int('/sys/fs/cgroup/memory.max')   # 'max' -> None via _read_int
        return (used, lim, _read_int('/sys/fs/cgroup/memory.peak'))

    node = _node_total_bytes()
    fallback = None
    for d in chain:
        used = _read_int(os.path.join(d, 'memory.usage_in_bytes'))
        if used is None:
            continue
        lim  = _read_int(os.path.join(d, 'memory.limit_in_bytes'))
        peak = _read_int(os.path.join(d, 'memory.max_usage_in_bytes'))
        if lim is not None and (node is None or lim < node):
            return (used, lim, peak)
        if fallback is None:
            fallback = (used, None, peak)
    return fallback


def rss_bytes() -> Optional[int]:
    """Resident set of THIS process.

    Smaller than the cgroup figure, which also counts page cache and any child
    processes. The gap between the two is what tells you whether a climbing
    cgroup number is your arrays or the kernel caching the WSI file.
    """
    try:
        with open('/proc/self/statm') as f:
            return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, IndexError):
        return None


def vram() -> str:
    """Device-wide VRAM plus torch's own share, or why there is none.

    mem_get_info is what the driver sees, so it includes every process on the
    GPU. memory_allocated is only torch's live tensors and memory_reserved is
    its cache -- the part it is holding but not using, and will not return
    until empty_cache().
    """
    try:
        import torch
    except ImportError:
        return 'vram n/a (no torch)'
    try:
        if not torch.cuda.is_available():
            return 'vram n/a (no cuda)'
        free, total = torch.cuda.mem_get_info()
        return (f'vram {(total - free) / _GIB:.1f}/{total / _GIB:.1f}G '
                f'(torch live={torch.cuda.memory_allocated() / _GIB:.1f} '
                f'cache={torch.cuda.memory_reserved() / _GIB:.1f} '
                f'peak={torch.cuda.max_memory_allocated() / _GIB:.1f})')
    except Exception as e:                       # driver hiccup is not fatal
        return f'vram n/a ({type(e).__name__})'


def mem_line(tag: str = '') -> str:
    """One line: job RAM used/limit/peak, this process's RSS, and VRAM.

    SLURM's --mem=200G means 200 GiB, and everything here is GiB too, so the
    'job' figure can be read straight against what the jobscript asked for.
    """
    parts = [f'[mem {tag}]' if tag else '[mem]']
    cg = cgroup_mem()
    if cg is not None:
        used, lim, peak = cg
        lim_s  = f'{lim / _GIB:.0f}G' if lim else 'unlimited'
        peak_s = f' peak={peak / _GIB:.1f}G' if peak else ''
        parts.append(f'job {used / _GIB:.1f}/{lim_s}{peak_s}')
    rss = rss_bytes()
    if rss is not None:
        parts.append(f'rss {rss / _GIB:.1f}G')
    parts.append(vram())
    return '  '.join(parts)
