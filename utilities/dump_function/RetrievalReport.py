"""One definition of every retrieval metric, and the tables that print them.

Two benches ask the same question at two scales -- bench_window_retrieval scores
a whole FoV window through stage 2, bench_tile_retrieval scores a single tile
against a store -- and both want the same answer shape: does this arm beat the
baseline, and how often is the truth inside a candidate budget.

This file exists so that "@1%" means one thing. When the two benches each held
their own copy, nothing would have flagged a divergence: both would print a
column headed @1%, both numbers would look plausible, and a comparison between
the two benches would be silently invalid. A metric that is defined twice is a
metric that is eventually defined differently.

What a caller has to supply
---------------------------
A list of dicts, one per (query, arm). Nothing here opens a store or a slide:

    arm           str    which configuration produced this row
    slide         str    grouping key
    level         int    grouping key
    fov_id        int    identifies the QUERY, so arms can be paired on it
    pool          int    how many candidates this query was ranked against
    rank_main     int    rank of the answer at the nearest main grid point
    rank_overlap  int    rank of the answer at the nearest overlap point
    d_main        float  distance to that main point
    d_overlap     float  distance to that overlap point

A bench with only one notion of the answer sets rank_overlap = rank_main and
d_overlap = d_main; truth and fine then coincide and gap@k is zero throughout,
which is the honest reading -- there is no second grid for retrieval to land on
instead.

The metrics
-----------
  truth_rank        the GEOMETRICALLY CLOSER grid point's rank. The metric of
                    record: it is the one that cannot be satisfied by finding a
                    different tile that happens to overlap.

  fine_rank         the better-ranked of the two. Strictly easier than truth.

  hit@k             truth_rank <= k as a rate over a group.

  gap@k             fine@k - truth@k. Non-negative by construction, since
                    fine_rank <= truth_rank always. Reads as "retrieval put the
                    OTHER grid point in the top k but not the geometrically
                    closer one" -- right to within half a tile, wrong member of
                    the overlapping pair. Large values mean the strict truth
                    definition is doing much of the work and the whole bench
                    should be read differently.

  k@f%              max(1, ceil(f * pool)). Per (slide, level), because pool
                    varies by orders of magnitude across levels.

  top@f%            truth_rank <= k@f%, as a rate. EACH query uses its own
                    k@f% and the hits are then averaged, so this is the one
                    level metric that survives aggregation: its null is
                    constant at f whatever the pool size, while rank@100 scores
                    100/pool -- 35% at a small level, 0.13% at a large one.

                    That holds while f * pool >= 1. Below it the max(1, ...)
                    takes over, k is 1 whatever f says, and the null goes back
                    to 1/pool. See K_FRACTIONS.

  W / L / T         paired against the baseline, per query, on truth_rank.
                    The pairing is what makes it immune to pool size: both arms
                    saw one query and one candidate pool, and only the scoring
                    differed.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

#: Fixed candidate budgets. Meaningful only where pool is comparable, so a
#: caller prints these within one (slide, level) or one level, never across.
K_FIXED = (1, 5, 10, 20, 30, 50, 100)

#: Pool fractions. Their null is constant at f, which is what lets them cross
#: levels where K_FIXED cannot.
#:
#: 0.01% is the exception and has to be read with the pool beside it. k@f% is
#: max(1, ceil(f * pool)), so the floor of 1 bites as soon as the pool is under
#: 10,000 and the column stops meaning what its name says. It is here because
#: the largest level is where the pool is big enough for the other three to
#: saturate -- @0.1% already sits near 78% there, so it separates nothing.
#: Read it per (slide, level); aggregated over levels it averages a genuine
#: 0.01% criterion with truth@1, which is not one criterion.
K_FRACTIONS = (0.0001, 0.001, 0.01, 0.10)


def frac_label(fraction: float) -> str:
    """0.0001 -> '@0.01%'.

    Header text and dict key are both derived from this, so a fraction cannot
    be printed under another one's name.
    """
    return f'@{fraction * 100:g}%'


# ── the two ranks ─────────────────────────────────────────────────────────────

def truth_rank(row: dict) -> int:
    """The geometrically closer grid point's rank. The metric of record."""
    return (row['rank_main'] if row['d_main'] <= row['d_overlap']
            else row['rank_overlap'])


def fine_rank(row: dict) -> int:
    """The better-ranked of the two. Strictly easier than truth."""
    return min(row['rank_main'], row['rank_overlap'])


def k_at(fraction: float, pool: int) -> int:
    """max(1, ceil(f * pool)). Per query, because pool is per (slide, level)."""
    return max(1, int(math.ceil(fraction * pool)))


# ── grouping and pairing ──────────────────────────────────────────────────────

def group_by(rows: list, keys: Sequence[str]) -> Dict[tuple, list]:
    out: Dict[tuple, list] = {}
    for row in rows:
        out.setdefault(tuple(row[k] for k in keys), []).append(row)
    return out


def attach_baseline(rows: list, baseline: str) -> list:
    """Give every row the baseline's truth rank for the SAME query.

    Rows whose query has no baseline are dropped and counted out loud: a paired
    statistic computed over a different set of queries than the baseline saw is
    not paired, and the difference would show up only as a slightly odd win%.
    """
    base = {(r['slide'], r['level'], r['fov_id']): truth_rank(r)
            for r in rows if r['arm'] == baseline}
    missing = 0
    for row in rows:
        key = (row['slide'], row['level'], row['fov_id'])
        if key in base:
            row['base_truth'] = base[key]
        else:
            missing += 1
    if missing:
        print(f'  WARNING {missing} rows have no baseline for their query')
    return [r for r in rows if 'base_truth' in r]


# ── statistics ────────────────────────────────────────────────────────────────

def paired_stats(rows: list) -> dict:
    """W / L / T / win% / ratio quartiles / top@f%, for one arm in one group."""
    wins = sum(1 for r in rows if truth_rank(r) < r['base_truth'])
    losses = sum(1 for r in rows if truth_rank(r) > r['base_truth'])
    ties = len(rows) - wins - losses
    ratios = np.array([r['base_truth'] / truth_rank(r) for r in rows],
                      dtype=float)
    stats = {'n': len(rows), 'W': wins, 'L': losses, 'T': ties,
             'n_cmp': wins + losses,
             'win_pct': wins / (wins + losses) if wins + losses else float('nan'),
             'ratio_q1': float(np.percentile(ratios, 25)),
             'ratio_med': float(np.median(ratios)),
             'ratio_q3': float(np.percentile(ratios, 75))}
    for fraction in K_FRACTIONS:
        hits = [truth_rank(r) <= k_at(fraction, r['pool']) for r in rows]
        stats[f'top{fraction}'] = float(np.mean(hits))
    return stats


def absolute_stats(rows: list) -> dict:
    """Only valid where `pool` is a single number: ranks, fixed k, gap@k."""
    truths = np.array([truth_rank(r) for r in rows], dtype=float)
    fines = np.array([fine_rank(r) for r in rows], dtype=float)
    stats = {'med_rank': float(np.median(truths)),
             'p90_rank': float(np.percentile(truths, 90))}
    for k in K_FIXED:
        stats[f'truth@{k}'] = float(np.mean(truths <= k))
        stats[f'gap@{k}'] = float(np.mean(fines <= k) - np.mean(truths <= k))
    return stats


# ── tables ────────────────────────────────────────────────────────────────────

def pct(value: float) -> str:
    return '   -' if not np.isfinite(value) else f'{value * 100:3.0f}%'


def grid_table(headers: Sequence[str], rows: Sequence[Sequence], *,
               aligns: Optional[str] = None, rule_after: Sequence[int] = (),
               emit: Callable[[str], None] = print) -> None:
    """A table with grid lines, every column as wide as its widest cell.

        +-----+------+
        | arm | wins |
        +-----+------+
        | a   |    4 |
        +-----+------+

    `aligns` is one letter per column, `l` or `r`; the default is the first
    column left and the rest right, which is a label and then numbers. A rule is
    drawn under the header and at the top and bottom, and after each row index in
    `rule_after` (a group boundary). Cells are strings, or anything `str` makes
    one of; widths come from the cells, so a longer name moves nothing else.
    """
    cells = [[str(c) for c in row] for row in rows]
    n = len(headers)
    aligns = aligns or 'l' + 'r' * (n - 1)
    widths = [max([len(str(headers[i]))] + [len(row[i]) for row in cells])
              for i in range(n)]
    rule = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'

    def line(values) -> str:
        return '| ' + ' | '.join(
            str(v).ljust(w) if a == 'l' else str(v).rjust(w)
            for v, w, a in zip(values, widths, aligns)) + ' |'

    emit(rule)
    emit(line(headers))
    emit(rule)
    last = len(cells) - 1
    for i, row in enumerate(cells):
        emit(line(row))
        if i in rule_after and i != last:
            emit(rule)
    emit(rule)


#: Two downsamples within this ratio are the same scale. A pyramid steps by 2x at
#: the least, so 5% never merges two levels, and it takes in the rounding a
#: non-integer pyramid carries (4.000022, 4.02) as one scale.
DS_SAME = 1.05


def group_levels(rows: list) -> Dict[object, list]:
    """Rows grouped by the scale they were measured at: `ds` values within
    `DS_SAME` of each other are one group, keyed by the group's median ds. Rows
    with no `ds` fall back to the level index.

    A level index is not a scale -- level 1 is ds 4 on a 4x pyramid and ds 2 on a
    2x one -- so grouping by it pools queries whose pools and difficulty have
    nothing in common."""
    has_ds = [r for r in rows if r.get('ds') not in (None, '')]
    if len(has_ds) != len(rows):
        out: Dict[object, list] = {}
        for row in rows:
            out.setdefault(row['level'], []).append(row)
        return dict(sorted(out.items()))
    groups: list = []          # [(first ds of the group, [rows])]
    for row in sorted(rows, key=lambda r: float(r['ds'])):
        d = float(row['ds'])
        if groups and d <= groups[-1][0] * DS_SAME:
            groups[-1][1].append(row)
        else:
            groups.append((d, [row]))
    out = {}
    for _, members in groups:
        ds_sorted = sorted(float(r['ds']) for r in members)
        out[float(f'{ds_sorted[len(ds_sorted) // 2]:.4g}')] = members
    return out


def print_level_heading(rows: list, level: object,
                        emit: Callable[[str], None] = print,
                        extras: Sequence[tuple] = ()) -> None:
    """`ds 4 -- 2 slides, 10 queries per arm` and one row per slide
    with how many queries it has and how many candidate windows it holds.

    `extras` are `(header, row key)` pairs for more candidate counts a bench
    stores (`pool_offset`, `pool_all`); a column appears only if some row has
    that value, and a slide with none shows `-`."""
    by_slide = group_by(rows, ['slide'])
    has_ds = bool(rows) and rows[0].get('ds') not in (None, '')
    ds_text = f' (ds {float(level):.4g})' if has_ds else ''
    columns = [('main windows', 'pool')] + [
        (h, k) for h, k in extras
        if any(r.get(k) not in (None, '') for r in rows)]
    body, total = [], 0
    for (slide,), subset in sorted(by_slide.items()):
        queries = len({r['fov_id'] for r in subset})
        total += queries
        counts = []
        for _, key in columns:
            value = next((r[key] for r in subset if r.get(key) not in (None, '')),
                         None)
            counts.append('-' if value is None else f'{int(value):,}')
        body.append([slide, queries] + counts)
    title = f'ds {float(level):.4g}' if has_ds else f'Level {level}'
    emit(f'\n{title} -- {len(by_slide)} '
         f'slide{"" if len(by_slide) == 1 else "s"}, {total} queries per arm')
    grid_table(['slide', 'queries'] + [h for h, _ in columns], body, emit=emit)


def print_paired(rows: list, title: Optional[str], arms: Sequence[str],
                 baseline: str, emit: Callable[[str], None] = print) -> None:
    """The block that is identical at every aggregation level.

    `emit` is print by default and a list's append when the caller is building
    a report file. Both benches print the same table; only one of them also
    keeps it. A `title` of None prints none, for a caller that has already
    put a heading of its own above it.

    wins / losses / ties are the paired comparison against the baseline on the
    same query; `decided` is wins + losses.
    """
    if title is not None:
        emit(f'\n{title}   n={len(rows) // max(1, len(arms))} per arm')
    else:
        emit('')
    headers = (['arm', 'wins', 'losses', 'ties', 'decided', 'win%', 'Q1', 'med',
                'Q3'] + [frac_label(f) for f in K_FRACTIONS])
    body = []
    by_arm = group_by(rows, ['arm'])
    for arm in arms:
        subset = by_arm.get((arm,))
        if not subset:
            continue
        s = paired_stats(subset)
        # Same f-string on both sides of the dict, so a column cannot be read
        # from a key paired_stats never wrote.
        tops = [pct(s[f'top{f}']).strip() for f in K_FRACTIONS]
        if arm == baseline:
            body.append([arm + ' (base)'] + ['-'] * 8 + tops)
        else:
            body.append([arm, s['W'], s['L'], s['T'], s['n_cmp'],
                         pct(s['win_pct']).strip(), f'{s["ratio_q1"]:.2f}',
                         f'{s["ratio_med"]:.2f}', f'{s["ratio_q3"]:.2f}'] + tops)
    grid_table(headers, body, emit=emit)


def print_fixed_k(rows: list, arms: Sequence[str], baseline: str,
                  emit: Callable[[str], None] = print) -> None:
    """Fixed candidate budgets. Only where pool sizes are comparable."""
    emit('\nfixed k -- truth@k')
    body = []
    by_arm = group_by(rows, ['arm'])
    for arm in arms:
        subset = by_arm.get((arm,))
        if not subset:
            continue
        s = absolute_stats(subset)
        body.append([arm] + [pct(s[f'truth@{k}']).strip() for k in K_FIXED])
    base = by_arm.get((baseline,))
    if base:
        s = absolute_stats(base)
        body.append(['gap@k (base)'] + [pct(s[f'gap@{k}']).strip() for k in K_FIXED])
    grid_table(['arm'] + [f'k={k}' for k in K_FIXED], body, emit=emit)


def print_pool_header(slide: str, level: int, pool: int, base_rows: list,
                      emit: Callable[[str], None] = print) -> None:
    """The 'pool N, k: ..., baseline med/p90' line above one group's tables.

    The k values are printed as labelled pairs rather than as two parallel
    lists: k@0.01% floors to 1 on a small pool, and the reader has to be able to
    see WHICH fraction floored without counting positions across two
    slash-separated runs.
    """
    ks = '  '.join(f'{frac_label(f)}={k_at(f, pool)}' for f in K_FRACTIONS)
    s = absolute_stats(base_rows)
    emit(f'\n{slide}  L{level}   pool {pool:,}   k: {ks}   '
         f'baseline med {s["med_rank"]:,.0f}  p90 {s["p90_rank"]:,.0f}')


def report(rows: list, arms: Sequence[str], baseline: str, *,
           per_slide: bool = False, level_extras: Sequence[tuple] = (),
           emit: Callable[[str], None] = print) -> None:
    """單片單層 -> 同層跨片 -> 單片跨層 -> 全部, in that order.

    Absolute numbers first and narrowest, conclusions last and widest, because
    a reader who stops early should stop on the numbers that are valid in the
    smallest scope rather than on an aggregate whose caveats they have not read
    yet.
    """
    emit(f'\n{"=" * 90}\n單片單層 -- absolute numbers, one pool each\n{"=" * 90}')
    for (slide, level), subset in sorted(group_by(rows, ['slide', 'level']).items()):
        pool = subset[0]['pool']
        base = [r for r in subset if r['arm'] == baseline]
        print_pool_header(slide, level, pool, base, emit=emit)
        print_paired(subset, f'  {slide} L{level}', arms, baseline, emit=emit)
        print_fixed_k(subset, arms, baseline, emit=emit)

    emit(f'\n{"=" * 90}\n同層跨片 -- PRIMARY\n{"=" * 90}')
    for level, subset in group_levels(rows).items():
        print_level_heading(subset, level, emit=emit, extras=level_extras)
        print_paired(subset, None, arms, baseline, emit=emit)
        print_fixed_k(subset, arms, baseline, emit=emit)

    if per_slide:
        emit(f'\n{"=" * 90}\n單片跨層\n{"=" * 90}')
        for (slide,), subset in sorted(group_by(rows, ['slide']).items()):
            print_paired(subset, slide, arms, baseline, emit=emit)

    emit(f'\n{"=" * 90}\n全部 -- CONCLUSION\n{"=" * 90}')
    print_paired(rows, 'all slides, all levels', arms, baseline, emit=emit)
