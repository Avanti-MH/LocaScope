#!/usr/bin/env python3
"""Read the stage 2 entries bench_stage2_retriever.py wrote, and say which way of finding the window wins.

Takes bench_stage2_retriever's own flags and computes the same addresses: for every slide its FoV draw and
render table, for every arm its stage 2 entries. No model and no GPU; the bench only runs and writes those
entries, and everything about them -- the rows, the CSV, the tables -- is here. Every arm has two modes, read
off the same entries:

  ranked   every window of the level ranked by the arm's score, no suppression (NMS off): the search itself
  listed   what the stage hands on: the best `--k` windows with suppression (NMS on); a shorter K is a prefix

Rows are read off the entries by the one definition of an answer and its ranks (`BenchCommon.answers_of_row`,
`listed_places`) and written to `<out>/stage2_retriever.csv`; every number in the tables is derived from the
stored integers (`RetrievalReport`, shared with any bench that asks the same question), so changing the k list
or the grouping never costs a GPU hour. An arm's name carries its encoder, so the CSVs of several runs (one
per encoder) can be reported together.

Usage:
    python -m locascope.cli.metrics.analyze_stage2_metrics <bench_stage2_retriever flags> [--out DIR]
    python -m locascope.cli.metrics.analyze_stage2_metrics --csv \\
        $LOCASCOPE_OUTPUT_ROOT/result/Stage2RetrieverBench/uni2/stage2_retriever.csv \\
        $LOCASCOPE_OUTPUT_ROOT/result/Stage2RetrieverBench/gigapath/stage2_retriever.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np
import torch

from locascope.cli.bench.BenchCommon import (ORACLE, Answers, Pools, answers_of_row, listed_of_rows,  # noqa: E402
                                             listed_places, run_slides, supply_for)
from locascope.cli.bench.RetrieverArms import (FAMILY_METHOD, FAMILY_POOLING, FAMILY_RERANK,  # noqa: E402
                                               MODE_LISTED, MODE_RANKED, ArmEngine)
from locascope.cli.bench.bench_stage2_retriever import (K_CURVE, RECALL_COLUMNS, TILE, Stage2Arm,  # noqa: E402
                                                        add_args, baseline_of, canonical_arm, parse_args,
                                                        resolve)
from locascope.cli.metrics.RetrievalReport import (K_FIXED, attach_baseline, grid_table,  # noqa: E402
                                                   group_by, group_levels, pct, print_level_heading,
                                                   print_paired, report)
from locascope.core.stage2_retrieval.ArmSpec import SEP, parse_arm                      # noqa: E402
from locascope.paths import job_result_dir                                              # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  The rows: read off the stage 2 entries
#
#  An arm's rows are not kept anywhere but the cache: every one is read off its entries, by the one
#  definition of an answer and its ranks (BenchCommon). A ranked row is its first phase's truth (the ranks
#  it holds); a listed row is the answers' places in its last phase's list (`listed_places`), judged on its
#  first phase's answers. The FoV's own columns come from the draw and the render table.
# ══════════════════════════════════════════════════════════════════════════════

class ArmRows(NamedTuple):
    """How one arm's rows are read off the cache: its text, the entry of its first phase (whose truth judges
    it) and of its last (whose list it is), its family, the `source` its rows say, its lattices (`grid`), the
    length its list is cut at (None: all it holds), and whether it has a ranked mode (a first phase that
    scored every window)."""
    text: str
    first: str
    last: str
    family: str
    source: str
    grid: str
    cut: Optional[int]
    ranked: bool


def arm_rows_of(streamed_arms: Sequence[Stage2Arm], streamed_rows: dict, engine: Optional[ArmEngine]
                ) -> list[ArmRows]:
    """Every arm of the run that has rows: the streamed ones (`streamed_rows`, arm text -> entry) and the
    engine's."""
    by_entry: dict = {a.entry_id: a for a in streamed_arms}
    out: list[ArmRows] = [ArmRows(text, entry_id, entry_id, FAMILY_POOLING, 'stream', by_entry[entry_id].grid,
                                  None, True)
                          for text, entry_id in streamed_rows.items()]
    if engine is not None:
        for text in engine.arm_names():
            spec = parse_arm(text)
            reranker: bool = len(spec.segments) > 1
            out.append(ArmRows(text, engine.first_entry_of(text), engine.entry_of(text),
                               FAMILY_RERANK if reranker else FAMILY_METHOD, 'retriever', 'all',
                               spec.segments[-1].cut or None,
                               not reranker and spec.retriever.method != 'SlidingWindow'))
    return out


def _entry_table(entry: Any, entry_id: str, role: str) -> dict:
    """`{FoV index: rows}` of one table of an entry; empty when the entry has no such table."""
    path = entry.path(role, entry_id, '.csv')
    out: dict = {}
    if not path.is_file():
        return out
    with open(path, newline='') as handle:
        for r in csv.DictReader(handle):
            out.setdefault(int(r['index']), []).append(r)
    return out


def slide_rows(entry: Any, arms: Sequence[ArmRows], *, encoder: str, dataset: str, stem: str, supply: Any,
               min_sep: float) -> tuple[list, int]:
    """`(rows, FoVs two entries place differently)` of one slide: for every arm and every FoV its first phase's
    truth ranks, a ranked row (when the arm has that mode) and a listed row. The answers come from that truth
    (`answers_of_row`); every first phase places them by the same geometry, and a FoV they disagree on is
    counted, not silently compared."""
    path = supply.render_entry.path('render', supply.microscope.cfg.identity_id(), '.csv')
    with open(path, newline='') as handle:
        render = {int(r['index']): r for r in csv.DictReader(handle)}
    metas = {i: s.meta for i, s in enumerate(supply.sampler)}
    tables: dict = {}

    def table(entry_id: str, role: str) -> dict:
        if (entry_id, role) not in tables:
            tables[(entry_id, role)] = _entry_table(entry, entry_id, role)
        return tables[(entry_id, role)]

    rows: list = []
    seen: dict = {}                                   # FoV -> its main answer's window, as the first entry put it
    differ: set = set()
    for arm in arms:
        truth: dict = table(arm.first, 'truth')
        lists: dict = table(arm.last, 'output')
        for fov, (t, *_) in sorted(truth.items()):
            found: Optional[Answers] = answers_of_row(t)
            if found is None or t.get('pool_main', '') == '' or fov not in lists:
                continue                              # not searched at its own level: nothing to rank
            if seen.setdefault(fov, (found.main.key, found.main.box)) != (found.main.key, found.main.box):
                differ.add(fov)
            pools = Pools(main=int(float(t['pool_main'])),
                          offset=int(float(t['pool_all'])) - int(float(t['pool_main'])))
            listed_rows = sorted((r for r in lists[fov] if r.get('rank', '') != ''), key=lambda r: int(r['rank']))
            first_row = lists[fov][0]
            ds = float(first_row['ds'])
            full: bool = arm.grid == 'all'
            judged: Answers = found if full else found.main_only()
            score = t.get('truth_main_score', '')
            columns = dict(
                # First column on purpose: `--csv` refuses a set that mixes encoders on this key.
                encoder=encoder, dataset=dataset, slide=stem, level=int(float(t['truth_level'])), ds=ds,
                fov_id=fov, pool=pools.main, d_main=round(found.main.dist_l0 / ds, 2),
                d_overlap=round(found.offset.dist_l0 / ds, 2) if found.offset is not None else '',
                white_frac=round(float(metas[fov].score), 4), bucket=metas[fov].bucket,
                rot_deg=float(render[fov]['rot_deg']), scale=round(float(render[fov]['scale']), 4),
                source=arm.source, grid=arm.grid, family=arm.family, arm=arm.text,
                truth_rot=found.main.rotation, truth_score=round(float(score), 6) if score != '' else '',
                seconds=float(first_row.get('t_s') or 0.0))
            blank = {k: '' for k in RECALL_COLUMNS}

            def row_of(mode: str, ranks: dict, list_len: int) -> dict:
                recall = ({**{k: ranks[k] for k in RECALL_COLUMNS if k in ranks},
                           'pool_offset': pools.offset, 'pool_all': pools.all} if full else blank)
                # rank_overlap repeats rank_main ON PURPOSE: the shared `truth_rank` reads it when the offset
                # point is the nearer one, and THE answer is the main one
                return dict(columns, mode=mode, list_len=list_len, rank_main=ranks['rank_main'],
                            rank_overlap=ranks['rank_main'], **recall)

            if arm.ranked:
                stored = {k: int(float(t[f'truth_{k}'])) for k in ('rank_main', 'rank_main_in_all',
                                                                   'rank_offset_in_offset', 'rank_offset_in_all')
                          if t.get(f'truth_{k}', '') != ''}
                rows.append(row_of(MODE_RANKED, stored, pools.all))
            cut_rows = listed_rows if arm.cut is None else listed_rows[:arm.cut]
            listed = listed_of_rows(cut_rows)
            ranks = listed_places(listed, judged, pools, float(min_sep) * TILE * ds)
            rows.append(row_of(MODE_LISTED, ranks, len(listed)))
    return rows, len(differ)


# ══════════════════════════════════════════════════════════════════════════════
#  The rows in a CSV and back
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(rows: list, path: Path) -> None:
    if not rows:
        print(f'  (nothing to write to {path.name})')
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, restval='')
        writer.writeheader()
        writer.writerows(rows)
    print(f'  {path}  ({len(rows):,} rows)')


def read_rows(paths, quiet: bool = False) -> list:
    integers = {'level', 'fov_id', 'pool', 'rank_main', 'rank_overlap', 'truth_rot',
                'list_len', *RECALL_COLUMNS}
    floats = {'d_main', 'd_overlap', 'white_frac', 'ds', 'rot_deg', 'scale',
              'truth_score', 'seconds'}
    rows = []
    for path in paths:
        with open(path, newline='') as handle:
            for raw in csv.DictReader(handle):
                row = {}
                for key, value in raw.items():
                    if key in integers or key in floats:
                        row[key] = (value if value == '' else
                                    int(float(value)) if key in integers
                                    else float(value))
                    else:
                        row[key] = value
                rows.append(row)
        if not quiet:
            print(f'  read {path}')
    return rows



# ══════════════════════════════════════════════════════════════════════════════
#  Derived metrics -- everything reads only the stored integers
#
#  The statistics themselves are in RetrievalReport.py, shared with any bench that asks the same question
#  one scale down: two copies of "@1%" would never have raised, both would print a plausible number under the
#  same header, and a comparison between the two benches would be quietly invalid.
# ══════════════════════════════════════════════════════════════════════════════

#: K_FIXED and K_FRACTIONS are imported from RetrievalReport, which owns them
#: along with every statistic computed from them. What the pools look like HERE,
#: which is what decides whether @0.01% means anything on a given row:
#:
#:     L0   pool 18,296..183,833   k = 2..19    a real 0.01% criterion
#:     L1   pool  3,432.. 10,495   k = 1..2     partly floored
#:     L2   pool    311..    499   k = 1        truth@1, null 0.20..0.32%
#:
#: Read @0.01% in 單片單層 and in the L0 row of 同層跨片; in 全部 it averages a
#: genuine 0.01% at L0 with truth@1 at L2, which is not one criterion.


def random_recall(pool_main: int, pool_offset: int, pool_all: int, k: int) -> tuple:
    """Expected (main, offset, all) recall@k of a RANDOM ranking on these pools.

    One target in a pool of P is in the top k with probability min(1, k/P). Two
    targets, both in the mixed pool, are both outside it with probability
    (P-k)(P-k-1) / (P(P-1)), so `all` starts near twice `main` before any
    scoring happens -- the number the rise from main to all is read against."""
    one = lambda pool: min(1.0, k / max(1, pool))                     # noqa: E731
    miss = (max(0, pool_all - k) * max(0, pool_all - k - 1)
            / max(1, pool_all * (pool_all - 1)))
    return one(pool_main), one(pool_offset), 1.0 - miss


#: Candidate counts a full-grid arm's rows carry beyond `pool` (the main windows):
#: what the level tables show as extra columns.
LEVEL_EXTRAS = (('offset windows', 'pool_offset'),
                ('main+offset windows', 'pool_all'))


def recall_block(rows: list, arms, emit=print) -> None:
    """recall_main / recall_offset / recall_all @k, per full-grid arm, one level.

    `rows` are the full-grid rows of one level. Pools differ between slides, so
    a fixed k is the axis (it is also what stage 3 pays for), not a fraction.
    One grid, a group of rows per arm: the three recalls, then the share of the
    hits that the main window answered first. `random` is the recall of a random
    ranking on the same pools."""
    by_arm = group_by(rows, ['arm'])
    first = next((by_arm[(a,)] for a in arms if (a,) in by_arm), None)
    if first is None:
        return
    body, rules = [], []
    rand = [[random_recall(r['pool'], r['pool_offset'], r['pool_all'], k)
             for r in first] for k in K_FIXED]
    for j, name in enumerate(('main', 'offset', 'all')):
        body.append(['random' if j == 0 else '', f'recall_{name}']
                    + [pct(float(np.mean([v[j] for v in rand[i]]))).strip()
                       for i in range(len(K_FIXED))])
    rules.append(len(body) - 1)
    ties = 0
    for arm in arms:
        subset = by_arm.get((arm,))
        if not subset:
            continue
        rm = np.array([r['rank_main'] for r in subset])
        ro = np.array([r['rank_offset_in_offset'] for r in subset])
        am = np.array([r['rank_main_in_all'] for r in subset])
        ao = np.array([r['rank_offset_in_all'] for r in subset])
        best = np.minimum(am, ao)
        ties += int((am == ao).sum())
        for j, (label, ranks) in enumerate((('recall_main', rm),
                                            ('recall_offset', ro),
                                            ('recall_all', best))):
            body.append([arm if j == 0 else '', label]
                        + [pct(float((ranks <= k).mean())).strip() for k in K_FIXED])
        shares = []
        for k in K_FIXED:
            hit = best <= k
            if not hit.any():
                shares.append('-')
                continue
            from_main = float((am[hit] <= ao[hit]).mean()) * 100
            shares.append(f'{from_main:.0f}:{100 - from_main:.0f}')
        body.append(['', 'all: main:offset'] + shares)
        rules.append(len(body) - 1)
    grid_table(['arm', 'metric'] + [f'k={k}' for k in K_FIXED], body,
               aligns='ll' + 'r' * len(K_FIXED), rule_after=rules, emit=emit)
    n = sum(len(v) for v in by_arm.values())
    emit(f'main and offset answers tied in the mixed pool on {ties}/{n} '
         f'(arm, FoV) rows; a tie counts for main')


def recall_report(rows: list, arms, emit=print) -> None:
    """What the offset grid adds: recall of the two answers, per ds.

    Full-grid arms only: a main-only arm has no offset windows to rank. Per level
    and not over all levels, because k means something different in a pool of a
    few hundred windows and in one of a hundred thousand."""
    full = [r for r in rows if r.get('grid') == 'all'
            and r.get('rank_offset_in_offset') not in ('', None)]
    if not full:
        emit('\n(no full-grid arm ran: no main vs offset vs all recall table)')
        return
    emit(f'\n{"=" * 90}\nrecall -- main vs offset vs all (main + offset), '
         f'full-grid arms\n{"=" * 90}')
    for level, subset in group_levels(full).items():
        print_level_heading(subset, level, emit=emit, extras=LEVEL_EXTRAS)
        emit('')
        recall_block(subset, arms, emit=emit)


#: The modes' names in an older CSV.
OLD_MODES: dict[str, str] = {'whole': MODE_RANKED, 'production': MODE_LISTED}


def mode_of(row: dict) -> str:
    """An older CSV has no `mode` and is all ranked, or names the modes `whole` and `production`."""
    mode: str = row.get('mode') or MODE_RANKED
    return OLD_MODES.get(mode, mode)


def family_of(row: dict) -> str:
    """An older CSV has no `family` and is all streamed sliding-window arms."""
    return row.get('family') or FAMILY_POOLING


def normalise(rows: list) -> list:
    """The rows with every arm in the bench's grammar: an older CSV's `cls_avg+geomean` is
    `<encoder>:SlidingWindow::cls_avg+geomean`, named by the encoder its rows say."""
    for r in rows:
        r['arm'] = canonical_arm(r['arm'], r.get('encoder') or 'uni2')
        r['mode'] = mode_of(r)
    return rows


def report_arms(rows: list) -> list[str]:
    """The arms a report shows: the streamed sliding windows, then the other retrievers, then the
    rerankers, each in the order they first appear."""
    arms: list[str] = []
    for family in (FAMILY_POOLING, FAMILY_METHOD, FAMILY_RERANK):
        arms += dict.fromkeys(r['arm'] for r in rows if family_of(r) == family)
    return list(dict.fromkeys(arms))


def pick_baseline(rows: list, requested: str = '') -> str:
    """The arm every other is paired with: the one asked for, else `uni2:SlidingWindow::cls+mean`, else
    the first encoder's. The FoVs are the same whatever the encoder, so any encoder's arm can be the one."""
    names = {r['arm'] for r in rows if mode_of(r) == MODE_RANKED}
    if requested:
        if requested not in names:
            raise SystemExit(f'--baseline {requested!r} is not an arm of these rows')
        return requested
    for encoder in ['uni2'] + [r.get('encoder') or 'uni2' for r in rows]:
        if baseline_of(encoder) in names:
            return baseline_of(encoder)
    raise SystemExit('no `cls+mean` sliding window in these rows to pair the others with')


def check_pools(rows: list, baseline: str, emit=print) -> int:
    """Every arm of a FoV is ranked in the same pools. The retriever-path arms count theirs off the
    retriever's maps and the streamed arms off the streamed windows; they must be equal, and a FoV where
    they are not is a place the two do not agree on. Returns how many."""
    base = {(r['slide'], r['level'], r['fov_id']): (r['pool'], r.get('pool_offset'), r.get('pool_all'))
            for r in rows if r['arm'] == baseline and mode_of(r) == MODE_RANKED}
    bad = 0
    total = 0
    for r in rows:
        if family_of(r) == FAMILY_POOLING:
            continue
        want = base.get((r['slide'], r['level'], r['fov_id']))
        if want is None:
            continue
        total += 1
        bad += int((r['pool'], r.get('pool_offset'), r.get('pool_all')) != want)
    if total:
        emit(f'pools: {total - bad}/{total} retriever-path rows are ranked in the same pools as the '
             f'baseline' + ('' if not bad else f'  -- {bad} are NOT: those FoVs are not comparable'))
    return bad


def listed_report(rows: list, arms, emit=print) -> None:
    """What a listed arm's list could not rank. An answer the list does not hold has rank pool + 1, a miss
    at every k and f%, so each arm's ceiling is the share of FoVs whose answer is listed at all -- here,
    in the bench's own terms -- and the lists' length."""
    by_arm = group_by([r for r in rows if r.get('list_len') not in ('', None)], ['arm'])
    body = []
    for arm in arms:
        subset = by_arm.get((arm,))
        if not subset:
            continue
        main = np.array([r['rank_main'] <= r['pool'] for r in subset])      # the answer nearer the shot
        if all(r.get('pool_all') not in ('', None) for r in subset):
            either = np.array([min(r['rank_main_in_all'], r['rank_offset_in_all']) <= r['pool_all']
                               for r in subset])
            either_text = pct(float(either.mean())).strip()
        else:
            either_text = '-'
        lengths = np.array([r['list_len'] for r in subset], dtype=float)
        body.append([arm, len(subset), f'{np.median(lengths):.0f}', pct(float(main.mean())).strip(),
                     either_text])
    if not body:
        return
    emit(f'\n{"=" * 90}\nlisted -- how much of the answer a list holds at all\n{"=" * 90}')
    grid_table(['arm', 'FoVs', 'list length (med)', 'nearer answer listed', 'either answer listed'],
               body, aligns='lrrrr', emit=emit)
    emit('(listed = the window itself, or one of its rotation within the suppression radius -- the one that '
         'kept its place; an answer with none is rank pool + 1: a miss at every k and every f%)')


def curve_report(rows: list, arms, emit=print) -> None:
    """How the recall of a listed arm's list grows with its length: `recall_all@K` for each K of `K_CURVE`,
    per level, for each arm. A list holds its best K as a prefix, so one list gives the whole curve; a K
    past a level's pool is the level, and an answer not listed is never a hit however large K is. The
    place to read which K covers the most of the answer without making the list needlessly long."""
    full = [r for r in rows if r.get('rank_offset_in_offset') not in ('', None)]
    if not full:
        return
    emit(f'\n{"=" * 90}\nrecall_all@K of a listed arm -- how long a list has to be\n{"=" * 90}')
    for level, subset in group_levels(full).items():
        print_level_heading(subset, level, emit=emit, extras=LEVEL_EXTRAS)
        by_arm = group_by(subset, ['arm'])
        body = []
        for arm in arms:
            rs = by_arm.get((arm,))
            if not rs:
                continue
            best = np.array([min(r['rank_main_in_all'], r['rank_offset_in_all']) for r in rs])
            pool = np.array([r['pool_all'] for r in rs])
            body.append([arm] + [pct(float((best <= np.minimum(k, pool)).mean())).strip() for k in K_CURVE]
                        + [f'{np.median([r["list_len"] for r in rs]):.0f}'])
        grid_table(['arm'] + [f'K={k}' for k in K_CURVE] + ['list (med)'], body,
                   aligns='l' + 'r' * (len(K_CURVE) + 1), emit=emit)


def rerank_report(rows: list, emit=print) -> None:
    """Each reranker against the list it was given: every arm `A::R` paired with `A`, query by query, in
    the bench's own terms (wins / losses / ties, win%, ratio, top@f%). Both hold the same windows, so the
    ceiling is shared and what differs is only where the answer stands in the list."""
    children: dict[str, list[str]] = {}
    for arm in report_arms(rows):
        tokens = arm.split(SEP)
        if len(tokens) > 2 and any(family_of(r) == FAMILY_RERANK for r in rows if r['arm'] == arm):
            children.setdefault(SEP.join(tokens[:-2]), []).append(arm)
    if not children:
        return
    emit(f'\n{"=" * 90}\nrerankers against the list they were given (each paired with its own retriever)\n{"=" * 90}')
    for parent, kids in children.items():
        subset = [r for r in rows if r['arm'] in [parent, *kids]]
        paired = attach_baseline(subset, parent)
        if not paired:
            continue
        for level, part in group_levels(paired).items():
            print_level_heading(part, level, emit=emit, extras=LEVEL_EXTRAS)
            print_paired(part, f'  {parent}  ->', [parent, *kids], parent, emit=emit)


MODE_HEADINGS = {
    MODE_RANKED: 'RANKED (NMS off) -- every window of the level ranked by the arm\'s score, no suppression: '
                 'the search itself, only the answer\'s own window counting',
    MODE_LISTED: 'LISTED (NMS on) -- the longest list an arm hands on, with suppression; a shorter K is its '
                 'prefix. An answer is its window or a window within the suppression radius (the one '
                 'that kept its place); with none it is rank pool + 1'}


def tabulate(rows: list, per_slide: bool, baseline: str = '', emit=print) -> None:
    """The whole report: one block per mode, each paired with the baseline; for the listed mode also the
    recall curve over list length and each reranker against its own retriever."""
    rows = normalise(rows)
    base: str = pick_baseline(rows, baseline)
    emit(f'baseline (every arm is paired with it): {base}')
    check_pools(rows, base, emit=emit)
    for mode in (MODE_RANKED, MODE_LISTED):
        subset = [r for r in rows if mode_of(r) == mode]
        if not subset:
            continue
        emit(f'\n{"#" * 90}\n# {MODE_HEADINGS[mode]}\n{"#" * 90}')
        arms = report_arms(subset)
        report(attach_baseline(subset, base), arms, base,
               per_slide=per_slide, level_extras=LEVEL_EXTRAS, emit=emit)
        if mode == MODE_RANKED:
            recall_report(subset, arms, emit=emit)      # main vs offset vs all: a list has no lattice of its own
        if mode == MODE_LISTED:
            curve_report(subset, arms, emit=emit)
            listed_report(subset, arms, emit=emit)
            rerank_report(subset, emit=emit)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='the stage 2 entries of a bench_stage2_retriever run -> rows, CSV and tables',
        allow_abbrev=False)
    parser.add_argument('--csv', nargs='+', default=None, metavar='CSV',
                        help='no entries: re-tabulate these finished CSVs (any number, of any encoders -- an '
                             'arm\'s name carries its encoder, an older CSV\'s is made into the bench\'s '
                             'grammar), so a table holds the arms of several runs side by side')
    parser.add_argument('--per-slide', action=argparse.BooleanOptionalAction, default=True,
                        help='print the 單片跨層 tables. Worth having only when the per-level tables show an '
                             'H&E / Ki67 split')
    parser.add_argument('--baseline', default='',
                        help='the arm every other is paired with in the report (default `uni2:SlidingWindow'
                             '::cls+mean`, else the first encoder\'s)')
    parser.add_argument('--out', default=None,
                        help='output directory, used verbatim. Default: result/Stage2RetrieverBench/<encoder>/')
    add_args(parser)
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument('--csv', nargs='+', default=None)
    known, _ = pre.parse_known_args()
    args = parse_args(parser, encoder_flags=known.csv is None)

    if args.csv:
        tabulate(read_rows(args.csv), args.per_slide, args.baseline)
        return 0

    plan = resolve(args, parser, torch.device('cpu'))
    arm_rows: list[ArmRows] = arm_rows_of(plan.streamed_arms, plan.streamed_rows, plan.engine)
    out_dir = Path(args.out or job_result_dir('Stage2RetrieverBench', encoder=plan.tag))
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = run_slides(args)
    print(f'slides ({len(selected)}), arms ({len(arm_rows)})')
    rows: list = []
    failed: list = []
    differ: int = 0
    for dataset, stem, path in selected:
        print(f'\n{stem}  ({dataset})', flush=True)
        try:
            supply = supply_for(str(path), plan.fov_run, plan.fov_masks, args, plan.job)
            if supply is None:
                continue
            entry = supply.render_address.on(plan.job).at(stage1=ORACLE).entry('stage2')
            got, apart = slide_rows(entry, arm_rows, encoder=plan.tag, dataset=dataset, stem=stem,
                                    supply=supply, min_sep=args.min_sep_tiles)
            rows.extend(got)
            differ += apart
            print(f'  {len(got)} rows read off the stage 2 entries'
                  + (f'   {apart} FoV whose answers two entries place differently: their rows compare '
                     f'other windows' if apart else ''), flush=True)
        except Exception as exc:                          # noqa: BLE001
            print(f'  {type(exc).__name__}: {exc}')
            failed.append(stem)

    print(f'\n{"=" * 78}')
    # The encoder is in the NAME as well as in every row: the name stops a second encoder's run from
    # overwriting the first one's numbers, the column stops the two from being averaged together afterwards.
    write_csv(rows, out_dir / 'stage2_retriever.csv')
    if rows:
        tabulate(rows, args.per_slide, args.baseline)
    if differ:
        print(f'\n{differ} FoV(s) whose answers two entries place differently: one geometry should place them, '
              f'and does not -- look at those entries before reading their rows')
    if failed:
        print(f'\n{len(failed)} slide(s) failed: {", ".join(failed)}')
    return 1 if failed or differ else 0


if __name__ == '__main__':
    sys.exit(main())
