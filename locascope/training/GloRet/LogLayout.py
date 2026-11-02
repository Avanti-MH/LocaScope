'''The training log laid out for a reader: a rule for each part of an epoch (TRAINING, VALIDATION, REAL PHOTOS, SELECTION), tables
for what is a table (the terms of the losses, the seconds of a step, the recalls), a line for what is a line. Functions
that return lines; `cli/train.py` prints them.
'''
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

WIDTH: int = 100
#: the terms of a head's loss, in the order of the table; the ones after `list` have a weight (a flag of the job or of the head)
LOSS_TERMS: Tuple[str, ...] = ('list', 'geo', 'corr', 'cycle', 'recon', 'kl')
WEIGHTED: Tuple[str, ...] = ('geo', 'corr', 'cycle', 'recon', 'kl')
#: the seconds of a step, by the key of the step's timers, and the name of its column
STEP_PARTS: Tuple[Tuple[str, str], ...] = (('t_pool', 'pool'), ('t_scores', 'scores'), ('t_keys', 'keys'), ('t_loss', 'loss'), ('t_back', 'back'))
#: the highest GPU memory (GB) of each phase, by the key of the step's log: what is held before the step, choosing the pools,
#: the scores (LogLoRet) or the window vectors (GloRet), the way back
MEM_PARTS: Tuple[Tuple[str, str], ...] = (('m_held', 'held before'), ('m_pool', 'pool'), ('m_scores', 'scores'), ('m_keys', 'keys'),
                                          ('m_back', 'back'))


def rule(title: str, right: str = '') -> str:
    """`━━ TITLE ━━━━━━━━━━━━━━━━━━━━━━━ right ━━`, WIDTH wide."""
    left: str = f'━━ {title} '
    tail: str = f' {right} ━━' if right else ''
    return left + '━' * max(WIDTH - len(left) - len(tail), 3) + tail


def num(value: Optional[float], digits: int = 3) -> str:
    """The number, or `-` for none."""
    if value is None or value != value:
        return '-'
    return f'{value:.{digits}f}'


def labels(names: Sequence[str]) -> Dict[str, str]:
    """A short name for each head: its family (`GloRet`, `LogLoRet`) when that tells the heads apart, else the whole name."""
    family: List[str] = [n.split('-')[0] for n in names]
    return {n: (f if family.count(f) == 1 else n) for n, f in zip(names, family)}


def table(header: Sequence[str], rows: Sequence[Sequence[str]], indent: int = 4) -> List[str]:
    """Columns aligned: the first to the left, the others to the right, a line under the header."""
    widths: List[int] = [max([len(header[i])] + [len(r[i]) for r in rows]) for i in range(len(header))]

    def line(cells: Sequence[str]) -> str:
        return ' ' * indent + '  '.join([cells[0].ljust(widths[0])] + [c.rjust(w) for c, w in zip(cells[1:], widths[1:])])

    return [line(header), ' ' * indent + '  '.join('─' * w for w in widths)] + [line(r) for r in rows]


def losses_block(names: Sequence[str], losses: Dict[str, float], weights: Dict[str, Dict[str, float]]) -> List[str]:
    """The terms of every head's loss, epoch means as they are BEFORE their weights, in one table, and the weights under it
    (once when the heads have the same, else a line a head). `loss` is what the step minimises (not the sum of the terms)."""
    lab: Dict[str, str] = labels(names)
    terms: List[str] = [t for t in LOSS_TERMS if any(f'{n}/{t}' in losses for n in names)]
    rows: List[List[str]] = [[lab[n]] + [num(losses.get(f'{n}/{t}'), 4) for t in terms] + [num(losses.get(f'{n}/loss'), 4)]
                             for n in names]
    out: List[str] = ['  losses   (epoch mean, each term as it is before its weight)'] + table(['head'] + terms + ['loss'], rows)
    used: List[str] = [t for t in terms if t in WEIGHTED]

    def text(w: Dict[str, float]) -> str:
        return ' · '.join(f'{t} ×{w[t]:g}' for t in used)

    if used:
        if all(weights[n] == weights[names[0]] for n in names):
            out.append(f'    weights  {text(weights[names[0]])}')
        else:
            out += [f'    weights  {lab[n]}: {text(weights[n])}' for n in names]
    return out


def steps_block(names: Sequence[str], losses: Dict[str, float]) -> List[str]:
    """The seconds of a step, by part, a row a head (the mean over the visits)."""
    lab: Dict[str, str] = labels(names)
    parts: List[Tuple[str, str]] = [(k, c) for k, c in STEP_PARTS if any(f'{n}/{k}' in losses for n in names)]
    if not parts:
        return []
    rows: List[List[str]] = [[lab[n]] + [num(losses.get(f'{n}/{k}'), 1) for k, _ in parts] for n in names]
    out: List[str] = ['  seconds   (mean over the visits; pool is per visit in all, the others per step)'] + table(
        ['head'] + [c for _, c in parts], rows)
    mem: List[Tuple[str, str]] = [(k, c) for k, c in MEM_PARTS if any(f'{n}/{k}' in losses for n in names)]
    if mem:
        out += ['  highest GPU memory in GB by phase   (of the whole card, so what other threads of the process hold counts too)'] + table(
            ['head'] + [c for _, c in mem], [[lab[n]] + [num(losses.get(f'{n}/{k}'), 1) for k, _ in mem] for n in names])
    return out


def hard_line(got: Dict[str, float]) -> str:
    """One head's hard negatives in a visit, from the keys of `hn/...` without the prefix (see `Episodes.hard_negative_stats`)."""
    g = lambda key: got.get(key, float('nan'))                                       # noqa: E731
    text: str = (f'hard negatives  pool {g("pool"):.0f} of {g("items"):.0f} items (K {g("k"):.0f}) · beat the answer '
                 f'{g("ahead"):.0f} (N over K median {g("ratio_med"):.2f}, p90 {g("ratio_p90"):.2f}) · padded {100 * g("padded"):.0f}%')
    if 'kept' in got:
        text += f' · pool kept from the last episode {100 * got["kept"]:.0f}%'
    return text + f' · a FoV trained {g("uses"):.1f}×'


def memory_line(got: Dict[str, float]) -> str:
    """One head's highest GPU memory (GB) by phase in a visit, from the keys of `m_...` without the prefix of the head."""
    return 'GPU peak GB by phase  ' + ' · '.join(f'{name} {got[key]:.1f}' for key, name in MEM_PARTS if key in got)


def hard_negatives_block(names: Sequence[str], losses: Dict[str, float]) -> List[str]:
    """The hard negatives of each head over the epoch (means over the visits): K the negatives asked for, N' how many beat the
    answer (the answer's rank is N'+1). N'/K near 0: most of the pool is padding, top-p can go down; N'/K at or over 1: the
    pool is all hard and the hardest are cut, top-p can go up. `kept` low: the episodes' pools differ; high: more episodes only repeat."""
    lab: Dict[str, str] = labels(names)
    columns: List[Tuple[str, str, int]] = [('items', 'items', 0), ('k', 'K', 0), ('ahead', "N'", 1), ('ratio_med', "N'/K med", 2),
                                           ('ratio_p90', "N'/K p90", 2), ('first', 'ranked 1st', 2), ('full', "N'>=K", 2),
                                           ('padded', 'padded', 2), ('pool', 'pool', 0), ('kept', 'kept', 2), ('uses', 'FoV uses', 1)]
    shown: List[Tuple[str, str, int]] = [c for c in columns if any(f'{n}/hn/{c[0]}' in losses for n in names)]
    if not shown:
        return []
    rows: List[List[str]] = [[lab[n]] + [num(losses.get(f'{n}/hn/{key}'), digits) for key, _, digits in shown] for n in names
                             if any(f'{n}/hn/{key}' in losses for key, _, _ in shown)]
    return ['  hard negatives   (mean over the visits)'] + table(['head'] + [c for _, c, _ in shown], rows) + [
        "    K = negatives asked for · N' = negatives that beat the answer · padded = share of K not beating it · kept = share of a "
        'pool also in the last pool of the FoV']


def recall_block(title: str, figures: Dict[str, Dict[str, float]], ks: Sequence[int], selected: Dict[str, int]) -> List[str]:
    """The share of FoVs whose first hit is in the top k, a row a head (the first phase's own rows first); `*` marks the k
    a head's weights are selected on."""
    names: List[str] = list(figures)
    lab: Dict[str, str] = labels(names)
    rows: List[List[str]] = [[lab[n]] + [num(figures[n][f'R@{k}']) + ('*' if selected.get(n) == k else '') for k in ks]
                             + [num(figures[n]['ceiling']), f'{figures[n]["n"]:.0f}'] for n in names]
    return [f'  {title}'] + table(['', *[f'R@{k}' for k in ks], 'ceiling', 'n'], rows) + ['    * = what the weights are selected on']


def timing_lines(wall: float, timers: Dict[str, float], lab: Dict[str, str]) -> List[str]:
    """Where `wall` seconds of this process went: what the training did and waited for, what making the visits took (the thread
    that makes them runs beside the training), and the seconds of each head's ranking."""
    t = lambda name: timers.get(name, 0.0)                                           # noqa: E731
    tiles: float = t('level_tiles')
    out: List[str] = []
    train: List[str] = []
    if t('train_steps') or t('wait_for_a_visit') or t('wait_for_the_others'):
        train = [f'steps {t("train_steps"):.0f} s', f'waited for a visit {t("wait_for_a_visit"):.0f} s',
                 f'for the other ranks {t("wait_for_the_others"):.0f} s']
    ranking: List[str] = [f'{lab.get(k[5:], k[5:])} {v:.0f} s' for k, v in timers.items() if k.startswith('rank ')]
    if train:
        out.append('    this process   ' + ' · '.join(train))
    if ranking:
        out.append('    ranking        ' + ' · '.join(ranking))
    made: List[str] = [f'encode {t("level_encode"):.0f} s ({tiles / 1e6:.2f}M tiles, {tiles / max(t("level_encode"), 1e-9):.0f}/s)'
                       if tiles else 'no encoding']
    made += [f'search {max(t("retrieve") - t("level_encode"), 0.0):.0f} s', f'open+mask {t("open_and_mask"):.0f} s',
             f'FoV draw {t("draw_positions"):.0f} s', f'first photo waits {t("photo_wait"):.0f} s']
    out.append('    visits made    ' + ' · '.join(made) + '   (beside the training)')
    return out


__all__ = ['LOSS_TERMS', 'MEM_PARTS', 'WIDTH', 'hard_line', 'memory_line', 'hard_negatives_block', 'labels', 'losses_block', 'num', 'recall_block', 'rule', 'steps_block', 'table', 'timing_lines']
