'''The training log's layout: a fixed-width table, a row every `log_every` steps (the window's means), its header at the start and
at every epoch; the layer mix on a line of its own every `mix_every` steps; an epoch's summary.

    step  ep |   loss  recon   lp8  lp24    pc    tw  place |  wait  sync   sim  targ   fwd   bwd   opt |   n blank s/step tiles/s | make: cand  wall  pack   read   work    cpu pids |  mem
      20   0 |  3.908  1.077 1.010 1.020 1.060 1.100  2.831 |  0.01  0.00  0.30  0.21  1.80  3.90  0.05 | 128   0.0   4.50    910 |       0.01  8.80  0.10   16.0   40.0   39.0   10 |  71.2

  the losses   loss = w_recon recon + w_pool place (+ w_koleo koleo, w_adv (fit + lam conf): only when their weights are > 0, not in
               the table, in log.tsv)
               recon  the Co-figure reconstruction (channels on one scale); after it the same a signal: lp8, lp24 (local phase),
                      pc (congruency), tw (T weighted) -- point: scaled MSE, their channel-weighted mean is recon; vdist: NLL
               place  cls' linear prediction of the place descriptor (the raw first and second moments of the Co-figure over 4 x 4
                      cells), scaled MSE
  the seconds  wait   this process waiting for its next batch (> 0: the data is the bottleneck)
               sync   then waiting for the slowest process to have its batch (> 0: another process's data is slower)
               sim    the views on the GPU: the reference reads over the bus, the framing and the photometric simulation, the centre
                      tiles and the targets' grey canvases
               targ   the Co-figure targets from them; fwd, bwd (with the gradients' all-reduce), opt: the network's step
               n      the places rank 0 read a step (places / processes)
               blank  the places that came back blank or off the slide although the pool was screened at start (0 is the rule:
                      > 0 = the screen leaked)
               s/step the wall time a step; tiles/s rank 0's places x the processes x the views, a second
               make   rank 0's batch, made ahead by the worker processes (the reads, the framing's and the simulation's draws; the
                      pixels are sim): cand (the step's draw), wall (from the batch's take-off to its last place: two batches are
                      made at once, so walls overlap; the rate is s/step), pack (the arrays)
               read   the workers' seconds in the slide reads, summed (part of work)
               work   the workers' wall seconds on that batch, summed (/ s/step ~ the workers busy at once when the data is
                      the bottleneck)
               cpu    their CPU seconds, summed (/ work ~ 1: each worker has a core of its own)
               pids   how many different workers reported
               mem    the GPU's peak memory in the window (GB)
'''
from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

Col = Tuple[str, str, int, Optional[int]]              # (key, title, width, decimals; None = a bar)


def columns(parts: Sequence[str] = ()) -> List[Col]:
    """The table's columns: the reconstruction a signal (`parts`: CoFigure.part_names, keys r_<part>) after `recon`, then `pool`
    (titled place: cls''s prediction of the place descriptor)."""
    recon: List[Col] = [('recon', 'recon', 6, 3)] + [(f'r_{p}', p, 6, 3) for p in parts] + [('pool', 'place', 6, 3)]
    return _HEAD + recon + _TAIL


_HEAD: List[Col] = [('step', 'step', 6, 0), ('epoch', 'ep', 3, 0), ('|', '|', 1, None), ('loss', 'loss', 7, 3)]
_TAIL: List[Col] = [
    ('|', '|', 1, None),
    ('t_wait', 'wait', 5, 2), ('t_sync', 'sync', 5, 2), ('t_sim', 'sim', 5, 2), ('t_targ', 'targ', 5, 2), ('t_fwd', 'fwd', 5, 2), ('t_bwd', 'bwd', 5, 2), ('t_opt', 'opt', 5, 2), ('|', '|', 1, None),
    ('n', 'n', 4, 0), ('blank', 'blank', 5, 1), ('s_step', 's/step', 6, 2), ('tiles_s', 'tiles/s', 7, 0), ('|', '|', 1, None),
    ('m_cand', 'make: cand', 10, 2), ('m_wall', 'wall', 5, 2), ('m_pack', 'pack', 5, 2), ('m_read', 'read', 6, 1),
    ('m_work', 'work', 6, 1), ('m_cpu', 'cpu', 6, 1), ('m_pids', 'pids', 4, 0), ('|', '|', 1, None),
    ('mem', 'mem', 5, 1)]


def header(cols: List[Col]) -> str:
    return ' '.join(f'{t:>{w}}' for _, t, w, _ in cols)


def row(v: Dict[str, float], cols: List[Col]) -> str:
    out: List[str] = []
    for k, _, w, d in cols:
        if d is None:
            out.append('|')
        elif k not in v:
            out.append(f'{"-":>{w}}')
        else:
            out.append(f'{v[k]:>{w}.{d}f}' if d else f'{int(round(v[k])):>{w}d}')
    return ' '.join(out)


def mix_line(step: int, layers: Sequence[int], weights: Sequence[float]) -> str:
    return f'{step:>6} mix  ' + '  '.join(f'L{l} {w:.3f}' for l, w in zip(layers, weights))


class Window:
    """The means of what is added over a log window."""

    def __init__(self) -> None:
        self.sum: DefaultDict[str, float] = defaultdict(float)
        self.n: DefaultDict[str, int] = defaultdict(int)

    def add(self, **kv: float) -> None:
        for k, v in kv.items():
            self.sum[k] += float(v)
            self.n[k] += 1

    def means(self) -> Dict[str, float]:
        return {k: self.sum[k] / self.n[k] for k in self.sum}

    def clear(self) -> None:
        self.sum.clear()
        self.n.clear()


def grad_line(step: int, g: Dict[str, Tuple[float, float]]) -> str:
    """The norm of each weighted loss's gradient on the latent and on cls' (what each pulls with)."""
    return f'{step:>6} grad ' + '  '.join(f'{k}: latent {lat:.2e} cls\' {cls:.2e}' for k, (lat, cls) in g.items())


def epoch_line(epoch: int, seconds: float, means: Dict[str, float]) -> str:
    keys: List[str] = ['loss', 'recon'] + sorted(k for k in means if k.startswith('r_')) + ['pool', 'nce', 'vnce', 'hnce', 'dnce', 'lnce', 'koleo', 'fit', 'conf', 'acc_cells', 'acc_cls']
    keep: str = '  '.join(f'{k} {means[k]:.3f}' for k in keys if k in means)
    return f'==== epoch {epoch} done in {seconds / 60:.1f} min | means: {keep}'


def cofig_line(tag: str, res: Dict[str, Any]) -> str:
    """Validate.real_cofig: the predicted Co-figure against the target, a signal: cos / scaled MSE."""
    parts: str = '  '.join(f'{k} {v["cos"]:.3f}/{v["mse"]:.3f}' for k, v in res.items() if isinstance(v, dict))
    n: str = ', '.join(f'{res[k]} {k}' for k in ('tiles', 'photos') if k in res)
    return f'     cofig predicted vs computed, {tag} ({n}), cos/mse a signal: {parts}'


def probe_table(pm: Dict[str, Any]) -> str:
    """Validate.probe_measures as tables. Feature tables, for each of two cosines (as it is; the anchors' mean removed first, `-m`) one
    block a group of `pm['groups']` (no feature name is written into this function): a row a view kind (the probe's order), a column a
    feature (the block's order), a cell `hit/consumed` (hit: the share of queries whose own anchor is the nearest of its cell's;
    consumed: (1 - cos to the own anchor) / (1 - the anchors' mean cos), 0 = untouched, 1 = as far as another place). The first row of
    a block, `anchors nn/neg`, is the anchors among themselves (`pm['anchors']`): over the other anchors of a cell, the largest / the mean
    cosine, the median over the anchors. Then `map`: a row each for the anchor and the twins, the decoder's Co-figure against the
    anchor's own, cos a signal; `map-direct` (when measured): the same with the view's own Co-figure, no network. Last: the effective
    rank of the anchors' cls', the places, the cells, the seconds."""
    feats: Dict[str, Dict[str, Dict[str, float]]] = pm['features']
    w: int = 13
    out: List[str] = []
    for tag, title in (('', 'the cosine as it is'), ('-m', 'the anchors\' mean removed first')):
        for block, names in pm['groups']:
            cols: List[str] = [n + tag for n in names]
            wc: int = max(13, 2 + max(len(c) for c in cols))
            kinds: List[str] = list(feats[cols[0]])
            out.append(f'     probe, {title}, {block}: hit/consumed (first row: the anchors among themselves, nearest/mean cosine)')
            out.append(f'     {"":<15}' + ''.join(f'{c:>{wc}}' for c in cols))
            out.append(f'     {"anchors nn/neg":<15}' + ''.join('{:>{w}}'.format('{:.2f}/{:.2f}'.format(pm['anchors'][c]['nn'], pm['anchors'][c]['neg']), w=wc)
                                                              for c in cols))
            for kd in kinds:
                cells: str = ''.join('{:>{w}}'.format('{:.2f}/{:.2f}'.format(feats[c][kd]['hit'], feats[c][kd]['consumed']) if kd in feats[c] else '-', w=wc)
                                     for c in cols)
                out.append(f'     {kd:<15}' + cells)
    maps: Dict[str, Dict[str, float]] = pm['map']
    if maps:
        parts: List[str] = list(next(iter(maps.values())))
        out.append('     map: the decoder\'s Co-figure of the view against the anchor\'s own, cos a signal')
        out.append(f'     {"":<14}' + ''.join(f'{p:>{w}}' for p in parts))
        for kd, v in maps.items():
            out.append(f'     {kd:<14}' + ''.join(f'{v[p]:>{w}.3f}' for p in parts))
    map_direct: Dict[str, Dict[str, float]] = pm['map_direct']
    if map_direct:
        parts = list(next(iter(map_direct.values())))
        out.append('     map-direct: the view\'s OWN Co-figure against the anchor\'s own (no network), cos a signal')
        out.append(f'     {"":<14}' + ''.join(f'{p:>{w}}' for p in parts))
        for kd, v in map_direct.items():
            out.append(f'     {kd:<14}' + ''.join(f'{v[p]:>{w}.3f}' for p in parts))
    out.append(f'     rank {pm["rank"]:.1f}  places {pm["places"]}  cells {pm["cells"]}  seconds {pm["seconds"]:.0f}')
    return '\n'.join(out)


