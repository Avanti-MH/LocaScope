#!/usr/bin/env python3
"""What is actually in a slide's feature space, and where does mpp live in it?

Every stage-1 experiment so far has asked "does method X estimate mpp better",
and three answers came back closed: the aggregation is not the bottleneck, more
reference tiles buy almost nothing, and clustering follows magnification rather
than tissue. Those all treat the 1536 numbers per tile as a black box.

This asks about the box. Two questions, in order:

    1. How many directions does this space really have, and what are they?
    2. Which of them carry mpp -- and how few would be enough?

The second question is the one with a consequence. A reference bank holds a few
thousand tiles in 1536 dimensions, which is a badly conditioned place to run a
nearest-neighbour search. If mpp turns out to live on three axes, the same KNN
in three dimensions is a far better posed problem than the one production runs
today, and that would be a bigger change than any amount of better arithmetic on
top of the full space.

Five numbers, no more
---------------------
Each carries its own null, because a value without one is not evidence:

    n_significant       components that beat a parallel-analysis null
                        (null: the same data with each dimension shuffled
                        independently, which destroys correlation between
                        dimensions and keeps every marginal)

    var_between         trace(S_B) / trace(S_T) -- the share of the variance
                        that is level rather than anything else

    corr_logmpp[pc]     how much each component tracks scale

    corr_white[pc]      how much each component tracks BACKGROUND instead.
                        Not decoration: the median grid position inside a
                        tissue region is 72% background and 46% are pure
                        background (result/RefStore), so "this component is
                        mpp" and "this component is emptiness" are genuinely
                        easy to confuse, and one of them is a dead end.

    r2(r)               how much of log mpp the first r components explain.
                        Because the components are orthogonal this is just the
                        running sum of corr_logmpp squared -- no fit needed, and
                        it is exact.
                        (null: r RANDOM directions. If those do as well, the
                        answer is that any r dimensions would, and the mpp
                        subspace is not special.)

Reads stores only: no GPU, no WSI, no model. One slide at a time on purpose --
comparing slides is a later step and would only make the first look at this
noisier.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _directory in ('utilities', 'aiNNModel'):
    _path = str(_ROOT / _directory)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import matplotlib                                                   # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402

import FeatureStore as FeatureStoreModule                           # noqa: E402
from GigaPathFunc import pooling_kinds                                # noqa: E402
import _paths                                                       # noqa: E402
from _paths import job_result_dir                                   # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_level(store_root, wsi_stem, level, pooling, sampler_id=None):
    """One level's tiles: features, coordinates, and background fraction.

    Returns (features, coords, white_fraction, meta):
        features        [n_tiles, dim] float32, L2-normalized
        coords          [n_tiles, 2] int64, level-0 top-left
        white_fraction  [n_tiles] float32, or all-NaN when the store predates it

    find_one rather than find: a root can hold two stores for the same slide and
    level that differ only in how their tiles were chosen, and picking whichever
    sorted first would describe a feature space nobody built.
    """
    selector = {} if sampler_id is None else {'sampler_id': sampler_id}
    path = FeatureStoreModule.find_one(
        store_root, what=f'reference store for {wsi_stem} L{level}',
        wsi_stem=wsi_stem, level=level, pooling='tokens', **selector)

    wanted = ['features', 'x', 'y']
    meta = FeatureStoreModule.load_meta(path)
    tensors, _ = FeatureStoreModule.load(path, keys=wanted)

    # white_frac only exists in stores written by the quota sampler. Asking for a
    # key that is not there raises inside safetensors, so it is fetched
    # separately and its absence reported rather than guessed at.
    try:
        extra, _ = FeatureStoreModule.load(path, keys=['white_frac'])
        white_fraction = extra['white_frac'].numpy().astype(np.float32)
    except Exception:                                     # noqa: BLE001
        white_fraction = np.full(tensors['features'].shape[0], np.nan,
                                 dtype=np.float32)

    slots = pooling_kinds(tensors['features'].float(), pooling, meta)
    features = torch.nn.functional.normalize(
        slots.reshape(slots.shape[0], -1), dim=-1)

    coords = np.stack([tensors['x'].numpy(), tensors['y'].numpy()],
                      axis=1).astype(np.int64)
    return features, coords, white_fraction, meta


def load_slide_balanced(store_root, wsi_stem, pooling, per_level,
                        sampler_id=None, seed=42):
    """Every level of one slide, cut to the same number of tiles each.

    Balancing is not tidiness. PCA finds the directions of greatest variance,
    and a level contributing three times as many tiles contributes three times
    as much variance -- so on an unbalanced set the leading component can be
    "which level has the most tiles" wearing the costume of a finding.
    """
    levels = sorted({meta.level for meta in
                     (FeatureStoreModule.load_meta(p)
                      for p in FeatureStoreModule.find(store_root,
                                                       wsi_stem=wsi_stem,
                                                       pooling='tokens'))})
    if not levels:
        raise FileNotFoundError(f'no reference store for {wsi_stem}')

    loaded = {}
    for level in levels:
        loaded[level] = load_level(store_root, wsi_stem, level, pooling,
                                   sampler_id)

    take = min(per_level, min(f.shape[0] for f, _, _, _ in loaded.values()))
    rng = np.random.default_rng(seed)

    feature_blocks, coord_blocks, white_blocks = [], [], []
    level_index, level_mpp = [], {}
    for position, level in enumerate(levels):
        features, coords, white_fraction, meta = loaded[level]
        chosen = rng.choice(features.shape[0], size=take, replace=False)
        feature_blocks.append(features[chosen])
        coord_blocks.append(coords[chosen])
        white_blocks.append(white_fraction[chosen])
        level_index.append(np.full(take, position, dtype=np.int64))
        level_mpp[position] = float(meta.mpp)

    return dict(
        features=torch.cat(feature_blocks, dim=0),
        coords=np.concatenate(coord_blocks, axis=0),
        white_fraction=np.concatenate(white_blocks),
        level_index=np.concatenate(level_index),
        level_mpp=level_mpp,
        levels=levels,
        per_level=take,
        wsi_path=loaded[levels[0]][3].wsi_path,
        tile_size=loaded[levels[0]][3].tile_size,
        level_ds={i: float(loaded[lv][3].ds) for i, lv in enumerate(levels)},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  The five numbers
# ══════════════════════════════════════════════════════════════════════════════

def principal_axes(features: torch.Tensor):
    """Eigen-decomposition of the centred covariance, largest first.

    Returns (eigenvalues [dim], axes [dim, dim], projections [n, dim]).
    Full decomposition rather than a truncated one: the eigenvalue tail is what
    the parallel-analysis null is compared against, so throwing it away would
    remove the thing that decides how many components are real.
    """
    centred = (features - features.mean(dim=0, keepdim=True)).double()
    covariance = (centred.T @ centred) / max(1, centred.shape[0] - 1)
    eigenvalues, axes = torch.linalg.eigh(covariance)     # ascending
    eigenvalues = eigenvalues.flip(0)
    axes = axes.flip(1)
    return (eigenvalues.numpy(), axes.float(),
            (centred.float() @ axes.float()).numpy())


def shuffled_eigenvalues(features: torch.Tensor, n_repeats: int, seed: int):
    """Parallel analysis: the eigenvalues of the same data with every dimension
    shuffled on its own.

    Shuffling within a column destroys the correlation BETWEEN dimensions while
    leaving each dimension's own distribution untouched, so the result is what
    this many samples in this many dimensions produce when there is no shared
    structure at all. Components above it are real; the ones below are the
    eigenvalue spread you get for free from finite sampling, which is what makes
    an eyeballed scree elbow such an unreliable count.
    """
    rng = np.random.default_rng(seed)
    values = features.numpy()
    stacked = []
    for _ in range(n_repeats):
        shuffled = np.empty_like(values)
        for column in range(values.shape[1]):
            shuffled[:, column] = rng.permutation(values[:, column])
        centred = torch.from_numpy(shuffled).double()
        centred = centred - centred.mean(dim=0, keepdim=True)
        covariance = (centred.T @ centred) / max(1, centred.shape[0] - 1)
        stacked.append(torch.linalg.eigh(covariance)[0].flip(0).numpy())
    return np.mean(np.stack(stacked), axis=0)


def variance_share_between_levels(features: torch.Tensor,
                                  level_index: np.ndarray) -> float:
    """trace(S_B) / trace(S_T): how much of the spread is level, not tissue.

    One number for "is this a space organised by magnification, or one where
    magnification is a minor axis". It needs no fitting and no threshold.
    """
    values = features.double()
    grand_mean = values.mean(dim=0, keepdim=True)
    total = float(((values - grand_mean) ** 2).sum())
    between = 0.0
    for level in np.unique(level_index):
        member = torch.from_numpy(level_index == level)
        group = values[member]
        between += float(group.shape[0]
                         * ((group.mean(dim=0, keepdim=True) - grand_mean) ** 2).sum())
    return between / total if total else float('nan')


def correlation_with(projections: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pearson correlation of every component with one per-tile quantity."""
    if np.all(np.isnan(target)):
        return np.full(projections.shape[1], np.nan)
    finite = ~np.isnan(target)
    centred_target = target[finite] - target[finite].mean()
    denominator = np.sqrt((centred_target ** 2).sum())
    out = np.zeros(projections.shape[1])
    for component in range(projections.shape[1]):
        column = projections[finite, component]
        column = column - column.mean()
        scale = np.sqrt((column ** 2).sum()) * denominator
        out[component] = (column @ centred_target) / scale if scale else 0.0
    return out


def r2_of_random_subspace(features: torch.Tensor, target: np.ndarray,
                          n_dimensions: int, n_repeats: int,
                          seed: int) -> tuple:
    """The same question asked of r directions chosen at random.

    The decoy for r2(r). A subspace that explains log mpp well is only
    interesting if an arbitrary subspace of the same size does not, and with
    1536 dimensions and a few thousand tiles an arbitrary one explains more than
    intuition expects. Returns (mean, std) over repeats.
    """
    rng = np.random.default_rng(seed)
    centred = (features - features.mean(dim=0, keepdim=True)).numpy()
    centred_target = target - target.mean()
    total = float((centred_target ** 2).sum())
    scores = []
    for _ in range(n_repeats):
        basis = rng.normal(size=(centred.shape[1], n_dimensions))
        basis, _ = np.linalg.qr(basis)
        projected = centred @ basis
        coefficients, *_ = np.linalg.lstsq(projected, centred_target, rcond=None)
        residual = centred_target - projected @ coefficients
        scores.append(1.0 - float((residual ** 2).sum()) / total)
    return float(np.mean(scores)), float(np.std(scores))


# ══════════════════════════════════════════════════════════════════════════════
#  Which tiles sit at the ends of each axis
# ══════════════════════════════════════════════════════════════════════════════

def extreme_tiles(slide, projections, n_components, n_each) -> list:
    """The n_each tiles at each end of each of the first n_components axes.

    A direction has no meaning until something is seen along it. The correlations
    say whether a component tracks scale or emptiness; only the images say
    whether it is fat against stroma or something with no name. Every field
    needed to read the tile back is carried, so the CSV stands alone:

        SafeSlide(wsi_path).read_region_rgb((x, y), level, (tile, tile))
    """
    rows = []
    for component in range(min(n_components, projections.shape[1])):
        order = np.argsort(projections[:, component])
        for side, indices in (('low', order[:n_each]),
                              ('high', order[-n_each:][::-1])):
            for rank, index in enumerate(indices):
                position = int(slide['level_index'][index])
                rows.append(dict(
                    wsi_stem=slide['wsi_stem'], pooling=slide['pooling'],
                    pc=component + 1, side=side, rank=rank + 1,
                    projection=round(float(projections[index, component]), 6),
                    wsi_path=slide['wsi_path'],
                    level=slide['levels'][position],
                    ds=slide['level_ds'][position],
                    mpp=round(slide['level_mpp'][position], 6),
                    x=int(slide['coords'][index, 0]),
                    y=int(slide['coords'][index, 1]),
                    tile_size=slide['tile_size'],
                    white_frac=round(float(slide['white_fraction'][index]), 4)
                    if not np.isnan(slide['white_fraction'][index]) else '',
                ))
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  Output
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(rows, path) -> None:
    if not rows:
        print(f'  (nothing to write to {os.path.basename(path)})')
        return
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'  {os.path.basename(path)}  {len(rows)} rows -> {path}')


def plot_scree(component_rows, path, n_show=60) -> None:
    rows = component_rows[:n_show]
    fig, axis = plt.subplots(figsize=(8, 4.6))
    axis.plot([r['pc'] for r in rows], [r['eigenvalue'] for r in rows],
              marker='o', ms=3, label='observed')
    axis.plot([r['pc'] for r in rows], [r['null_eigenvalue'] for r in rows],
              ls='--', label='shuffled null (parallel analysis)')
    n_significant = sum(1 for r in component_rows if r['significant'])
    axis.axvline(n_significant + 0.5, color='k', ls=':', lw=1.4)
    axis.text(n_significant + 0.5, axis.get_ylim()[1] * 0.9,
              f'  {n_significant} above the null', fontsize=9)
    axis.set_yscale('log')
    axis.set_xlabel('component')
    axis.set_ylabel('eigenvalue (log)')
    axis.set_title('How many directions are real?', fontsize=11)
    axis.legend(fontsize=9)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  {os.path.basename(path)} -> {path}')


def plot_correlations(component_rows, path, n_show=30) -> None:
    """The figure that separates scale from emptiness.

    A component can track log mpp because the tissue really does look different
    at another magnification, or because coarse tiles cover more background.
    Those are the same number until they are drawn side by side.
    """
    rows = component_rows[:n_show]
    positions = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(9.5, 4.6))
    axis.bar(positions - 0.2, [abs(r['corr_logmpp']) for r in rows], width=0.4,
             label='|corr| with log mpp')
    white = [abs(r['corr_white']) if r['corr_white'] == r['corr_white'] else 0.0
             for r in rows]
    axis.bar(positions + 0.2, white, width=0.4,
             label='|corr| with background fraction')
    axis.set_xticks(positions)
    axis.set_xticklabels([r['pc'] for r in rows], fontsize=7)
    axis.set_xlabel('component')
    axis.set_ylabel('|correlation|')
    axis.set_title('Which components carry scale, and which carry emptiness?',
                   fontsize=11)
    axis.legend(fontsize=9)
    axis.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  {os.path.basename(path)} -> {path}')


def plot_r2(component_rows, random_rows, path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.6))
    axis.plot([r['pc'] for r in component_rows],
              [r['r2_cumulative'] for r in component_rows],
              marker='o', ms=3, label='first r components')
    if random_rows:
        dims = [r['r'] for r in random_rows]
        mean = np.array([r['r2_random_mean'] for r in random_rows])
        deviation = np.array([r['r2_random_std'] for r in random_rows])
        axis.plot(dims, mean, ls='--', color='crimson',
                  label='r random directions (decoy)')
        axis.fill_between(dims, mean - deviation, mean + deviation,
                          color='crimson', alpha=0.15)
    axis.set_xscale('log')
    axis.set_xlabel('r (dimensions kept)')
    axis.set_ylabel('R² for log mpp')
    axis.set_ylim(0, 1.02)
    axis.set_title('How few dimensions hold the scale?', fontsize=11)
    axis.legend(fontsize=9)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  {os.path.basename(path)} -> {path}')


def plot_scatter(projection_rows, path) -> None:
    levels = sorted({r['level'] for r in projection_rows})
    pc1 = np.array([r['pc1'] for r in projection_rows])
    pc2 = np.array([r['pc2'] for r in projection_rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    colours = plt.get_cmap('viridis')
    for position, level in enumerate(levels):
        member = np.array([r['level'] == level for r in projection_rows])
        axes[0].scatter(pc1[member], pc2[member], s=6, linewidths=0,
                        color=colours(position / max(1, len(levels) - 1)),
                        label=f'L{level}')
    axes[0].set_title('coloured by level', fontsize=10)
    axes[0].legend(fontsize=8)

    white = np.array([r['white_frac'] if r['white_frac'] != '' else np.nan
                      for r in projection_rows], dtype=float)
    if not np.all(np.isnan(white)):
        dots = axes[1].scatter(pc1, pc2, c=white, s=6, linewidths=0,
                               cmap='magma', vmin=0, vmax=1)
        fig.colorbar(dots, ax=axes[1], label='background fraction')
        axes[1].set_title('coloured by background fraction', fontsize=10)
    else:
        axes[1].text(0.5, 0.5, 'no white_frac in this store',
                     ha='center', transform=axes[1].transAxes)
        axes[1].set_title('coloured by background fraction', fontsize=10)

    for axis in axes:
        axis.set_xlabel('PC1')
        axis.set_ylabel('PC2')
        axis.grid(alpha=0.25)
    fig.suptitle('The same tiles, read two ways', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  {os.path.basename(path)} -> {path}')


# ══════════════════════════════════════════════════════════════════════════════

def analyse_slide(store_root, wsi_stem, args, out_dir,
                  summary_rows, component_rows, decoy_rows,
                  projection_rows, extreme_rows) -> None:
    """One slide, appending to the shared row lists.

    Every row already carries wsi_stem, so several slides pool into one set of
    CSVs without a second key. The per-slide numbers stay separate -- nothing
    here averages across slides, because whether the axes agree BETWEEN slides
    is a different question and answering it by accident would be worse than not
    answering it.
    """
    slide = load_slide_balanced(store_root, wsi_stem, args.pooling,
                                args.per_level, args.sampler_id, args.seed)
    slide['wsi_stem'] = wsi_stem
    slide['pooling'] = args.pooling
    features = slide['features']
    n_tiles, dim = features.shape
    has_white = not np.all(np.isnan(slide['white_fraction']))

    print(f'{wsi_stem}   pooling {args.pooling}')
    print(f'  levels {slide["levels"]}   {slide["per_level"]} tiles each   '
          f'{n_tiles} x {dim}')
    if n_tiles < dim:
        print(f'  !! {n_tiles} tiles in {dim} dimensions: the covariance is '
              f'singular, so r2_full is 1.0\n     by construction and every '
              f'component past {n_tiles - 1} is an artefact of that. The '
              f'binding\n     constraint is the coarsest level, which offers '
              f'the fewest tiles.')
    if not has_white:
        print('  !! this store has no white_frac, so "is this component scale '
              'or emptiness"\n     cannot be answered here -- it needs a store '
              'from the quota sampler')

    eigenvalues, _axes, projections = principal_axes(features)
    null_eigenvalues = shuffled_eigenvalues(features, args.null_repeats,
                                            args.seed)
    log_mpp = np.array([np.log(slide['level_mpp'][int(i)])
                        for i in slide['level_index']])
    corr_logmpp = correlation_with(projections, log_mpp)
    corr_white = correlation_with(projections, slide['white_fraction'])
    var_between = variance_share_between_levels(features, slide['level_index'])

    significant = eigenvalues > null_eigenvalues
    n_significant = int(np.argmin(significant)) if not significant.all() \
        else int(len(significant))
    r2_cumulative = np.cumsum(corr_logmpp ** 2)

    component_rows.extend(dict(
        wsi_stem=wsi_stem, pooling=args.pooling, pc=i + 1,
        eigenvalue=round(float(eigenvalues[i]), 9),
        null_eigenvalue=round(float(null_eigenvalues[i]), 9),
        significant=int(i < n_significant),
        explained_var=round(float(eigenvalues[i] / eigenvalues.sum()), 6),
        corr_logmpp=round(float(corr_logmpp[i]), 5),
        corr_white=(round(float(corr_white[i]), 5) if has_white else float('nan')),
        r2_cumulative=round(float(min(r2_cumulative[i], 1.0)), 5),
    ) for i in range(dim))

    for dimension in [d for d in (1, 2, 3, 5, 10, 20, 50, 100) if d <= dim]:
        mean, deviation = r2_of_random_subspace(
            features, log_mpp, dimension, args.decoy_repeats, args.seed)
        decoy_rows.append(dict(
            wsi_stem=wsi_stem, pooling=args.pooling, r=dimension,
            r2_first_r=round(float(min(r2_cumulative[dimension - 1], 1.0)), 5),
            r2_random_mean=round(mean, 5), r2_random_std=round(deviation, 5)))

    reached = int(np.argmax(r2_cumulative >= 0.9 * r2_cumulative[-1]) + 1)
    strongest = int(np.argmax(np.abs(corr_logmpp)))
    summary_rows.append(dict(
        wsi_stem=wsi_stem, pooling=args.pooling,
        n_tiles=n_tiles, n_levels=len(slide['levels']), dim=dim,
        n_significant=n_significant,
        var_between=round(var_between, 5),
        top_scale_pc=strongest + 1,
        corr_top_scale=round(float(corr_logmpp[strongest]), 4),
        corr_white_of_that_pc=(round(float(corr_white[strongest]), 4)
                               if has_white else float('nan')),
        r2_at_1=round(float(min(r2_cumulative[0], 1.0)), 4),
        r2_at_5=round(float(min(r2_cumulative[min(4, dim - 1)], 1.0)), 4),
        r_for_90pct=reached,
        has_white_frac=int(has_white)))

    projection_rows.extend(dict(
        wsi_stem=wsi_stem, pooling=args.pooling,
        level=slide['levels'][int(slide['level_index'][i])],
        mpp=round(slide['level_mpp'][int(slide['level_index'][i])], 6),
        white_frac=(round(float(slide['white_fraction'][i]), 4)
                    if has_white else ''),
        x=int(slide['coords'][i, 0]), y=int(slide['coords'][i, 1]),
        **{f'pc{c + 1}': round(float(projections[i, c]), 5)
           for c in range(min(10, dim))},
    ) for i in range(n_tiles))

    extreme_rows.extend(extreme_tiles(slide, projections,
                                      args.extreme_components,
                                      args.extreme_tiles))

    print(f'  {n_significant} directions above the null   '
          f'var_between {var_between:.3f}   '
          f'PC{strongest + 1} corr {corr_logmpp[strongest]:+.3f}   '
          f'R2 at r=1 {r2_cumulative[0]:.3f}, r=5 '
          f'{min(r2_cumulative[min(4, dim - 1)], 1.0):.3f}\n', flush=True)


def _by_slide(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['wsi_stem'], []).append(row)
    return {stem: grouped[stem] for stem in sorted(grouped)}


def plot_scree(component_rows, path, n_show=60) -> None:
    """One line per slide against its own shuffled null.

    The nulls are drawn too, and faintly: they sit almost on top of each other
    because they depend on the sample count and the dimension rather than on the
    slide, which is the point -- the same yardstick for every curve.
    """
    fig, axis = plt.subplots(figsize=(9, 5))
    for stem, rows in _by_slide(component_rows).items():
        shown = rows[:n_show]
        n_significant = sum(1 for r in rows if r['significant'])
        line, = axis.plot([r['pc'] for r in shown],
                          [r['eigenvalue'] for r in shown],
                          marker='o', ms=2.5, label=f'{stem}  ({n_significant})')
        axis.plot([r['pc'] for r in shown],
                  [r['null_eigenvalue'] for r in shown],
                  ls='--', lw=0.8, alpha=0.35, color=line.get_color())
    axis.set_yscale('log')
    axis.set_xlabel('component')
    axis.set_ylabel('eigenvalue (log)')
    axis.set_title('How many directions are real?   '
                   '(dashed = that slide\'s shuffled null; '
                   'the count is in the legend)', fontsize=10)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  {os.path.basename(path)} -> {path}')


def plot_correlations(component_rows, path, n_show=15) -> None:
    """One panel per slide: scale against emptiness, component by component.

    Kept as panels rather than an overlay because the question is asked of each
    slide separately -- whether PC1 means the same thing on two slides is a
    different question, and drawing them on one axis would invite answering it
    by eye when the axes are not comparable.
    """
    grouped = _by_slide(component_rows)
    n_cols = min(3, len(grouped))
    n_rows = int(np.ceil(len(grouped) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 3.4 * n_rows),
                             squeeze=False)
    for axis, (stem, rows) in zip(axes.ravel(), grouped.items()):
        shown = rows[:n_show]
        positions = np.arange(len(shown))
        axis.bar(positions - 0.2, [abs(r['corr_logmpp']) for r in shown],
                 width=0.4, label='log mpp')
        axis.bar(positions + 0.2,
                 [abs(r['corr_white']) if r['corr_white'] == r['corr_white']
                  else 0.0 for r in shown],
                 width=0.4, label='background')
        axis.set_xticks(positions)
        axis.set_xticklabels([r['pc'] for r in shown], fontsize=6)
        axis.set_ylim(0, 1)
        axis.set_title(stem, fontsize=9)
        axis.grid(alpha=0.3, axis='y')
    axes.ravel()[0].legend(fontsize=8)
    for axis in axes.ravel()[len(grouped):]:
        axis.axis('off')
    fig.suptitle('|correlation| of each component with scale and with emptiness',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  {os.path.basename(path)} -> {path}')


def plot_r2(component_rows, decoy_rows, path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    for stem, rows in _by_slide(component_rows).items():
        axis.plot([r['pc'] for r in rows], [r['r2_cumulative'] for r in rows],
                  marker='o', ms=2, label=stem)
    grouped_decoy = _by_slide(decoy_rows)
    if grouped_decoy:
        dims = sorted({r['r'] for r in decoy_rows})
        mean = np.array([np.mean([r['r2_random_mean'] for r in decoy_rows
                                  if r['r'] == d]) for d in dims])
        spread = np.array([np.mean([r['r2_random_std'] for r in decoy_rows
                                    if r['r'] == d]) for d in dims])
        axis.plot(dims, mean, ls='--', color='crimson', lw=2,
                  label='r random directions (decoy, all slides)')
        axis.fill_between(dims, mean - spread, mean + spread,
                          color='crimson', alpha=0.12)
    axis.set_xscale('log')
    axis.set_xlabel('r (dimensions kept)')
    axis.set_ylabel('R² for log mpp')
    axis.set_ylim(0, 1.02)
    axis.set_title('How few dimensions hold the scale?   '
                   'The gap to the decoy at SMALL r is the evidence', fontsize=10)
    axis.legend(fontsize=8, loc='lower right')
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  {os.path.basename(path)} -> {path}')


def plot_scatter(projection_rows, path) -> None:
    grouped = _by_slide(projection_rows)
    n_cols = min(4, len(grouped))
    n_rows = int(np.ceil(len(grouped) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 3.8 * n_rows),
                             squeeze=False)
    colours = plt.get_cmap('viridis')
    for axis, (stem, rows) in zip(axes.ravel(), grouped.items()):
        levels = sorted({r['level'] for r in rows})
        pc1 = np.array([r['pc1'] for r in rows])
        pc2 = np.array([r['pc2'] for r in rows])
        for position, level in enumerate(levels):
            member = np.array([r['level'] == level for r in rows])
            axis.scatter(pc1[member], pc2[member], s=4, linewidths=0,
                         color=colours(position / max(1, len(levels) - 1)),
                         label=f'L{level}')
        axis.set_title(stem, fontsize=9)
        axis.set_xlabel('PC1', fontsize=8)
        axis.set_ylabel('PC2', fontsize=8)
        axis.tick_params(labelsize=7)
        axis.legend(fontsize=6)
        axis.grid(alpha=0.25)
    for axis in axes.ravel()[len(grouped):]:
        axis.axis('off')
    fig.suptitle('PC1 against PC2, coloured by level', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  {os.path.basename(path)} -> {path}')


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('wsi_stem', nargs='+',
                        help='one or more slides, e.g. BRACS_1228')
    parser.add_argument('--stores',
                        default=str(Path(_paths.RESULT_DIR) / 'cache' / 'features'))
    parser.add_argument(
        '--out', default='',
        help='output directory, used verbatim. Default: '
             'result/<SLURM_JOB_NAME or FeatureAxes>/<encoder>/, where the '
             'encoder is the last component of --stores. That level is added '
             'only to the derived path, so name it yourself when you pass one.')
    parser.add_argument('--pooling', default='cls')
    parser.add_argument('--sampler-id', default=None,
                        help="which sampling rule's stores to read when a root "
                             "holds more than one; '' selects the pre-quota draws")
    parser.add_argument('--per-level', type=int, default=1000)
    parser.add_argument('--null-repeats', type=int, default=5,
                        help='shuffles for the parallel-analysis null')
    parser.add_argument('--decoy-repeats', type=int, default=10,
                        help='random subspaces drawn per r')
    parser.add_argument('--extreme-components', type=int, default=10)
    parser.add_argument('--extreme-tiles', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    store_root = (args.stores if os.path.isabs(args.stores)
                  else str(_ROOT / args.stores))

    # This bench has no encoder of its own -- it reads features somebody else
    # wrote -- so the tag comes from the input. The store root's last component
    # IS the encoder tag under the output convention (result/cache/features/
    # gigapath/), which is readable; meta.encoder_id would also identify the
    # encoder but it is eight hex characters of sha256 and cannot be inverted,
    # so a directory named after it would say nothing to whoever opens it.
    # Pointed somewhere else, this degrades to naming the directory the features
    # came from, which is still the honest answer to "whose features are these".
    # Derived path only -- an explicit --out is used verbatim.
    out_dir = args.out or job_result_dir(
        'FeatureAxes',
        encoder=os.path.basename(os.path.normpath(store_root)))
    os.makedirs(out_dir, exist_ok=True)

    summary_rows, component_rows, decoy_rows = [], [], []
    projection_rows, extreme_rows, failures = [], [], []
    for wsi_stem in args.wsi_stem:
        try:
            analyse_slide(store_root, wsi_stem, args, out_dir, summary_rows,
                          component_rows, decoy_rows, projection_rows,
                          extreme_rows)
        except Exception as error:                          # noqa: BLE001
            failures.append((wsi_stem, f'{type(error).__name__}: {error}'))
            print(f'  {wsi_stem}: FAILED -- {type(error).__name__}: {error}\n',
                  flush=True)

    write_csv(summary_rows, os.path.join(out_dir, 'axes_summary.csv'))
    write_csv(component_rows, os.path.join(out_dir, 'axes_components.csv'))
    write_csv(decoy_rows, os.path.join(out_dir, 'axes_r2_decoy.csv'))
    write_csv(projection_rows, os.path.join(out_dir, 'axes_projection.csv'))
    write_csv(extreme_rows, os.path.join(out_dir, 'axes_extremes.csv'))

    if component_rows:
        plot_scree(component_rows, os.path.join(out_dir, 'axes_scree.png'))
        plot_correlations(component_rows,
                          os.path.join(out_dir, 'axes_correlation.png'))
        plot_r2(component_rows, decoy_rows, os.path.join(out_dir, 'axes_r2.png'))
        plot_scatter(projection_rows, os.path.join(out_dir, 'axes_scatter.png'))

    if summary_rows:
        print('\n' + '=' * 78)
        print(f'{"slide":<24}{"real":>6}{"var_lv":>8}{"top PC":>8}'
              f'{"corr":>8}{"bg":>8}{"R2@1":>8}{"R2@5":>8}{"r90":>5}')
        for row in summary_rows:
            print(f'{row["wsi_stem"]:<24}{row["n_significant"]:>6}'
                  f'{row["var_between"]:>8.3f}{row["top_scale_pc"]:>8}'
                  f'{row["corr_top_scale"]:>+8.3f}'
                  f'{row["corr_white_of_that_pc"]:>+8.3f}'
                  f'{row["r2_at_1"]:>8.3f}{row["r2_at_5"]:>8.3f}'
                  f'{row["r_for_90pct"]:>5}')
        print('\n"top PC" is the component most correlated with log mpp, and '
              '"bg" is that same\ncomponent\'s correlation with background. A '
              'scale axis has a large corr and a\nsmall bg; the two being '
              'close together is the confound, not a finding.')
        print('=' * 78)

    if failures:
        print(f'\n{len(failures)} slide(s) failed:')
        for stem, message in failures:
            print(f'  {stem}: {message}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
