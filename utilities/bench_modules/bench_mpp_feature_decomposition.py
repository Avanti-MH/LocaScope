#!/usr/bin/env python3
"""Where does scale live in a slide's feature space, and does routing get
better if a candidate method uses that instead of the raw 1536-D space?

Three parts, selected with `--parts` (`all` runs every one):

    axes             feature_axes_analysis (was bench_feature_axes.py). Purely
                     descriptive -- PCA + a parallel-analysis null, which
                     components track log mpp, which track background instead.
                     NEVER runs a KNN, NEVER produces a routing decision.
    subspace_knn     was bench_subspace_knn.py. Step 5 of the same
                     investigation: does a KNN run in the subspace `axes`
                     found actually route better? Reads the SAME cached
                     FeatureStore reference/query stores `axes` does, with the
                     synthetic-camera-distortion query set (arm B) that gives
                     this part its domain gap. This is a routing method, not
                     analysis -- every setting produces a real accuracy number.
    sampler_routing  NEW (2026-09-14). Neither of the above ever draws a fresh
                     tile: both read whatever is already sitting in
                     `result/cache/features/`. This part samples its OWN,
                     UNCACHED tiles straight off the WSI (`TileSampler`, never
                     written to `result/cache/tiles/`) and runs the CURRENT
                     production baseline (a plain KNN vote) alongside a
                     projected-subspace candidate on the exact same fresh
                     draw -- the comparison arena any FUTURE routing method
                     (Prototypical/Relation/Siamese, still just reading, not
                     built yet) drops into next to the baseline it has to beat.
                     Both run through `patch_vote`/`fit_basis`/
                     `select_components`/`project` directly (the same free
                     functions `subspace_knn` itself uses), not through a
                     class -- see below.

The production estimators' head-to-head (`stage1_compare`) moved out to
bench_stage1_mpp.py on 2026-09-29: it reads no store, and its result
directory (`result/Stage1MppBench/`) is not one of these parts'. Jobscript:
jobscripts/MppRoutingExp.sh (was Benchmarks/MppFeatureDecomposition.sh and
SubspaceKnn.sh).

NOT A `SubspaceKnnEstMpp` CLASS MIMICKING `KnnEstMpp` (removed 2026-09-17).
An earlier draft of `sampler_routing` was written to need one -- the comment
this replaces argued that running the baseline and a subspace candidate side
by side needed the same staged, inspectable shape
(`build_samples`/`build_ref_features`/`build_query_features`/`estimate`) a
`KnnEstMpp`-alike would give it. What actually got built instead
(`run_sampler_routing`, below) calls `fit_basis`/`select_components`/
`project`/`knn_labels` directly on tensors it samples and encodes itself --
the class the comment described was never once constructed anywhere in this
file. Confirmed by grep before deleting it: `SubspaceKnnEstiMpp(` had exactly
one call site, inside its own dead `estimate` method.

WHAT THIS MERGE DID NOT CHANGE
=================================
The `axes` and `subspace_knn` bodies are the ORIGINAL bench_feature_axes.py /
bench_subspace_knn.py logic, moved here under `--parts` instead of being two
separate `if __name__ == '__main__'` scripts. Every function, every docstring
rationale, every gate is unchanged. Two per-slide-comparison `plot_*`
functions in the old bench_feature_axes.py were dead code (a same-named
single-slide version defined earlier in the file was shadowed by a second
definition of the same name -- Python keeps the last one, so `main()` only
ever called the second) -- only the live (second) versions are kept here.

Settled findings this file exists to keep visible (2026-08 runs, BRACS_1228 +
six more slides):

    axes            log mpp lives on 2-10 of 1536 directions (R2@1 ~ 0.8 on
                    H&E, collapses to 0.14-0.15 on two Ki67 slides where PC1
                    is taken by something else -- the subspace is NOT
                    stain-universal).
    subspace_knn    the premise was wrong: production already scores 0.990 on
                    clean tile-to-tile (arm A), so there was no ill-
                    conditioning to fix. Ranking components by |corr(log mpp)|
                    ('scale') loses to plain PCA ('variance') at every r,
                    because it walks into the background confound. Whether the
                    dimension cut survives the REAL domain gap (arm B,
                    `uncentred` against `production`) is the only open
                    question with a consequence, and it is answered per-slide
                    in `subspace_knn_accuracy__<slide>.png`.
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# _paths holds the one definition of every package's sys.path entry
# (setup_import_paths) -- utilities/ goes on the path here, by hand, because
# that function is INSIDE it and this is the one step nothing else can do
# for this file. Same idiom every test_modules/cli entry point uses.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / 'utilities'))
import _paths                                                       # noqa: E402
_paths.setup_import_paths()

from Store import FeatureStore                                      # noqa: E402
from GigaPathFunc import pooling_kinds                                # noqa: E402
from stage1_estimation.KnnEstMpp import KnnClassifier                                  # noqa: E402
from _paths import job_result_dir                                   # noqa: E402
from AccessDatasets import locate                                    # noqa: E402
from training.MppRoutingHead.Datasets import add_cache_args         # noqa: E402
from PatchingLib import QueryPatchContainer                          # noqa: E402
from SafeSlide import SafeSlide                                     # noqa: E402
from TissueMaskConfig import MASK_RECIPES                           # noqa: E402
from TileSampler import (OverlapConfig, RichnessConfig,                 # noqa: E402
                         SamplerConfig, TileSampler)
from DsLadder import DEFAULT_RUNGS, DsLadder                        # noqa: E402
from ReadGeometry import sensor_size                                 # noqa: E402
from ReadGeometry import ReadSpec                                    # noqa: E402
from SlideReader import SlideReader                                 # noqa: E402
from pipeline import simulate_microscope_photo                        # noqa: E402


def write_csv(rows, path) -> None:
    if not rows:
        print(f'  (nothing to write to {Path(path).name})')
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, restval='')
        writer.writeheader()
        writer.writerows(rows)
    print(f'  {path}  ({len(rows)} rows)')


# ══════════════════════════════════════════════════════════════════════════════
#  Shared loading -- both `axes` and `subspace_knn` read the same reference
#  stores this way.
# ══════════════════════════════════════════════════════════════════════════════

def load_level(path, pooling):
    """One level's tiles: features, coordinates, and background fraction.

    Returns (features, coords, white_fraction, meta):
        features        [n_tiles, dim] float32, L2-normalized
        coords          [n_tiles, 2] int64, level-0 top-left
        white_fraction  [n_tiles] float32
    """
    tensors, meta = FeatureStore.load(path, keys=['features', 'x', 'y', 'white_frac'])
    slots = pooling_kinds(tensors['features'].float(), pooling, meta)
    features = torch.nn.functional.normalize(
        slots.reshape(slots.shape[0], -1), dim=-1)
    coords = np.stack([tensors['x'].numpy(), tensors['y'].numpy()],
                      axis=1).astype(np.int64)
    return features, coords, tensors['white_frac'].numpy().astype(np.float32), meta


def load_slide_balanced(key_dir, pooling, per_level, seed=42):
    """Every level of one slide's draw, cut to the same number of tiles each.

    Balancing is not tidiness. PCA finds the directions of greatest variance,
    and a level contributing three times as many tiles contributes three
    times as much variance -- so on an unbalanced set the leading component
    can be "which level has the most tiles" wearing the costume of a finding.
    """
    paths = FeatureStore.levels(key_dir, 'tokens')
    levels = sorted(paths)
    loaded = {level: load_level(paths[level], pooling) for level in levels}

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


def _by_slide(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['wsi_stem'], []).append(row)
    return {stem: grouped[stem] for stem in sorted(grouped)}


# ══════════════════════════════════════════════════════════════════════════════
#  PART "axes"  --  feature_axes_analysis (was bench_feature_axes.py)
# ══════════════════════════════════════════════════════════════════════════════

def principal_axes(features: torch.Tensor):
    """Eigen-decomposition of the centred covariance, largest first.

    Returns (eigenvalues [dim], axes [dim, dim], projections [n, dim]). Full
    decomposition rather than a truncated one: the eigenvalue tail is what the
    parallel-analysis null is compared against, so throwing it away would
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
    """Parallel analysis: eigenvalues of the same data with every dimension
    shuffled on its own -- destroys correlation BETWEEN dimensions while
    leaving each dimension's own distribution untouched. Components above it
    are real; the ones below are the eigenvalue spread finite sampling gives
    for free, which is what makes an eyeballed scree elbow unreliable.
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
    """trace(S_B) / trace(S_T): how much of the spread is level, not tissue."""
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
    """R2 of r RANDOM directions -- the decoy for r2(r). If random does as
    well, any r dimensions would, and the mpp subspace is not special."""
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


def extreme_tiles(slide, projections, n_components, n_each) -> list:
    """The n_each tiles at each end of each of the first n_components axes."""
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


def analyse_slide_axes(key_dir, wsi_stem, args, summary_rows, component_rows,
                       decoy_rows, projection_rows, extreme_rows) -> None:
    """One slide, appending to the shared row lists -- see `extreme_tiles`
    etc. Nothing here averages across slides; whether the axes agree BETWEEN
    slides is `--parts axes`'s own separate figure, not this function."""
    slide = load_slide_balanced(key_dir, args.pooling, args.per_level, args.seed)
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
              f'singular, so r2_full is 1.0 by construction and every '
              f'component past {n_tiles - 1} is an artefact of that.')
    if not has_white:
        print('  !! this store has no white_frac, so "is this component '
              'scale or emptiness" cannot be answered here')

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


def plot_axes_scree(component_rows, path, n_show=60) -> None:
    """One line per slide against its own shuffled null."""
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
    print(f'  {Path(path).name} -> {path}')


def plot_axes_correlations(component_rows, path, n_show=15) -> None:
    """One panel per slide: scale against emptiness, component by component."""
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
    print(f'  {Path(path).name} -> {path}')


def plot_axes_r2(component_rows, decoy_rows, path) -> None:
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
    print(f'  {Path(path).name} -> {path}')


def plot_axes_scatter(projection_rows, path) -> None:
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
    print(f'  {Path(path).name} -> {path}')


def _draw_dir(args, wsi_stem):
    """One slide's key directory in `--stores`, under `--seg`'s recipe and the
    `--draw` the dump printed."""
    if not (args.stores and args.draw):
        raise SystemExit('axes / subspace_knn read a stored draw: pass --stores '
                         'and --draw (the dump prints the draw)')
    mask_cfg = MASK_RECIPES[args.seg]
    return FeatureStore.key_dir(args.stores, seg_id=mask_cfg.seg_id(),
                                slide=wsi_stem, region_id=mask_cfg.region_id(),
                                key=args.draw)


def run_axes(args, out_dir: Path) -> int:
    summary_rows, component_rows, decoy_rows = [], [], []
    projection_rows, extreme_rows, failures = [], [], []
    for wsi_stem in args.wsi_stem:
        try:
            analyse_slide_axes(_draw_dir(args, wsi_stem), wsi_stem, args, summary_rows,
                              component_rows, decoy_rows, projection_rows,
                              extreme_rows)
        except Exception as error:                          # noqa: BLE001
            failures.append((wsi_stem, f'{type(error).__name__}: {error}'))
            print(f'  {wsi_stem}: FAILED -- {type(error).__name__}: {error}\n',
                  flush=True)

    write_csv(summary_rows, out_dir / 'axes_summary.csv')
    write_csv(component_rows, out_dir / 'axes_components.csv')
    write_csv(decoy_rows, out_dir / 'axes_r2_decoy.csv')
    write_csv(projection_rows, out_dir / 'axes_projection.csv')
    write_csv(extreme_rows, out_dir / 'axes_extremes.csv')

    if component_rows:
        plot_axes_scree(component_rows, out_dir / 'axes_scree.png')
        plot_axes_correlations(component_rows, out_dir / 'axes_correlation.png')
        plot_axes_r2(component_rows, decoy_rows, out_dir / 'axes_r2.png')
        plot_axes_scatter(projection_rows, out_dir / 'axes_scatter.png')

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
        print('=' * 78)

    if failures:
        print(f'\n{len(failures)} slide(s) failed:')
        for stem, message in failures:
            print(f'  {stem}: {message}')
    return 1 if failures else 0


# ══════════════════════════════════════════════════════════════════════════════
#  PART "subspace_knn"  --  was bench_subspace_knn.py
# ══════════════════════════════════════════════════════════════════════════════

K_NEIGHBOURS = 5
R_VALUES = (1, 2, 3, 5, 10, 20, 50)
R_REPORT_POINTS = (2, 10)
RANDOM_DRAWS = 10
TEST_FRACTION = 0.25
RULES = (('scale', True), ('variance', True), ('random', True),
        ('uncentred', False))

DEFINITIONS = [
    ('production', 'cosine on the raw 1536-D L2-normalised features: no '
                   'projection, no mean removed. What KnnClassifier does today.'),
    ('centred', 'mean removed, ALL components kept. An orthogonal rotation '
                'does not move a cosine, so this row is the mean removal '
                'measured alone.'),
    ('scale', 'the r components with the largest |corr(component, log mpp)|, '
              'mean removed. Ranked on the fit set only.'),
    ('variance', 'the r components with the largest eigenvalue, mean '
                 'removed. Plain PCA.'),
    ('random', f'a uniformly random r-dimensional subspace, mean removed, '
               f'averaged over {RANDOM_DRAWS} draws. The decoy for "any r '
               f'directions would do".'),
    ('uncentred', 'the same directions as `variance`, but projected WITHOUT '
                  'removing the mean. Differs from variance by the constant '
                  'V_r^T mu, which a cosine is not invariant to.'),
    ('chance', 'the largest class share in the test set: what always '
               'predicting one level would score.'),
    ('r', 'how many dimensions are kept. NOT "the r-th component".'),
    ('r = 1', 'meaningless by construction: L2 normalising a scalar leaves '
              'its sign, so every query collapses to one of two values and '
              'every rule scores exactly chance.'),
    ('wins_vs_production', 'test items this setting got right and production '
                           'got wrong. Paired -- the same items go through '
                           'both.'),
    ('losses_vs_production', 'test items production got right and this '
                             'setting got wrong.'),
    ('level accuracy', 'the predicted mpp mapped to the nearest level in '
                       'LOG mpp equals the true level.'),
    ('arm A', 'tile -> tile. Reference and test are both grid tiles from the '
              'reference store. No domain gap.'),
    ('arm B', 'photo -> tile. Queries are synthesised FoV renders carrying '
              'the domain gap, scored per FoV by median-of-medians.'),
    ('split band', 'the test set is the top quartile by x within each level. '
                   'Spatially separated from the reference bank.'),
    ('split random', 'the test set is drawn at random. Leaky, because the '
                     'store holds overlap positions sharing half their '
                     'pixels with a main position.'),
    ('subset white<T', 'every tile whose own background fraction is at or '
                       'above T is dropped from the reference bank, and the '
                       'levels are rebalanced afterwards.'),
    ('coarse share of errors', 'among the wrong answers, the fraction that '
                               'predicted a COARSER level.'),
]


def fit_basis(features: torch.Tensor, mpp_labels: np.ndarray) -> dict:
    """Fit mu and V on ONE set of tiles, and rank the components by scale.

    `features` are already L2-normalised, so the centring is applied to the
    normalised vectors rather than to raw ones. The correlation is measured
    on these same tiles, which is why the caller must pass the FIT set.
    """
    matrix = features.numpy().astype(np.float64)
    mean = matrix.mean(axis=0)
    centred = matrix - mean

    _, singular_values, right_vectors = np.linalg.svd(centred,
                                                      full_matrices=False)
    variance = singular_values ** 2 / max(len(matrix) - 1, 1)
    variance_ratio = variance / variance.sum()

    projection = centred @ right_vectors.T
    log_mpp = np.log(mpp_labels.astype(np.float64))
    log_mpp_centred = log_mpp - log_mpp.mean()
    log_mpp_norm = np.linalg.norm(log_mpp_centred)
    correlation = np.zeros(projection.shape[1])
    for component in range(projection.shape[1]):
        column = projection[:, component] - projection[:, component].mean()
        denominator = np.linalg.norm(column) * log_mpp_norm
        correlation[component] = (float(column @ log_mpp_centred / denominator)
                                  if denominator > 0 else 0.0)

    return dict(mean=mean, components=right_vectors,
                variance_ratio=variance_ratio,
                correlation_with_log_mpp=correlation)


def select_components(rule: str, r: int, basis: dict,
                      rng: np.random.Generator) -> np.ndarray:
    """The r directions this rule proposes, as [r, dim] orthonormal rows."""
    components = basis['components']
    if rule == 'scale':
        order = np.argsort(-np.abs(basis['correlation_with_log_mpp']))
        return components[order[:r]]
    if rule in ('variance', 'uncentred'):
        return components[:r]
    if rule == 'random':
        gaussian = rng.standard_normal((components.shape[1], r))
        orthonormal, _ = np.linalg.qr(gaussian)
        return orthonormal.T
    raise ValueError(f'unknown selection rule {rule!r}')


def project(features: torch.Tensor, basis: dict, directions: np.ndarray,
           centre: bool = True) -> torch.Tensor:
    """Centre (or not), project, renormalise -- the renormalisation is not
    cosmetic: the thing being replaced is a cosine, so the subspace version
    has to be a cosine of the projected coordinates."""
    matrix = features.numpy().astype(np.float64)
    if centre:
        matrix = matrix - basis['mean']
    projected = torch.from_numpy(matrix @ directions.T).float()
    return torch.nn.functional.normalize(projected, dim=-1)


def knn_labels(reference: torch.Tensor, reference_mpp: np.ndarray,
              query: torch.Tensor, k: int = K_NEIGHBOURS) -> np.ndarray:
    """Per-query median of its k nearest neighbours' mpp labels. Identical in
    form to KnnClassifier.predict, stopping one median short -- that class
    returns the median over all patches of one FoV; here the grouping differs
    between callers, so it is done by them."""
    k = min(k, reference.shape[0])
    similarity = query @ reference.T
    neighbours = similarity.topk(k, dim=1).indices.numpy()
    return np.median(reference_mpp[neighbours], axis=1)


def nearest_level(mpp: np.ndarray, level_mpp_values: np.ndarray) -> np.ndarray:
    """Which level an mpp belongs to, in LOG space -- levels are
    geometrically spaced, so a linear nearest would be wrong on a 4x pyramid.
    """
    return np.argmin(np.abs(np.log(mpp)[:, None]
                            - np.log(level_mpp_values)[None, :]), axis=1)


def score(predicted_mpp: np.ndarray, true_mpp: np.ndarray,
         level_mpp_values: np.ndarray) -> dict:
    """Four numbers, and the chance rate that makes the first one readable."""
    predicted_level = nearest_level(predicted_mpp, level_mpp_values)
    true_level = nearest_level(true_mpp, level_mpp_values)

    correct = predicted_level == true_level
    relative_error = np.abs(predicted_mpp - true_mpp) / true_mpp

    wrong = ~correct
    coarse_share = (float((predicted_mpp[wrong] > true_mpp[wrong]).mean())
                    if wrong.any() else float('nan'))

    _, counts = np.unique(true_level, return_counts=True)
    return dict(n_test=int(len(predicted_mpp)),
                level_accuracy=float(correct.mean()),
                mpp_error_relative_p50=float(np.median(relative_error)),
                coarse_share_of_errors=coarse_share,
                chance_accuracy=float(counts.max() / counts.sum()))


def split_band(coords: np.ndarray, level_index: np.ndarray) -> np.ndarray:
    """Hold out the top quartile by x, WITHIN each level."""
    held_out = np.zeros(len(coords), dtype=bool)
    for level in np.unique(level_index):
        rows = np.flatnonzero(level_index == level)
        threshold = np.quantile(coords[rows, 0], 1.0 - TEST_FRACTION)
        held_out[rows[coords[rows, 0] >= threshold]] = True
    return held_out


def split_random(level_index: np.ndarray, seed: int) -> np.ndarray:
    """The leaky split, kept on purpose so the leak can be measured."""
    rng = np.random.default_rng(seed)
    held_out = np.zeros(len(level_index), dtype=bool)
    for level in np.unique(level_index):
        rows = np.flatnonzero(level_index == level)
        chosen = rng.choice(rows, size=int(round(len(rows) * TEST_FRACTION)),
                            replace=False)
        held_out[chosen] = True
    return held_out


def rebalance(keep: np.ndarray, level_index: np.ndarray,
             seed: int) -> np.ndarray:
    """Cut a filtered subset back to the same count at every level."""
    rng = np.random.default_rng(seed)
    levels = np.unique(level_index)
    take = min(int((keep & (level_index == level)).sum()) for level in levels)
    balanced = np.zeros(len(keep), dtype=bool)
    for level in levels:
        rows = np.flatnonzero(keep & (level_index == level))
        balanced[rng.choice(rows, size=take, replace=False)] = True
    return balanced


def gate_pinned(reference, reference_mpp, query) -> dict:
    """The production setting must BE production, not a re-implementation."""
    classifier = KnnClassifier(reference, reference_mpp, k=K_NEIGHBOURS)
    theirs = float(classifier.predict(query))
    ours = float(np.median(knn_labels(reference, reference_mpp, query)))
    return dict(gate='pinned', theirs=theirs, ours=ours,
                difference=abs(theirs - ours), passed=bool(theirs == ours))


def gate_full_rank(reference, reference_mpp, query, basis) -> dict:
    """`uncentred` at full rank must equal `production`, exactly -- the only
    check on the projection arithmetic itself."""
    directions = basis['components']
    theirs = float(np.median(knn_labels(reference, reference_mpp, query)))
    ours = float(np.median(knn_labels(
        project(reference, basis, directions, centre=False), reference_mpp,
        project(query, basis, directions, centre=False))))
    return dict(gate='full_rank', theirs=theirs, ours=ours,
                difference=abs(theirs - ours), passed=bool(theirs == ours))


def gate_shuffled(reference, reference_mpp, query, query_mpp,
                  level_mpp_values, real_accuracy, seed) -> dict:
    """Permuted labels must fall to chance -- scored against the real run
    rather than a fixed threshold."""
    rng = np.random.default_rng(seed)
    shuffled = reference_mpp[rng.permutation(len(reference_mpp))]
    shuffled_score = score(knn_labels(reference, shuffled, query),
                          query_mpp, level_mpp_values)
    midpoint = (real_accuracy + shuffled_score['chance_accuracy']) / 2
    return dict(gate='shuffled',
                shuffled_accuracy=shuffled_score['level_accuracy'],
                chance=shuffled_score['chance_accuracy'],
                real_accuracy=real_accuracy,
                passed=bool(shuffled_score['level_accuracy'] < midpoint))


def evaluate_settings(fit_features, fit_mpp, test_features, test_mpp,
                      level_mpp_values, basis, group_ids, seed) -> tuple:
    """Every setting on one (fit, test) pair. `group_ids` is the FoV each test
    row belongs to, or None when one row is one query."""
    rng = np.random.default_rng(seed)
    score_rows, group_rows = [], []
    groups = np.unique(group_ids) if group_ids is not None else None
    production_correct = None

    def run_once(reference, query):
        predicted = knn_labels(reference, fit_mpp, query)
        truth = test_mpp
        if groups is not None:
            predicted = np.array([np.median(predicted[group_ids == g])
                                  for g in groups])
            truth = np.array([test_mpp[group_ids == g][0] for g in groups])
        predicted_level = nearest_level(predicted, level_mpp_values)
        correct = predicted_level == nearest_level(truth, level_mpp_values)
        return (score(predicted, truth, level_mpp_values), predicted, truth,
                correct)

    def paired(correct) -> dict:
        if production_correct is None:
            return dict(wins_vs_production=0, losses_vs_production=0)
        return dict(
            wins_vs_production=int((correct & ~production_correct).sum()),
            losses_vs_production=int((~correct & production_correct).sum()))

    def emit(rule, r, reference, query):
        result, predicted, truth, correct = run_once(reference, query)
        score_rows.append(dict(rule=rule, r=r, **result, **paired(correct)))
        if groups is not None:
            predicted_level = nearest_level(predicted, level_mpp_values)
            true_level = nearest_level(truth, level_mpp_values)
            for index, group in enumerate(groups):
                group_rows.append(dict(
                    rule=rule, r=r, fov_id=int(group),
                    true_level=int(true_level[index]),
                    predicted_level=int(predicted_level[index]),
                    true_mpp=float(truth[index]),
                    predicted_mpp=float(predicted[index])))
        return correct

    production_correct = emit('production', fit_features.shape[1],
                              fit_features, test_features)

    all_directions = basis['components']
    emit('centred', all_directions.shape[0],
        project(fit_features, basis, all_directions),
        project(test_features, basis, all_directions))

    for rule, centre in RULES:
        for r in R_VALUES:
            if r > all_directions.shape[0]:
                continue
            if rule != 'random':
                directions = select_components(rule, r, basis, rng)
                emit(rule, r,
                    project(fit_features, basis, directions, centre=centre),
                    project(test_features, basis, directions, centre=centre))
                continue

            draws = []
            for _ in range(RANDOM_DRAWS):
                directions = select_components(rule, r, basis, rng)
                result, _, _, correct = run_once(
                    project(fit_features, basis, directions, centre=centre),
                    project(test_features, basis, directions, centre=centre))
                draws.append({**result, **paired(correct)})
            averaged = {key: float(np.nanmean([d[key] for d in draws]))
                       for key in draws[0]}
            averaged['n_test'] = int(draws[0]['n_test'])
            averaged['level_accuracy_std'] = float(
                np.std([d['level_accuracy'] for d in draws]))
            score_rows.append(dict(rule=rule, r=r, **averaged))

    return score_rows, group_rows


def load_query_level(key_dir, level, pooling):
    """One level's synthesised FoV tiles: features and the FoV each came from."""
    path = FeatureStore.levels(key_dir, 'query_tokens')[level]
    tensors, meta = FeatureStore.load(path, keys=['features', 'fov_id'])

    slots = pooling_kinds(tensors['features'].float(), pooling, meta)
    features = torch.nn.functional.normalize(
        slots.reshape(slots.shape[0], -1), dim=-1)
    return features, tensors['fov_id'].numpy().astype(np.int64), float(meta.mpp)


def record_selection(wsi_stem, pooling, subset_name, basis, white_fraction,
                     fit_features) -> list:
    """Which components the scale rule picked, and what else they track."""
    matrix = fit_features.numpy().astype(np.float64) - basis['mean']
    projection = matrix @ basis['components'].T
    order = np.argsort(-np.abs(basis['correlation_with_log_mpp']))

    usable = np.isfinite(white_fraction)
    rows = []
    for rank, component in enumerate(order[:max(R_VALUES)]):
        column = projection[:, component]
        corr_white = (float(np.corrcoef(column[usable],
                                        white_fraction[usable])[0, 1])
                     if usable.sum() > 2 else float('nan'))
        rows.append(dict(wsi_stem=wsi_stem, pooling=pooling,
                         subset=subset_name, scale_rank=rank + 1,
                         component=int(component) + 1,
                         corr_log_mpp=float(
                             basis['correlation_with_log_mpp'][component]),
                         corr_white=corr_white,
                         variance_ratio=float(
                             basis['variance_ratio'][component])))
    return rows


def report(prefix: str, rows: list) -> None:
    def find(rule, r=None):
        for row in rows:
            if row['rule'] == rule and (r is None or row['r'] == r):
                return row
        return None

    parts = []
    for rule in ('production', 'centred'):
        row = find(rule)
        if row:
            parts.append(f'{rule} {row["level_accuracy"]:.3f}')
    for rule in ('uncentred', 'variance', 'scale', 'random'):
        got = [(r, find(rule, r)) for r in R_REPORT_POINTS]
        got = [(r, row) for r, row in got if row]
        if got:
            scores = '/'.join(f'{row["level_accuracy"]:.3f}' for _, row in got)
            at = '/'.join(str(r) for r, _ in got)
            parts.append(f'{rule}@{at} {scores}')
    print(f'  {prefix}  ' + '   '.join(parts), flush=True)


def run_arm_a(features, mpp_labels, level_index, coords, white_fraction,
             level_mpp_values, wsi_stem, subset_name, args) -> tuple:
    score_rows, selected_rows, gate_rows = [], [], []
    for split_name, held_out in (
            ('band', split_band(coords, level_index)),
            ('random', split_random(level_index, args.seed))):
        fit_features, test_features = features[~held_out], features[held_out]
        fit_mpp, test_mpp = mpp_labels[~held_out], mpp_labels[held_out]

        basis = fit_basis(fit_features, fit_mpp)
        rows, _ = evaluate_settings(fit_features, fit_mpp, test_features,
                                    test_mpp, level_mpp_values, basis,
                                    group_ids=None, seed=args.seed)
        for row in rows:
            row.update(wsi_stem=wsi_stem, pooling=args.pooling, arm='A',
                      split=split_name, subset=subset_name)
        score_rows.extend(rows)

        if split_name == 'band':
            selected_rows.extend(record_selection(
                wsi_stem, args.pooling, subset_name, basis,
                white_fraction[~held_out], fit_features))
            baseline = next(r for r in rows if r['rule'] == 'production')
            gate_rows.append(dict(wsi_stem=wsi_stem, subset=subset_name,
                                  **gate_pinned(fit_features, fit_mpp,
                                               test_features)))
            gate_rows.append(dict(wsi_stem=wsi_stem, subset=subset_name,
                                  **gate_full_rank(fit_features, fit_mpp,
                                                  test_features, basis)))
            gate_rows.append(dict(wsi_stem=wsi_stem, subset=subset_name,
                                  **gate_shuffled(fit_features, fit_mpp,
                                                 test_features, test_mpp,
                                                 level_mpp_values,
                                                 baseline['level_accuracy'],
                                                 args.seed)))
        report(f'arm A  {subset_name:14s} {split_name:6s}', rows)
    return score_rows, selected_rows, gate_rows


def run_arm_b(key_dir, wsi_stem, args, data, features, mpp_labels,
             level_mpp_values, subset_name) -> tuple:
    """Reference is the (possibly filtered) bank; queries are the FoV
    renders. The background filter reaches this only through `features` --
    the reference side, which is where the confound lives."""
    query_blocks, group_blocks, query_mpp_blocks = [], [], []
    group_offset = 0
    for level in data['levels']:
        block_features, fov_id, level_mpp = load_query_level(
            key_dir, level, args.pooling)
        query_blocks.append(block_features)
        group_blocks.append(fov_id + group_offset)
        group_offset += int(fov_id.max()) + 1
        query_mpp_blocks.append(np.full(len(fov_id), level_mpp))

    query_features = torch.cat(query_blocks, dim=0)
    group_ids = np.concatenate(group_blocks)
    query_mpp = np.concatenate(query_mpp_blocks)

    basis = fit_basis(features, mpp_labels)
    rows, group_rows = evaluate_settings(features, mpp_labels, query_features,
                                        query_mpp, level_mpp_values, basis,
                                        group_ids=group_ids, seed=args.seed)
    for row in rows:
        row.update(wsi_stem=wsi_stem, pooling=args.pooling, arm='B',
                  split='none', subset=subset_name)
    for row in group_rows:
        row.update(wsi_stem=wsi_stem, pooling=args.pooling, arm='B',
                  subset=subset_name)
    report(f'arm B  {subset_name:14s} {len(np.unique(group_ids))} FoV', rows)
    return rows, group_rows


def analyse_slide_subspace(key_dir, wsi_stem, args) -> tuple:
    print(f'\n{"=" * 78}\n{wsi_stem}   pooling {args.pooling}\n{"=" * 78}',
          flush=True)

    data = load_slide_balanced(key_dir, args.pooling, args.per_level, seed=args.seed)
    features = data['features']
    level_index = data['level_index']
    level_mpp_values = np.array([data['level_mpp'][i]
                                for i in sorted(data['level_mpp'])])
    mpp_labels = level_mpp_values[level_index]
    print(f'  {features.shape[0]} tiles, {len(level_mpp_values)} levels, '
          f'{data["per_level"]} per level, dim {features.shape[1]}', flush=True)

    subsets = [('all', np.ones(len(features), dtype=bool))]
    if args.white_max is not None:
        low_background = data['white_fraction'] < args.white_max
        counts = {int(level): int((low_background & (level_index == level)).sum())
                 for level in np.unique(level_index)}
        print(f'  background < {args.white_max}: {counts}', flush=True)
        if min(counts.values()) < args.min_subset:
            print(f'    -> skipped: a level has under {args.min_subset} tiles',
                  flush=True)
        else:
            balanced = rebalance(low_background, level_index, args.seed)
            print(f'    -> rebalanced to {int(balanced.sum()) // len(counts)} '
                  f'per level', flush=True)
            subsets.append((f'white<{args.white_max}', balanced))

    score_rows, group_rows, selected_rows, gate_rows = [], [], [], []
    for subset_name, subset_mask in subsets:
        rows = np.flatnonzero(subset_mask)
        arm_a = run_arm_a(features[rows], mpp_labels[rows], level_index[rows],
                          data['coords'][rows], data['white_fraction'][rows],
                          level_mpp_values, wsi_stem, subset_name, args)
        score_rows.extend(arm_a[0])
        selected_rows.extend(arm_a[1])
        gate_rows.extend(arm_a[2])

        if not args.skip_arm_b:
            try:
                arm_b = run_arm_b(key_dir, wsi_stem, args, data,
                                 features[rows], mpp_labels[rows],
                                 level_mpp_values, subset_name)
                score_rows.extend(arm_b[0])
                group_rows.extend(arm_b[1])
            except Exception as exc:                        # noqa: BLE001
                print(f'  arm B  UNAVAILABLE: {type(exc).__name__}: {exc}',
                      flush=True)

    return score_rows, group_rows, selected_rows, gate_rows


CURVE_STYLE = (('variance', '--s', 'tab:orange'),
              ('uncentred', '-D', 'tab:green'),
              ('scale', '-o', 'tab:blue'),
              ('random', ':^', 'tab:grey'))

FIGURE_CAPTION = (
    'production = cosine on the raw 1536-D features, no projection, no mean '
    'removed (what KnnClassifier does today).    centred = all components '
    'kept, mean removed.\n'
    'variance = top-r components by eigenvalue, mean removed.    uncentred = '
    'the SAME directions, projected without removing the mean.\n'
    'scale = top-r by |corr(component, log mpp)|, mean removed.    random = a '
    f'uniformly random r-dim subspace, mean of {RANDOM_DRAWS} draws (band = '
    'std).\n'
    'chance = the largest class share in the test set.    r = how many '
    'dimensions are kept, not "the r-th component".')


def plot_subspace_accuracy(score_rows, wsi_stem, path) -> None:
    panels = [('A', 'band', 'arm A: tile to tile'),
             ('B', 'none', 'arm B: photo to tile')]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)

    drew = False
    for axis, (arm, split, title) in zip(axes, panels):
        rows = [r for r in score_rows
               if r['wsi_stem'] == wsi_stem and r['arm'] == arm
               and r['split'] == split and r['subset'] == 'all']
        if not rows:
            axis.axis('off')
            continue
        drew = True
        for rule, style, colour in CURVE_STYLE:
            points = sorted((r['r'], r['level_accuracy'], r.get(
                'level_accuracy_std', 0.0) or 0.0)
                for r in rows if r['rule'] == rule)
            if not points:
                continue
            x = [p[0] for p in points]
            y = np.array([p[1] for p in points])
            spread = np.array([p[2] for p in points])
            axis.plot(x, y, style, color=colour, label=rule, markersize=4)
            if spread.any():
                axis.fill_between(x, y - spread, y + spread, color=colour,
                                  alpha=0.2, linewidth=0)
        for rule, colour in (('production', 'k'), ('centred', 'tab:red')):
            match = [r for r in rows if r['rule'] == rule]
            if match:
                axis.axhline(match[0]['level_accuracy'], color=colour,
                            linewidth=1.2, label=rule)
        axis.axhline(rows[0]['chance_accuracy'], color='grey', linewidth=1,
                    linestyle='--', label='chance')
        axis.set_xscale('log')
        axis.set_xlabel('r (dimensions kept)')
        axis.set_title(title, fontsize=10)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
    if not drew:
        plt.close(figure)
        return

    axes[0].set_ylabel('level accuracy\n(predicted level == true level)')
    axes[1].legend(fontsize=8, loc='lower right')
    figure.suptitle(f'{wsi_stem} -- does the cut survive without centring?',
                   fontsize=11)
    figure.tight_layout(rect=(0, 0.20, 1, 0.96))
    figure.text(0.01, 0.005, FIGURE_CAPTION, fontsize=7.2, va='bottom',
               family='monospace', linespacing=1.5)
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(f'  {path}')


def plot_subspace_confusion(group_rows, wsi_stem, path) -> None:
    rows = [r for r in group_rows
           if r['wsi_stem'] == wsi_stem and r['subset'] == 'all']
    if not rows:
        return
    wanted = ([('production', None), ('centred', None)]
             + [('uncentred', r) for r in R_REPORT_POINTS])
    present = [(rule, r) for rule, r in wanted
              if any(x['rule'] == rule and (r is None or x['r'] == r)
                    for x in rows)]
    if not present:
        return

    n_levels = max(max(r['true_level'], r['predicted_level'])
                  for r in rows) + 1
    figure, axes = plt.subplots(1, len(present),
                                figsize=(3.6 * len(present) + 1.2, 3.9),
                                squeeze=False)
    for axis, (rule, r) in zip(axes[0], present):
        matrix = np.zeros((n_levels, n_levels), dtype=int)
        for row in rows:
            if row['rule'] != rule or (r is not None and row['r'] != r):
                continue
            matrix[row['true_level'], row['predicted_level']] += 1
        axis.imshow(matrix, cmap='Blues', vmin=0)
        for i in range(n_levels):
            for j in range(n_levels):
                axis.text(j, i, str(matrix[i, j]), ha='center', va='center',
                         fontsize=9,
                         color='white' if matrix[i, j] > matrix.max() * 0.6
                         else 'black')
        axis.set_xticks(range(n_levels))
        axis.set_yticks(range(n_levels))
        axis.set_xlabel('predicted level')
        axis.set_ylabel('true level')
        label = rule if r is None else f'{rule}, r={r}'
        axis.set_title(f'{label}   (n={matrix.sum()} FoV)', fontsize=10)
    figure.suptitle(f'{wsi_stem} -- arm B: where the errors go', fontsize=11)
    figure.tight_layout(rect=(0, 0.12, 1, 0.94))
    figure.text(0.01, 0.01,
               'Counts are FoVs, not tiles. Level index 0 is the finest.\n'
               'Mass in a single column = collapse to one answer; a diagonal '
               'leaking to one side = a bias with a direction.',
               fontsize=7.5, va='bottom', family='monospace',
               linespacing=1.5)
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(f'  {path}')


def plot_subspace_versus_baseline(score_rows, wsi_stem, path) -> None:
    settings = [('centred', None)]
    for rule in ('uncentred', 'variance', 'scale', 'random'):
        settings += [(rule, r) for r in R_REPORT_POINTS]

    arms = [('A', 'band', 'arm A: tile to tile'),
           ('B', 'none', 'arm B: photo to tile')]
    subsets = sorted({row['subset'] for row in score_rows
                     if row['wsi_stem'] == wsi_stem})
    colours = {name: colour for name, colour in
              zip(subsets, ('tab:blue', 'tab:orange', 'tab:green'))}

    figure, axes = plt.subplots(1, len(arms),
                                figsize=(7.0 * len(arms), 5.2), sharey=True)
    drew = False
    for axis, (arm, split, title) in zip(np.atleast_1d(axes), arms):
        for offset, subset in enumerate(subsets):
            rows = [r for r in score_rows
                   if r['wsi_stem'] == wsi_stem and r['arm'] == arm
                   and r['split'] == split and r['subset'] == subset]
            baseline = next((r for r in rows if r['rule'] == 'production'),
                           None)
            if baseline is None:
                continue
            drew = True
            shift = (offset - (len(subsets) - 1) / 2) * 0.3
            for position, (rule, r) in enumerate(settings):
                match = next((x for x in rows if x['rule'] == rule
                            and (r is None or x['r'] == r)), None)
                if match is None:
                    continue
                delta = (match['level_accuracy']
                        - baseline['level_accuracy']) * 100
                y = len(settings) - 1 - position + shift
                axis.plot(delta, y, 'o', color=colours[subset], markersize=7)
                axis.annotate(
                    f'+{match.get("wins_vs_production", 0):.0f}'
                    f'/-{match.get("losses_vs_production", 0):.0f}',
                    (delta, y), textcoords='offset points', xytext=(9, 0),
                    fontsize=7, va='center', color=colours[subset])
        axis.axvline(0, color='k', linewidth=1.4)
        axis.set_yticks(range(len(settings)))
        axis.set_yticklabels(
            [f'{rule}@{r}' if r else rule
            for rule, r in reversed(settings)], fontsize=9)
        axis.set_xlabel('level accuracy - production (points)\n'
                       'right of 0 = better than what stage 1 runs today')
        axis.set_title(title, fontsize=10)
        axis.grid(axis='x', alpha=0.25)
    if not drew:
        plt.close(figure)
        return

    handles = [plt.Line2D([], [], marker='o', linestyle='', color=colours[s],
                         label=f'reference bank: {s}') for s in subsets]
    np.atleast_1d(axes)[-1].legend(handles=handles, fontsize=8,
                                  loc='lower right')
    figure.suptitle(f'{wsi_stem} -- which recipe beats the baseline?',
                   fontsize=11)
    figure.tight_layout(rect=(0, 0.10, 1, 0.94))
    figure.text(0.01, 0.01,
               '0 = production (cosine on the raw 1536-D features, nothing '
               'projected, no mean removed).\n'
               '+w/-l = paired counts on the same test items.',
               fontsize=7.2, va='bottom', family='monospace', linespacing=1.5)
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(f'  {path}')


def run_subspace_knn(args, out_dir: Path) -> int:
    if args.white_max is not None and args.white_max < 0:
        args.white_max = None

    all_scores, all_groups, all_selected, all_gates, failed = [], [], [], [], []
    for slide in args.wsi_stem:
        try:
            scores, groups, selected, gates = analyse_slide_subspace(
                _draw_dir(args, slide), slide, args)
            all_scores.extend(scores)
            all_groups.extend(groups)
            all_selected.extend(selected)
            all_gates.extend(gates)
        except Exception as exc:                            # noqa: BLE001
            print(f'\n{slide}: {type(exc).__name__}: {exc}')
            traceback.print_exc()
            failed.append(slide)

    write_csv(all_scores, out_dir / 'subspace_knn_scores.csv')
    write_csv(all_groups, out_dir / 'subspace_knn_arm_b_fovs.csv')
    write_csv(all_selected, out_dir / 'subspace_knn_selected.csv')
    write_csv(all_gates, out_dir / 'subspace_knn_gates.csv')
    write_csv([dict(term=term, means=means) for term, means in DEFINITIONS],
             out_dir / 'subspace_knn_definitions.csv')
    for slide in {row['wsi_stem'] for row in all_scores}:
        stem = slide.replace(',', '_')
        plot_subspace_accuracy(all_scores, slide,
                              out_dir / f'subspace_knn_accuracy__{stem}.png')
        plot_subspace_confusion(all_groups, slide,
                                out_dir / f'subspace_knn_confusion__{stem}.png')
        plot_subspace_versus_baseline(
            all_scores, slide, out_dir / f'subspace_knn_vs_baseline__{stem}.png')

    failed_gates = [g for g in all_gates if not g['passed']]
    if failed_gates:
        print(f'\n{len(failed_gates)} GATE FAILURE(S) -- the scores above do '
             f'not mean what they say:')
        for gate in failed_gates:
            print(f'  {gate["wsi_stem"]}  {gate["gate"]}  {gate}')
    if failed:
        print(f'\n{len(failed)} slide(s) failed: {", ".join(failed)}')
    return 1 if (failed or failed_gates) else 0


# ══════════════════════════════════════════════════════════════════════════════
#  PART "sampler_routing"  --  NEW, 2026-09-14
# ══════════════════════════════════════════════════════════════════════════════

#: (rule, r) settings run against the baseline in `sampler_routing` -- the
#: two points every subspace_knn figure already reports at (R_REPORT_POINTS),
#: for both candidates that setting isolates (variance = the dimension cut
#: alone; uncentred = the same cut without mean removal, the one arm B's own
#: verdict actually turns on).
SAMPLER_ROUTING_CANDIDATES = tuple(
    (rule, r) for rule in ('variance', 'uncentred') for r in R_REPORT_POINTS)


def sample_reference_and_query_positions(wsi, mask, tile_size, rungs,
                                         n_ref_per_rung, n_query_per_rung,
                                         seed):
    """The SHAPE of SuperPoint's stageA recipe
    (`training/SuperPathPoint/common/Corpora.RECIPES['stageA']`): `DsLadder`
    rungs, disjoint lattice, PLAIN DEFAULT `RichnessConfig()` (bg85_95/bg95_100
    capped at 0 -- a background-heavy tile carries no scale information and
    only adds a wrong neighbour), no chains. Not its n, and not its corpus:
    `sampler_routing` draws its own and writes no pre-tiles.

    One draw of `n_ref_per_rung + n_query_per_rung` positions per rung; the
    caller splits it per rung, which is what guarantees reference and query
    never share a position -- a query whose twin is in the reference bank is
    found by identity, not by scale. Reference tiles are read straight off this
    sampler (real grid pixels); query POSITIONS only are read off it, then
    rendered through `SlideReader.read` + `simulate_microscope_photo` instead --
    see `run_sampler_routing` -- so reference and query differ by more than
    which grid cell they happened to land on.
    """
    ladder = DsLadder(rungs=tuple(sorted(rungs)))
    plans = ladder.plan_for(wsi, tile_size)
    n_per_rung = n_ref_per_rung + n_query_per_rung
    max_tries = max(1, 2500 // max(n_per_rung, 1))     # 5x n, extract_pretiles.py's own ratio
    cfg = SamplerConfig(n_per_rung=n_per_rung, seed=seed,
                        max_tries_per_tile=max_tries,
                        richness=RichnessConfig(), overlap=OverlapConfig())
    sampler = TileSampler(wsi, mask, cfg)
    sampler.sample(plans)
    return sampler


def _score_table(predicted, true_mpp, true_ds, level_mpp_values, rungs) -> dict:
    """`{ds: score_dict}` for every rung that has at least one query, plus
    `'total'` = score over everything pooled -- so a bad rung and a good one
    cannot average into something unremarkable the way one overall number
    would."""
    table = {}
    for ds in rungs:
        member = np.isclose(true_ds, ds)
        if not member.any():
            continue
        table[ds] = score(predicted[member], true_mpp[member], level_mpp_values)
    table['total'] = score(predicted, true_mpp, level_mpp_values)
    return table


def _vote_distribution(predicted, true_ds, level_mpp_values, rungs) -> dict:
    """For each true rung, which reference LEVEL the k-NN vote landed on, as
    counts -- "who does this rung's queries tend to vote for". This is
    `log/TODO.log`'s own long-standing '第 3 項：stage 1 的 patch 票數診斷'
    note, finally wired up: a near-even split says the median is unstable
    (fix: break ties toward the finer label); a lopsided vote onto one wrong
    level says the FEATURE is distorted at that scale (fix: the encoder,
    not the classifier).
    """
    predicted_level = nearest_level(predicted, level_mpp_values)
    out = {}
    for ds in rungs:
        member = np.isclose(true_ds, ds)
        if not member.any():
            continue
        levels, counts = np.unique(predicted_level[member], return_counts=True)
        out[ds] = {float(level_mpp_values[int(lv)]): int(c)
                  for lv, c in zip(levels, counts)}
    return out


def _print_method_report(method: str, setting: str, table: dict, votes: dict,
                         rungs) -> None:
    print(f'\nMethod: {method}   Setting: {setting}')
    for ds in list(rungs) + ['total']:
        if ds not in table:
            continue
        row = table[ds]
        label = f'ds {ds:g}' if ds != 'total' else 'total'
        print(f'  {label}')
        print(f'      level_acc {row["level_accuracy"]:.3f}   '
             f'mpp_err_p50 {row["mpp_error_relative_p50"]:.3f}   '
             f'coarse_share {row["coarse_share_of_errors"]:.3f}   '
             f'(n={row["n_test"]})')
        if ds != 'total' and ds in votes:
            vote_str = ', '.join(f'{mpp:.3f}um:{c}'
                                for mpp, c in sorted(votes[ds].items()))
            print(f'      votes went to -> {vote_str}')


def run_sampler_routing(args, out_dir: Path) -> int:
    from GigaPathFunc import GigaPathEncoderConfig                   # noqa: PLC0415

    entry = locate(args.wsi_name)
    wsi = SafeSlide(entry.path)
    # --seg's recipe, as in bench_stage1_mpp. This used to be hsv at the old
    # default ds 32, built here and recorded nowhere.
    mask = MASK_RECIPES[args.seg].build(wsi, args.device)
    encoder = GigaPathEncoderConfig(batch_size=args.batch_size)\
        .with_model(dtype='fp32').build(args.device)

    rungs = tuple(sorted(args.rungs))
    sampler = sample_reference_and_query_positions(
        wsi, mask, args.tile, rungs, args.sampler_n_per_rung,
        args.sampler_query_per_rung, args.seed)
    ds_of = np.array([float(s.meta.ds) for s in sampler])
    print(f'[sampler_routing]  {entry.name}  rungs {rungs}  '
         f'{len(sampler)} fresh positions  '
         f'({args.sampler_n_per_rung} ref + {args.sampler_query_per_rung} '
         f'query per rung, uncached)', flush=True)

    rng = np.random.default_rng(args.seed)
    ref_mask = np.zeros(len(sampler), dtype=bool)
    for ds in np.unique(ds_of):
        rows = np.flatnonzero(ds_of == ds)
        take = min(args.sampler_n_per_rung, len(rows))
        ref_mask[rng.choice(rows, size=take, replace=False)] = True
    query_mask = ~ref_mask     # disjoint by construction -- one draw, split
    print(f'  {int(ref_mask.sum())} reference tiles (real grid pixels) / '
         f'{int(query_mask.sum())} query positions (rendered as photos), '
         f'split per rung', flush=True)

    # ── reference: real grid pixels, encoded directly ──────────────────────
    ref_indices = np.flatnonzero(ref_mask)
    samples = list(sampler)
    ref_images = SlideReader(wsi, resize='area').read_samples(
        [samples[i] for i in ref_indices], ReadSpec(args.tile, args.tile))
    # numpy from here on (the SVD, the KNN labels): moved to the host once
    ref_features = encoder(ref_images).cpu()
    ref_ds = ds_of[ref_indices]
    ref_mpp = wsi.base_mpp * ref_ds
    level_mpp_values = np.array(sorted({float(m) for m in ref_mpp}))

    # ── query: SAME positions, rendered as a photo would be, not read as a
    #    plain grid tile -- SlideReader.read + simulate_microscope_photo is the
    #    exact pair test_gigapath_knn_esti_mpp.py's own load_query() uses.
    #    MPixels=1.475 matches CLAUDE.md's own real-photo spec (1440x1024,
    #    45:32) -- the query is sized like the real pipeline's photos, not
    #    query_sim's 4:3/12MP default.
    #
    #    THE PHOTO IS SPLIT INTO tile_size PATCHES BEFORE ENCODING, exactly
    #    like production's `build_query_features` (QueryPatchContainer +
    #    KnnClassifier's own median-of-medians) -- a 1440x1024 photo resized
    #    whole into the encoder's fixed input is a SECOND, silent
    #    downsampling stacked on top of the requested mpp: its own footprint
    #    is ~5.6x a 256px tile's at the same mpp, so whole-image encoding
    #    quietly shifts every query about two DsLadder rungs coarser before
    #    the KNN ever votes -- which is exactly the 4x/`mpp_err_p50==3.000`
    #    pattern that made every rung score 0.000. Encoding per patch keeps
    #    the query at the SAME effective scale as the 256px reference tiles.
    #    Every candidate below (baseline + every SubspaceKnn setting)
    #    votes on these same per-query patch features -- the fix is not
    #    baseline-only. ──
    query_indices = np.flatnonzero(query_mask)
    query_patch_feats: list[torch.Tensor] = []   # [M_i, D] per surviving query
    query_ds = []
    photo_reader = SlideReader(wsi)                   # lanczos: the photo's filter
    photo_spec = ReadSpec(*sensor_size('4:3', args.mpixels))
    for i in query_indices:
        meta = sampler[i].meta
        ds = float(meta.ds)
        image = photo_reader.read(int(meta.x), int(meta.y), photo_spec, ds)
        if image is None:
            continue
        # simulate_microscope_photo returns an ndarray (its own docstring:
        # "backward-compat entry point"); QueryPatchContainer takes either,
        # but wrapped as PIL so this stays the same type the reference side
        # reads (SlideReader.read_samples).
        photo = Image.fromarray(simulate_microscope_photo(image))
        qc = QueryPatchContainer(photo)
        qc.extract_all(args.tile, overlap=True)
        qfm = qc.to_features(encoder)
        query_patch_feats.append(torch.stack(list(qfm.iter_main_features())).cpu())
        query_ds.append(ds)
    if len(query_patch_feats) < query_mask.sum():
        print(f'  {int(query_mask.sum()) - len(query_patch_feats)} query crop(s) '
             f'ran off the slide and were dropped', flush=True)
    query_ds = np.asarray(query_ds, np.float64)
    query_mpp = wsi.base_mpp * query_ds

    basis = fit_basis(ref_features, ref_mpp)
    all_rows = []

    def run_method(method: str, setting: str, predicted: np.ndarray) -> None:
        table = _score_table(predicted, query_mpp, query_ds, level_mpp_values,
                             rungs)
        votes = _vote_distribution(predicted, query_ds, level_mpp_values, rungs)
        _print_method_report(method, setting, table, votes, rungs)
        for ds, row in table.items():
            all_rows.append(dict(method=method, setting=setting, ds=ds,
                                 votes=votes.get(ds, ''), **row))

    def patch_vote(ref_feats: torch.Tensor, ref_labels: np.ndarray,
                  patch_feats_per_query: list[torch.Tensor]) -> np.ndarray:
        '''Per query: median of each patch's k-NN median -- KnnClassifier's
        own two-stage vote, done here on a python list because each query
        contributes a different patch count.'''
        return np.array([
            float(np.median(knn_labels(ref_feats, ref_labels, feats, k=args.k)))
            for feats in patch_feats_per_query
        ])

    predicted_baseline = patch_vote(ref_features, ref_mpp, query_patch_feats)
    run_method('KnnEstMpp (baseline)',
              f'KNN (k={args.k}) reference patches={args.sampler_n_per_rung}/rung',
              predicted_baseline)

    for rule, r in SAMPLER_ROUTING_CANDIDATES:
        r = min(r, basis['components'].shape[0])
        directions = select_components(rule, r, basis, rng)
        centre = rule != 'uncentred'
        proj_ref = project(ref_features, basis, directions, centre=centre)
        proj_query_feats = [project(feats, basis, directions, centre=centre)
                           for feats in query_patch_feats]
        predicted = patch_vote(proj_ref, ref_mpp, proj_query_feats)
        run_method(f'SubspaceKnn[{rule}@{r}]',
                  f'KNN (k={args.k}) subspace={rule}@{r} reference '
                  f'patches={args.sampler_n_per_rung}/rung',
                  predicted)

    write_csv(all_rows, out_dir / 'sampler_routing_scores.csv')
    return 0


# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--parts', nargs='+',
                        choices=['axes', 'subspace_knn', 'sampler_routing',
                                'all'],
                        default=['all'])
    parser.add_argument('wsi_stem', nargs='*',
                        help='slides for axes/subspace_knn, e.g. BRACS_1228 '
                             '(unused by sampler_routing, which takes '
                             '--wsi-name instead -- it reads a live WSI, not '
                             'a cached store)')

    # ── shared / axes+subspace_knn (cached FeatureStore reads) ──────────────
    parser.add_argument('--stores', default=None,
                        help="axes/subspace_knn: one encoder's feature root, "
                             'result/cache/<job>_features/<encoder>/ (with '
                             '--seg and --draw it names one key directory '
                             'per slide)')
    parser.add_argument('--draw', default=None,
                        help='axes/subspace_knn: the draw to read, '
                             '<sampler_id>_<plan> -- printed by the dump')
    parser.add_argument('--pooling', default='cls')
    parser.add_argument('--per-level', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)

    # ── axes only ─────────────────────────────────────────────────────────
    parser.add_argument('--null-repeats', type=int, default=5)
    parser.add_argument('--decoy-repeats', type=int, default=10)
    parser.add_argument('--extreme-components', type=int, default=10)
    parser.add_argument('--extreme-tiles', type=int, default=10)

    # ── subspace_knn only ────────────────────────────────────────────────
    parser.add_argument('--white-max', type=float, default=0.15,
                        help='negative to skip the background-matched subset')
    parser.add_argument('--min-subset', type=int, default=100)
    parser.add_argument('--skip-arm-b', action='store_true')

    # ── sampler_routing only ────────────────────────────────────────────
    parser.add_argument('--wsi-name', default=None,
                        help='sampler_routing: resolved via AccessDatasets')
    parser.add_argument('--tile', type=int, default=256)
    parser.add_argument('--rungs', type=float, nargs='+',
                        default=list(DEFAULT_RUNGS),
                        help='sampler_routing: DsLadder rungs to sample -- '
                             "SuperPoint stageA's own ladder by default "
                             '(1, 2, 4, 8, 16, 32)')
    parser.add_argument('--sampler-n-per-rung', type=int, default=100,
                        help='sampler_routing: REFERENCE tiles per rung -- '
                             "SuperPoint stageA's own n (extract_pretiles.py's "
                             "_RECIPES['stageA']), not KnnEstMpp's "
                             'own default of 40')
    parser.add_argument('--sampler-query-per-rung', type=int, default=20,
                        help='sampler_routing: held-out query positions per '
                             'rung, DISJOINT from the reference draw -- '
                             'rendered as a photo (SlideReader.read + '
                             'simulate_microscope_photo), not read as a '
                             'plain grid tile')
    parser.add_argument('--mpixels', type=float, default=1.475,
                        help='sampler_routing: query size -- '
                             '1.475 MPixels at 45:32 matches CLAUDE.md\'s '
                             'real-photo spec (1440x1024), not query_sim\'s '
                             '4:3/12MP default')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    # --seg / --mask-cache-job / --sampler-cache-job: sampler_routing reads --seg.
    add_cache_args(parser)

    parser.add_argument(
        '--out', default=None,
        help='output directory, used verbatim for every part. Default: '
             'result/<SLURM_JOB_NAME or MppRoutingExp>/<encoder>/ (tagged by '
             '--stores, the one encoder the parts read)')
    args = parser.parse_args()

    parts = (['axes', 'subspace_knn', 'sampler_routing']
            if 'all' in args.parts else args.parts)
    if {'axes', 'subspace_knn'} & set(parts) and not args.wsi_stem:
        parser.error('axes/subspace_knn need at least one wsi_stem')
    if 'sampler_routing' in parts and not args.wsi_name:
        parser.error('sampler_routing needs --wsi-name')

    # CLAUDE.md's rule for a bench with no encoder of its own: take the tag
    # from the store root it reads.
    encoder_tag = Path(args.stores).name if args.stores else 'gigapath'
    out_dir = Path(args.out or job_result_dir('MppRoutingExp',
                                              encoder=encoder_tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    status = 0
    if 'axes' in parts:
        print('\n======== [axes] ========', flush=True)
        status = max(status, run_axes(args, out_dir))
    if 'subspace_knn' in parts:
        print('\n======== [subspace_knn] ========', flush=True)
        status = max(status, run_subspace_knn(args, out_dir))
    if 'sampler_routing' in parts:
        print('\n======== [sampler_routing] ========', flush=True)
        status = max(status, run_sampler_routing(args, out_dir))
    return status


if __name__ == '__main__':
    sys.exit(main())
