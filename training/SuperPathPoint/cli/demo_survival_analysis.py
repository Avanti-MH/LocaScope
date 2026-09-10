#!/usr/bin/env python3
"""Two independent demos, selected with `--parts` (both run by default).
spec.md 3.2, plan.md 2.1②.

    python training/SuperPathPoint/cli/demo_survival_analysis.py
    python training/SuperPathPoint/cli/demo_survival_analysis.py --parts chains_stack
    python training/SuperPathPoint/cli/demo_survival_analysis.py --parts merge_grid
    python training/SuperPathPoint/cli/demo_survival_analysis.py --parts merge_grid \
        --checkpoint <.pt> --wsi-name BRACS_1228 --real-flow

Merged 2026-09-11 from two standalone scripts (`demo_chains_stack.py`,
`demo_merge_grid.py`) that had grown into one jobscript family each -- they
share `--wsi-name`/`--tile`/`--c-rungs`/`--cache-root`/`--out` and nothing
else, so this file keeps them as two independent functions
(`_run_chains_stack`, `_run_merge_grid`) dispatched from one `main()`, never
forcing one part's logic through the other's.

`--out` is now a DIRECTORY shared by both parts (`result/<job>/`, default
`job_result_dir('DemoSurvivalAnalysis')`) rather than each part resolving
its own default -- `chains_stack` writes under `<out>/figures/`, exactly as
before; `merge_grid` writes `<out>/merge_grid_demo*.png`, exactly as before.
Neither part's own filenames changed, only where the shared root comes from.

=====================================================================
PART "chains_stack" -- smoke-test all three axes' OWN and REUSE-F paths
against a REAL slide (formerly `demo_chains_stack.py`)
=====================================================================
`test_chain_stack.py` (2.1①) proved the geometry is correct against synthetic
coordinates, and `own`'s wiring against a fake `PreTileStore` fixture. What
neither can catch: a wrong pyramid LEVEL, a transposed axis, an off-by-one in
`_read_wsi_tile`'s level resolution -- anything that still produces a
plausible `tile x tile` RGB array. Those only show up against real tissue,
which is what this part runs.

FIVE PATHS, NOT TWO
    F           `FStack.from_own`                     -- own, F's only source
    R  own      `RStack.from_own`                     -- stageA, independent
    R  reuse-F  `RStack.derive(chain, source='F')`     -- same chain as F's own
    C  own      `CStack.from_own`                      -- stageB-cOwn, independent
    C  reuse-F  `CStack.from_mother` + `FStack.read`   -- same chain's ds-16 tile

`own` and `reuse-F` are NOT the same physical point for R/C: `own`'s centre is
wherever its own independent draw landed; `reuse-F`'s centre is F's own
chain's centre. Each is checked and PLOTTED on its own, never overlaid.

ONE QUANTITATIVE CHECK, ON BOTH C TREES: crop the mother's own image down to
where a real child tile sits, and correlate that crop against the child's own
independent read -- against a DECOY (the SAME crop vs a DIFFERENT child), the
way every criterion in this project is a margin over a decoy and not a bare
threshold.

`sampler_id` FOR EACH OWN CALL, COMPUTED THE SAME WAY `prepare_chain_stack.py`
DOES -- imported from it directly, not re-derived here.

=====================================================================
PART "merge_grid" -- three implementations of `_merge_within_radius`
compared (formerly `demo_merge_grid.py`)
=====================================================================
ADOPTED. `A` (grid hash) is what `SurvivalProcess._merge_within_radius` runs
now -- the evidence this part produced (identical kept-index sets on
synthetic data; 131x faster on a real C-tree's raw ds=1 detections, 34503
points; 0% anchor mismatch and bit-identical aggregated curves deployed
end-to-end in `cli/survival_alpha_analysis.py`'s own C-axis flow, 1.5x
faster overall) is what cleared it. `B`, called here through
`AlphaCalibration.merge_anchors`, now ALSO runs the grid version underneath
(it calls the same module attribute) -- so `B` and `A` will always agree
going forward; the two are kept as SEPARATE, independently-written copies
specifically so a future edit to one that silently diverges from the other
shows up as a mismatch here, not as a downstream number nobody double-checks.
`original` remains the one genuinely different, genuinely slower baseline.

No model, no GPU, no slide by default -- synthetic keypoints. `--checkpoint`
(+ `--wsi-name`) adds a THIRD comparison on REAL detections: one C-tree's
actual keypoints at one rung, read the same way
`cli/survival_alpha_analysis.py`'s C axis does. `--real-flow` adds a FOURTH:
each implementation deployed (monkey-patched) into
`survival_alpha_analysis.py`'s actual C-axis flow -- anchors, probe,
`alpha_curve`, `aggregate_curves` -- so the comparison is not just "does the
merge primitive agree in isolation" but "does swapping it change one number
the real pipeline reports."

FOUR COMPARISONS, because they answer different questions -- every one
reports a QUANTITATIVE mismatch (a count/fraction of points, or a max abs
diff on aggregated numbers), never just a boolean "agree":
    the FIGURE (`--n-clusters` etc.)   small enough to plot and look at.
    the TIMING (`--timing-n-clusters`) `--repeats` INDEPENDENT synthetic
                                       draws, a timing DISTRIBUTION and a
                                       disagreement RATE.
    REAL DATA (`--checkpoint`)         one C-tree's actual detections.
    REAL FLOW (`--real-flow`)          each implementation deployed into the
                                       real C-axis flow.

Deferred imports (torch, AccessDatasets, SafeSlide, ChainStack,
SurvivalProcess, prepare_chain_stack, reeval_density, survival_alpha_analysis)
inside `merge_grid`'s real-data/real-flow functions ONLY -- so a
`--parts merge_grid` run with no `--checkpoint` stays torch-free. `chains_
stack` needs all of those unconditionally (it always reads a real slide), so
they sit at module level.

Outputs (in `result/<SLURM_JOB_NAME or DemoSurvivalAnalysis>/`):
    figures/r_stack_{own,reuseF}.png                      [chains_stack]
    figures/pyramid_{lineage,overview}_{own,reuseF}.png   [chains_stack]
    merge_grid_demo.png                                   [merge_grid]
    merge_grid_demo_real.png        only with --checkpoint --plot-real
    merge_grid_demo_real_flow.png   only with --real-flow
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', '..', '..', 'utilities'),
          os.path.join(_HERE, '..')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _HERE not in sys.path:            # prepare_chain_stack.py lives right here
    sys.path.insert(0, _HERE)

from _paths import job_result_dir, setup_import_paths          # noqa: E402

setup_import_paths()

import matplotlib                                              # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.patches import Rectangle                        # noqa: E402
import numpy as np                                              # noqa: E402
import cv2                                                       # noqa: E402

import AccessDatasets                                              # noqa: E402
from SafeSlide import SafeSlide                                  # noqa: E402
from SurvivalAnalysis import ChainStack                          # noqa: E402
FStack, RStack, CStack = ChainStack.FStack, ChainStack.RStack, ChainStack.CStack
from SurvivalAnalysis.AlphaCalibration import merge_anchors        # noqa: E402
import prepare_chain_stack                                        # noqa: E402

DEFAULT_WSI_NAME = 'BRACS_1228'
DEFAULT_TILES_ROOT = prepare_chain_stack.DEFAULT_TILES_ROOT


# =============================================================================
#  chains_stack
# =============================================================================

def _looks_like_tissue(img) -> bool:
    """Cheap sanity, not a real assertion: a blank/near-uniform read (wrong
    level reading past the slide edge, a black hole) has near-zero std."""
    return float(np.std(img)) > 5.0


def _correlate(a, b) -> float:
    a = a.astype(np.float32).ravel() - a.astype(np.float32).mean()
    b = b.astype(np.float32).ravel() - b.astype(np.float32).mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _cache_stats(cache_root):
    """`(n_files, n_bytes)` under `cache_root`, ANY depth (`rglob`, not
    `glob` -- tiles sit in `cache_root/<Axis>Stack/<pyramid>/*.png`)."""
    if not cache_root or not os.path.isdir(cache_root):
        return 0, 0
    files = list(Path(cache_root).rglob('*.png'))
    return len(files), sum(f.stat().st_size for f in files)


def check_C(mother, mother_img, groups_by_ds, wsi, wsi_stem, tile, cache_root,
           lineage_index, label):
    """One lineage walk down from `mother`, PLUS the quantitative check:
    crop the mother's own image down to where a real child tile sits, and
    correlate that crop against the child's own independent read -- against a
    DECOY (the SAME crop vs a DIFFERENT child). Geometry (`mother`,
    `groups_by_ds`) is the caller's responsibility -- `own`'s comes from
    `CStack.from_own`, `reuse-F`'s from `CStack.from_mother` -- this only
    reads and checks, so both trees run through the exact same code.
    """
    t_read0 = time.perf_counter()
    lineage = {}                      # ds -> (group, [main imgs], overlap img)
    node = mother
    n_reads = 0
    for ds in sorted(groups_by_ds, reverse=True):
        group = next(g for g in groups_by_ds[ds] if g.parent == node)
        main_imgs = [CStack.read_one(m, wsi, wsi_stem=wsi_stem, root=mother,
                                     finest=min(groups_by_ds), tile=tile,
                                     cache_root=cache_root)
                    for m in group.main]
        overlap_img = CStack.read_one(group.overlap, wsi, wsi_stem=wsi_stem,
                                      root=mother, finest=min(groups_by_ds),
                                      tile=tile, cache_root=cache_root)
        n_reads += len(group.main) + 1
        lineage[ds] = (group, main_imgs, overlap_img)
        node = group.main[min(lineage_index, len(group.main) - 1)]
    t_read = time.perf_counter() - t_read0

    print(f"C  {label}: read {n_reads} real descendant tiles (one lineage) "
          f"in {t_read*1000:.1f} ms   ({t_read/max(n_reads,1)*1000:.1f} ms/tile)")

    # ── the quantitative check: mother crop vs child read, against a decoy ──
    first_ds = max(groups_by_ds)
    group0, mains0, _ = lineage[first_ds]
    scale = tile / mother.size_px
    def _crop_of(child):
        x0 = int(round((child.x - mother.x) * scale))
        y0 = int(round((child.y - mother.y) * scale))
        s = max(1, int(round(child.size_px * scale)))
        patch = mother_img[y0:y0 + s, x0:x0 + s]
        return cv2.resize(patch, (tile, tile), interpolation=cv2.INTER_AREA)

    crop = _crop_of(group0.main[0])
    real_corr = _correlate(crop, mains0[0])
    decoy_corr = _correlate(crop, mains0[-1])         # same crop, WRONG child
    passed = real_corr > decoy_corr + 0.1
    print(f"C  {label}: mother-crop vs its real child:  real={real_corr:.3f}  "
          f"decoy(wrong child)={decoy_corr:.3f}   "
          f"{'OK -- real wins' if passed else 'FAIL -- geometry/pixels disagree'}")

    return lineage


def read_whole_tree(wsi, wsi_stem, groups_by_ds, mother, mother_img, tile,
                    cache_root):
    """Every tile in the pyramid, GROUPED BY RUNG (`{ds: [(info, img), ...]}`).
    For the overview figure -- NOT what `check_C`'s own numbers cost, which
    only reads one lineage. Cached, so a second run of the same centre is
    nearly free, but the FIRST run at the default 4-transition `--c-rungs` is
    ~425 real reads.
    """
    t0 = time.perf_counter()
    finest = min(groups_by_ds)
    by_ds = {mother.ds: [(mother, mother_img)]}
    n = 1
    for ds, groups in groups_by_ds.items():
        items = []
        for g in groups:
            for m in g.main:
                items.append((m, CStack.read_one(m, wsi, wsi_stem=wsi_stem,
                                                 root=mother, finest=finest,
                                                 tile=tile,
                                                 cache_root=cache_root)))
            items.append((g.overlap,
                         CStack.read_one(g.overlap, wsi, wsi_stem=wsi_stem,
                                        root=mother, finest=finest, tile=tile,
                                        cache_root=cache_root)))
        by_ds[ds] = items
        n += len(items)
    dt = time.perf_counter() - t0
    print(f"C  overview: read all {n} real tiles in {dt:.2f} s "
          f"({dt/n*1000:.1f} ms/tile)")
    return by_ds


def _strip_figure(images_by_ds, title, out_path, footprints=None):
    """Left-to-right, finest to coarsest -- one panel per rung, no nesting."""
    rungs = sorted(images_by_ds)
    fig, axes = plt.subplots(1, len(rungs), figsize=(2.3 * len(rungs), 2.8))
    if len(rungs) == 1:
        axes = [axes]
    for ax, ds in zip(axes, rungs):
        ax.imshow(images_by_ds[ds])
        label = f'ds={ds:g}'
        if footprints is not None:
            label += f'\n{footprints[ds]:.0f} L0 px'
        ax.set_title(label, fontsize=9)
        ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _draw_tile_box(ax, info, *, color, style):
    ax.add_patch(Rectangle((info.x, info.y), info.size_px, info.size_px,
                            fill=False, edgecolor=color, linewidth=0.8,
                            linestyle=style))


#: One colour for every line in the overview, deliberately -- see plan.md's
#: case history: a colour-by-rung continuous colormap failed on its own
#: terms once the finest rung's sheer tile count dominated every panel.
_LINE_COLOR = '#d6273c'


def _overview_figure(by_ds, mother, title, out_path):
    """One panel per rung, coarsest (left) to finest (right), SAME physical
    area in every panel -- the tile count growing is the whole point."""
    rungs = sorted(by_ds, reverse=True)
    fig, axes = plt.subplots(1, len(rungs), figsize=(3.2 * len(rungs), 3.6))
    if len(rungs) == 1:
        axes = [axes]
    for ax, ds in zip(axes, rungs):
        for info, img in by_ds[ds]:
            ax.imshow(img, extent=(info.x, info.x + info.size_px,
                                   info.y + info.size_px, info.y))
            _draw_tile_box(ax, info, color=_LINE_COLOR,
                          style='-' if info.kind == 'main' else '--')
        ax.set_xlim(mother.x, mother.x + mother.size_px)
        ax.set_ylim(mother.y + mother.size_px, mother.y)
        ax.set_aspect('equal')
        n_main = sum(1 for info, _ in by_ds[ds] if info.kind == 'main')
        ax.set_title(f'ds={ds:g}\n{n_main} main tile'
                     f'{"s" if n_main != 1 else ""}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


#: Fixed, hand-picked list rather than a continuous colormap -- same
#: complaint as `_LINE_COLOR`: `viridis`/`plasma` both route through yellow
#: and through blue-and-purple-together.
_LINEAGE_COLORS = ('#d6273c', '#e07b1a', '#1a7a3c', '#6b4423', '#555555')


def _lineage_figure(mother, mother_img, lineage, title, out_path):
    """Real pixels, not empty rectangles -- the whole point of this run."""
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(mother_img, extent=(mother.x, mother.x + mother.size_px,
                                  mother.y + mother.size_px, mother.y))
    _draw_tile_box(ax, mother, color='black', style='-')

    rungs = sorted(lineage, reverse=True)
    for i, ds in enumerate(rungs):
        group, main_imgs, overlap_img = lineage[ds]
        color = _LINEAGE_COLORS[i % len(_LINEAGE_COLORS)]
        for m, img in zip(group.main, main_imgs):
            ax.imshow(img, extent=(m.x, m.x + m.size_px, m.y + m.size_px, m.y))
            _draw_tile_box(ax, m, color=color, style='-')
        o = group.overlap
        ax.imshow(overlap_img, extent=(o.x, o.x + o.size_px,
                                       o.y + o.size_px, o.y), alpha=0.6)
        _draw_tile_box(ax, o, color=color, style='--')

    ax.set_xlim(mother.x, mother.x + mother.size_px)
    ax.set_ylim(mother.y + mother.size_px, mother.y)
    ax.set_aspect('equal')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _run_chains_stack(args, out_dir: str) -> None:
    wsi_name = args.wsi_name or DEFAULT_WSI_NAME
    if args.wsi is None:
        entry = AccessDatasets.locate(wsi_name)
        wsi_path, wsi_stem = entry.path, entry.name
    else:
        wsi_path, wsi_stem = args.wsi, args.wsi_stem

    tiles_root = args.tiles_root or DEFAULT_TILES_ROOT
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    cache_root = args.cache_root or ChainStack.DEFAULT_CACHE_ROOT
    mother_ds = max(args.c_rungs)

    # Computed the same way prepare_chain_stack.py does -- imported, not
    # re-derived, see the module docstring for why this matters once stageA
    # and the two own corpora share one root.
    f_sid = prepare_chain_stack._sampler_config_for(
        'stageB-fOwn', args.tile).sampler_id()
    r_sid = prepare_chain_stack._sampler_config_for(
        'stageA', args.tile).sampler_id()
    c_sid = prepare_chain_stack._sampler_config_for(
        'stageB-cOwn', args.tile).sampler_id()

    print(f"slide: {wsi_path}")
    files_before, bytes_before = _cache_stats(cache_root)

    r_own_stack = None
    c_own_mother = c_own_img = c_own_groups = lineage_own = whole_tree_own = None

    with SafeSlide(wsi_path) as wsi:
        # ── F: own is F's only source ───────────────────────────────────
        f_own = FStack.from_own(tiles_root, wsi_stem, tile=args.tile,
                                rungs=args.rungs, sampler_id=f_sid)
        if not len(f_own):
            raise RuntimeError(
                f'F has no complete chain for {wsi_stem} under '
                f'{tiles_root} (sampler_id={f_sid}) -- run '
                f'prepare_chain_stack.py --axes F for this slide first')
        chain_id = next(iter(f_own))
        chain = f_own.chains[chain_id]
        print(f"centre: ({chain.cx:.0f}, {chain.cy:.0f}), chain {chain_id}")
        t0 = time.perf_counter()
        f_stack = f_own[chain_id]
        dt = time.perf_counter() - t0
        oks = [_looks_like_tissue(f_stack[d]) for d in f_stack]
        print(f"F  own: chain {chain_id}, {len(f_stack)} rungs read in "
              f"{dt*1000:.1f} ms   {'OK' if all(oks) else 'FAIL'}")

        # ── R: own (stageA, independent) and reuse-F (this same chain) ──
        r_own = RStack.from_own(tiles_root, wsi_stem, args.rungs,
                                tile=args.tile, sampler_id=r_sid,
                                cache_root=args.cache_root)
        if len(r_own):
            t0 = time.perf_counter()
            r_own_stack = r_own[0]
            dt = time.perf_counter() - t0
            oks = [_looks_like_tissue(r_own_stack[d]) for d in r_own_stack]
            print(f"R  own: tile 0, {len(r_own_stack)} rungs derived in "
                  f"{dt*1000:.1f} ms   {'OK' if all(oks) else 'FAIL'}")
        else:
            print(f"R  own: 0 tiles found for {wsi_stem} under "
                  f"{tiles_root} (sampler_id={r_sid}) -- skipped")

        t0 = time.perf_counter()
        r_stack = RStack.derive(chain, args.rungs, tile=args.tile, source='F')
        dt = time.perf_counter() - t0
        oks = [_looks_like_tissue(r_stack[d]) for d in args.rungs]
        print(f"R  reuse-F: derived {len(args.rungs)} rungs from chain "
              f"{chain_id} in {dt*1000:.1f} ms   {'OK' if all(oks) else 'FAIL'}")

        # ── C: own (stageB-cOwn, independent) ────────────────────────────
        c_own = CStack.from_own(tiles_root, wsi_stem, args.c_rungs,
                                wsi, tile=args.tile, sampler_id=c_sid,
                                cache_root=cache_root)
        if len(c_own):
            t0 = time.perf_counter()
            c_own_mother, c_own_img, c_own_groups, _ = c_own[0]
            dt = time.perf_counter() - t0
            print(f"C  own: tree 0, mother ds {c_own_mother.ds:g} read in "
                  f"{dt:.2f} s")
            lineage_own = check_C(c_own_mother, c_own_img, c_own_groups, wsi,
                                  wsi_stem, args.tile, cache_root,
                                  args.lineage_index, label='own')
            whole_tree_own = read_whole_tree(wsi, wsi_stem, c_own_groups,
                                             c_own_mother, c_own_img,
                                             args.tile, cache_root)
        else:
            print(f"C  own: 0 trees found for {wsi_stem} under "
                  f"{tiles_root} (sampler_id={c_sid}) -- skipped")

        # ── C: reuse-F (this same chain's own ds-16 read as the mother) ──
        if mother_ds not in chain.members:
            raise RuntimeError(
                f"chain {chain_id} has no ds {mother_ds:g} member -- C's "
                f"reuse-F path needs F's own chain to cover C's mother rung")
        reuse_mother_img = FStack.read(chain, tile=args.tile)[mother_ds]
        t_geom0 = time.perf_counter()
        mother, mother_img, groups_by_ds, _ = CStack.from_mother(
            reuse_mother_img, chain.cx, chain.cy, args.c_rungs, wsi,
            wsi_stem=wsi_stem, tile=args.tile, cache_root=cache_root)
        t_geom = time.perf_counter() - t_geom0
        n_groups = sum(len(g) for g in groups_by_ds.values())
        print(f"C  reuse-F: pyramid ({n_groups} groups, mother ds "
              f"{mother.ds:g}) built + read in {t_geom:.2f} s")
        lineage_reuse = check_C(mother, mother_img, groups_by_ds, wsi,
                                wsi_stem, args.tile, cache_root,
                                args.lineage_index, label='reuse-F')
        whole_tree_reuse = read_whole_tree(wsi, wsi_stem, groups_by_ds,
                                          mother, mother_img, args.tile,
                                          cache_root)

    files_after, bytes_after = _cache_stats(cache_root)
    print(f"local cache: +{files_after - files_before} files, "
          f"+{(bytes_after - bytes_before)/1024:.0f} KB THIS RUN "
          f"({files_after} files, {bytes_after/1024:.0f} KB total on disk)")

    # ONE C TREE AT A TIME, on its own -- the question "how big is one
    # pyramid" needs the pyramid's OWN directory (`_pyramid_dir`), not the
    # run-wide delta above, which also includes R's tiles and both trees.
    for label, root_mother in (('own', c_own_mother), ('reuse-F', mother)):
        if root_mother is None:
            continue
        c_dir = ChainStack._pyramid_dir(cache_root, 'C', wsi_stem,
                                        root_mother, args.tile,
                                        min(args.c_rungs))
        c_files, c_bytes = _cache_stats(c_dir)
        print(f"C  {label} tree on disk: {c_files} files, "
              f"{c_bytes/1024:.0f} KB ({c_bytes/1024/1024:.2f} MB)  -> {c_dir}")

    _strip_figure(r_stack, 'R axis (reuse-F), real tissue -- fixed footprint, '
                 'degraded resolution',
                 os.path.join(fig_dir, 'r_stack_reuseF.png'))
    if r_own_stack is not None:
        _strip_figure(r_own_stack, 'R axis (own), real tissue -- fixed '
                     'footprint, degraded resolution',
                     os.path.join(fig_dir, 'r_stack_own.png'))

    _lineage_figure(mother, mother_img, lineage_reuse,
                    'C axis (reuse-F), real tissue -- one lineage down from '
                    'the mother\nsolid = main   dashed = overlap (半透明, '
                    'shows the 1/4 share)',
                    os.path.join(fig_dir, 'pyramid_lineage_reuseF.png'))
    _overview_figure(whole_tree_reuse, mother,
                     'C axis (reuse-F), real tissue -- SAME area every '
                     'panel, coarsest (left) to finest (right)\n'
                     'solid = main (recurses)   dashed = overlap (leaf)',
                     os.path.join(fig_dir, 'pyramid_overview_reuseF.png'))
    if lineage_own is not None:
        _lineage_figure(c_own_mother, c_own_img, lineage_own,
                        'C axis (own), real tissue -- one lineage down from '
                        'the mother\nsolid = main   dashed = overlap (半透明, '
                        'shows the 1/4 share)',
                        os.path.join(fig_dir, 'pyramid_lineage_own.png'))
        _overview_figure(whole_tree_own, c_own_mother,
                         'C axis (own), real tissue -- SAME area every '
                         'panel, coarsest (left) to finest (right)\n'
                         'solid = main (recurses)   dashed = overlap (leaf)',
                         os.path.join(fig_dir, 'pyramid_overview_own.png'))

    print(f"\nfigures -> {fig_dir}/"
          f"{{r_stack_own,r_stack_reuseF,"
          f"pyramid_lineage_own,pyramid_overview_own,"
          f"pyramid_lineage_reuseF,pyramid_overview_reuseF}}.png")


# =============================================================================
#  merge_grid
# =============================================================================

def merge_within_radius_original(points: np.ndarray, radius: float,
                                 priority: Optional[np.ndarray] = None
                                 ) -> np.ndarray:
    """The pre-fix version. `kept_xy` is a Python list of points, rebuilt into
    a fresh array with `np.asarray` on every iteration -- that rebuild's own
    cost is O(k) per iteration, so summed over all n iterations it is its own
    O(n^2), stacked on top of the O(n*k) the comparisons already cost.
    """
    n = len(points)
    if not n or float(radius) <= 0.0:
        return np.arange(n, dtype=np.int64)
    order = (np.argsort(-np.asarray(priority)) if priority is not None
            else np.arange(n))
    kept: List[int] = []
    kept_xy: List[np.ndarray] = []
    for i in order:
        i = int(i)
        if kept_xy:
            near = np.linalg.norm(np.asarray(kept_xy) - points[i], axis=1)
            if float(near.min()) <= float(radius):
                continue
        kept.append(i)
        kept_xy.append(points[i])
    return np.sort(np.asarray(kept, np.int64))


def merge_within_radius_grid(points: np.ndarray, radius: float,
                             priority: Optional[np.ndarray] = None
                             ) -> np.ndarray:
    """Method A. Same contract as `SurvivalProcess._merge_within_radius`:
    indices to KEEP after a greedy, priority-ordered radius merge -- visit
    points highest priority first (input order if `priority` is None), keep
    a point only if no already-kept point is within `radius` of it.

    Only the SEARCH SCOPE changes. A candidate compares against the
    already-kept points in its own radius-sized grid cell and the
    surrounding 3x3, not every already-kept point: any point within `radius`
    of a cell of side `radius` must fall in that 3x3 neighbourhood, so
    nothing is missed. That is where the speed difference comes from, not
    from a looser merge.
    """
    n = len(points)
    if not n or float(radius) <= 0.0:
        return np.arange(n, dtype=np.int64)
    order = (np.argsort(-np.asarray(priority)) if priority is not None
            else np.arange(n))
    cell = float(radius)
    grid: Dict[Tuple[int, int], List[int]] = {}
    kept: List[int] = []
    for i in order:
        i = int(i)
        cx = int(np.floor(points[i, 0] / cell))
        cy = int(np.floor(points[i, 1] / cell))
        near = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cx + dx, cy + dy), ()):
                    if np.linalg.norm(points[j] - points[i]) <= radius:
                        near = True
                        break
                if near:
                    break
            if near:
                break
        if near:
            continue
        kept.append(i)
        grid.setdefault((cx, cy), []).append(i)
    return np.sort(np.asarray(kept, dtype=np.int64))


def synthetic_points(n_clusters: int, max_per_cluster: int, jitter: float,
                     extent: float, seed: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Fake keypoints: `n_clusters` physical points, each hit by 1..
    `max_per_cluster` near-duplicate detections (jitter = the sub-pixel
    disagreement a real point gets across rungs/tiles) scattered over an
    `extent` x `extent` canvas. Returns `(points, cluster_id)` --
    `cluster_id` is for the figure's ground-truth colouring only, never fed to
    any of the three merges.
    """
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, extent, size=(n_clusters, 2))
    pts: List[np.ndarray] = []
    cid: List[int] = []
    for k, c in enumerate(centres):
        m = int(rng.integers(1, max_per_cluster + 1))
        pts.append(c + rng.normal(0.0, jitter, size=(m, 2)))
        cid += [k] * m
    return np.concatenate(pts, axis=0), np.asarray(cid, dtype=np.int64)


def _load_real_points(args) -> Tuple[np.ndarray, float, str]:
    """Real detections from one C-tree, one rung -- read the same way `cli/
    survival_alpha_analysis.py`'s C axis does. Returns `(points, radius,
    label)`; `radius` is `net.cfg.nms_radius`, the same value production
    passes to `anchors_of_generations` for this exact call, not a demo value.

    Deferred imports: none of this loads (no torch, no model) unless
    `--checkpoint` is actually given, so the synthetic-only invocation stays
    model-free.
    """
    import torch                                                  # noqa: PLC0415
    from SurvivalAnalysis import SurvivalProcess                    # noqa: PLC0415
    from reeval_density import load_arm                              # noqa: PLC0415

    if not args.wsi_name:
        raise SystemExit('--wsi-name is required with --checkpoint')

    entry = AccessDatasets.locate(args.wsi_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, _identity, _ = load_arm(args.checkpoint, device)
    threshold = (args.score_threshold if args.score_threshold is not None
                else float(net.cfg.detection_threshold))

    tiles_root = args.tiles_root or DEFAULT_TILES_ROOT
    c_sid = prepare_chain_stack._sampler_config_for(
        'stageB-cOwn', args.tile).sampler_id()

    with SafeSlide(entry.path) as wsi:
        forest = ChainStack.CStack.from_own(
            tiles_root, entry.name, args.c_rungs, wsi, tile=args.tile,
            sampler_id=c_sid,
            cache_root=args.cache_root or ChainStack.DEFAULT_CACHE_ROOT)
        mother, mother_image, groups_by_ds, images_by_ds = forest[args.tree_index]
        _per_rung_tiles, per_rung = SurvivalProcess.detect_all_generations(
            mother, mother_image, groups_by_ds, images_by_ds, net,
            score_threshold=threshold)

    rung = args.real_rung if args.real_rung is not None else min(args.c_rungs)
    if rung not in per_rung:
        raise SystemExit(f'--real-rung {rung} not in --c-rungs {args.c_rungs}')
    xy0, _score = per_rung[rung]
    label = f'{entry.name}  C-tree {args.tree_index}  ds={rung:g}'
    return np.asarray(xy0, np.float64), float(net.cfg.nms_radius), label


def _run_real_flow_comparison(args) -> Dict[str, Tuple[Optional[dict], float,
                                                       int, List[np.ndarray]]]:
    """Deploy `original` / `B` / `A` into `cli/survival_alpha_analysis.py`'s
    own C-axis flow (`anchors_of_generations` -> `probe_real` ->
    `probe_decoy` -> `alpha_curve` -> `aggregate_curves`), monkey-patching
    `SurvivalProcess._merge_within_radius` for the duration of each candidate
    -- both `anchors_of_generations` and `AlphaCalibration.merge_anchors`
    look this name up on the `SurvivalProcess` module at CALL time, so
    replacing the module attribute redirects every caller without touching
    either file.

    Detection (`detect_all_generations`, the GPU part) runs ONCE per tree and
    is shared by all three candidates -- only the merge onward is repeated
    per candidate, both because that isolates what actually changed and
    because it is the only way this stays a demo and not three independent
    full analysis runs.

    `rng` is reseeded from `args.seed` before each candidate so the SAME
    decoy draw is used for all three -- any numeric difference in the
    aggregated curves can then only come from the merge implementation, not
    from a different random decoy.
    """
    import survival_alpha_analysis as saa                          # noqa: PLC0415
    import torch                                                    # noqa: PLC0415
    from SurvivalAnalysis import SurvivalProcess                    # noqa: PLC0415
    from reeval_density import load_arm                                # noqa: PLC0415

    if not args.wsi_name:
        raise SystemExit('--wsi-name is required with --real-flow')

    entry = AccessDatasets.locate(args.wsi_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, _identity, _ = load_arm(args.checkpoint, device)
    threshold = (args.score_threshold if args.score_threshold is not None
                else float(net.cfg.detection_threshold))
    tiles_root = args.tiles_root or DEFAULT_TILES_ROOT
    c_sid = prepare_chain_stack._sampler_config_for(
        'stageB-cOwn', args.tile).sampler_id()
    order = sorted(float(r) for r in args.c_rungs)
    alphas = np.arange(0.5, 4.0 + 0.125, 0.25)     # survival_alpha_analysis.py's own default

    with SafeSlide(entry.path) as wsi:
        forest = ChainStack.CStack.from_own(
            tiles_root, entry.name, args.c_rungs, wsi, tile=args.tile,
            sampler_id=c_sid,
            cache_root=args.cache_root or ChainStack.DEFAULT_CACHE_ROOT)
        chainstacks = [forest[i] for i in forest][:args.real_flow_n_trees]

        tree_data = []
        tree_max_n = []
        for mother, mother_image, groups_by_ds, images_by_ds in chainstacks:
            per_rung_tiles, per_rung = SurvivalProcess.detect_all_generations(
                mother, mother_image, groups_by_ds, images_by_ds, net,
                score_threshold=threshold)
            tree_data.append((per_rung_tiles, per_rung))
            tree_max_n.append(max(len(xy0) for xy0, _sc in per_rung.values()))

    # `original` gets the SAME guard the other three comparisons already
    # have -- real C-tree point counts run into the tens of thousands (see
    # the [real data] section), where `original`'s extra O(n^2) rebuild cost
    # on top of B's O(n*k) can take far longer than B itself, per tree, with
    # no progress printed inside the merge to show it is still working.
    skip_original = any(n > args.skip_original_above for n in tree_max_n)
    if skip_original:
        print(f'  original skipped -- max {max(tree_max_n)} raw points in a '
             f'rung exceeds --skip-original-above {args.skip_original_above}',
             flush=True)

    saved_impl = SurvivalProcess._merge_within_radius
    candidates = {'original': merge_within_radius_original,
                 'B': None,
                 'A': merge_within_radius_grid}
    results: Dict[str, Tuple[Optional[dict], Optional[float], int,
                            List[np.ndarray]]] = {}
    try:
        for name, fn in candidates.items():
            if name == 'original' and skip_original:
                results[name] = (None, None, len(tree_data), [])
                continue
            SurvivalProcess._merge_within_radius = (fn if fn is not None
                                                    else saved_impl)
            rng = np.random.default_rng(args.seed)
            t0 = time.perf_counter()
            curves = []
            anchors_per_tree: List[np.ndarray] = []
            for per_rung_tiles, per_rung in tree_data:
                anchors = SurvivalProcess.anchors_of_generations(
                    per_rung_tiles, order, net.cfg.nms_radius)
                anchors_per_tree.append(anchors)
                dist, score, _rival, _ = SurvivalProcess.probe_real(
                    anchors, per_rung, order)
                decoy_shift = saa._decoy_shift_per_rung(
                    'fixed', 8.0, order, 'C', rng)
                decoy_dist, decoy_score = saa.probe_decoy(
                    anchors, per_rung, order, decoy_shift)
                curves.append(saa.alpha_curve(
                    dist, score, decoy_dist, decoy_score, rungs=order,
                    alphas=alphas, tau_floor=0.0, threshold=threshold))
            dt = time.perf_counter() - t0
            bucket = saa.aggregate_curves(curves) if curves else None
            results[name] = (bucket, dt, len(tree_data), anchors_per_tree)
    finally:
        SurvivalProcess._merge_within_radius = saved_impl

    return results


def _anchor_mismatch(ref_anchors: List[np.ndarray], cmp_anchors: List[np.ndarray]
                     ) -> float:
    """Mean, across trees, of the symmetric difference between two
    candidates' anchor sets (rounded to 1e-6 -- these are deterministic
    floating-point results, so identical inputs through the identical
    algorithm produce identical values; rounding only guards against
    reordered-summation noise) as a fraction of the reference tree's anchor
    count. 0.0 means every anchor in every tree matched exactly, not just
    that the COUNTS matched.
    """
    fracs = []
    for a_ref, a_cmp in zip(ref_anchors, cmp_anchors):
        set_ref = set(map(tuple, np.round(a_ref, 6).tolist()))
        set_cmp = set(map(tuple, np.round(a_cmp, 6).tolist()))
        n_total = max(len(set_ref), 1)
        fracs.append(len(set_ref ^ set_cmp) / n_total)
    return float(np.mean(fracs)) if fracs else 0.0


def _plot_real_flow(anchors_by_candidate: Dict[str, np.ndarray],
                    curves_by_candidate: Dict[str, Optional[dict]],
                    order: List[float], label: str, out_path: str) -> None:
    """Two panels: the first tree's anchors from all three candidates
    OVERLAID (identical implementations put identical markers on top of each
    other -- a visible offset IS a disagreement), and the finest rung's
    match_rate vs alpha, same overlay logic, for the aggregated curves.
    """
    colours = {'original': 'steelblue', 'B': 'crimson', 'A': 'seagreen'}
    markers = {'original': 'x', 'B': 'o', 'A': '+'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    for name, anchors in anchors_by_candidate.items():
        ax1.scatter(anchors[:, 0], anchors[:, 1], s=24, c=colours[name],
                   marker=markers[name], alpha=0.6,
                   label=f'{name} ({len(anchors)})')
    ax1.set_title('anchors, first C-tree (overlaid)', fontsize=9)
    ax1.invert_yaxis()
    ax1.set_aspect('equal')
    ax1.legend(fontsize=7)

    finest = order[0]
    for name, bucket in curves_by_candidate.items():
        if bucket is None:
            continue
        ax2.plot(bucket['alphas'], bucket['match_rate_mean'][0, :],
                color=colours[name], marker=markers[name], ms=4,
                label=name)
    ax2.set_xlabel('alpha')
    ax2.set_ylabel('match_rate')
    ax2.set_title(f'match_rate vs alpha, finest rung ds={finest:g} (overlaid)',
                 fontsize=9)
    ax2.legend(fontsize=7)

    fig.suptitle(f'real flow -- {label}', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _draw_scatter_only(ax, points: np.ndarray, kept_idx: np.ndarray,
                       title: str) -> None:
    """Same colouring as `_draw_merge_grid_figure`, no per-point circles and
    no grid lines -- real detection counts run into the thousands, where
    circles for every kept point would be unreadable clutter rather than an
    aid.
    """
    ax.set_title(title, fontsize=9)
    dropped = np.setdiff1d(np.arange(len(points)), kept_idx)
    ax.scatter(points[dropped, 0], points[dropped, 1], s=4, c='lightgray',
              label=f'merged away ({len(dropped)})')
    ax.scatter(points[kept_idx, 0], points[kept_idx, 1], s=10, c='crimson',
              label=f'kept ({len(kept_idx)})', zorder=3)
    ax.set_aspect('equal')
    ax.invert_yaxis()                    # image row order, level-0 y grows down
    ax.legend(fontsize=7, loc='upper right', markerscale=2)


def _mismatch(a: np.ndarray, b: np.ndarray, n_total: int) -> Tuple[int, float]:
    """Symmetric difference between two KEPT index sets: how many points one
    implementation kept that the other dropped, or vice versa, as a count
    and a fraction of `n_total`. `len(a) == len(b)` is not enough -- two
    same-SIZE sets can still disagree on WHICH points survived, and a bare
    boolean `agree` says nothing about how far apart a disagreement is.
    """
    diff = len(set(a.tolist()) ^ set(b.tolist()))
    return diff, (diff / n_total if n_total else 0.0)


def _time_it(fn, points, radius, priority, repeats: int) -> Tuple[np.ndarray, float]:
    """Best-of-`repeats` wall time -- the min, not the mean, because a stray
    scheduler hiccup can only make a run SLOWER, never faster than its true
    cost.
    """
    best = None
    idx = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        idx = fn(points, radius, priority=priority)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return idx, best


def _timing_multi_draw(args) -> bool:
    """`--repeats` INDEPENDENT synthetic draws -- a fresh seed every time, not
    the same data timed repeatedly (that would only measure scheduler noise
    on ONE layout). Each draw is timed once per implementation and checked
    for mismatch against B; the report is a timing DISTRIBUTION across draws
    plus a disagreement RATE (how many of the draws produced ANY mismatch),
    which a single boolean 'agree' from one draw cannot tell you.
    """
    times: Dict[str, List[float]] = {'B': [], 'A': [], 'original': []}
    mismatch_ba: List[float] = []
    mismatch_ob: List[float] = []
    n_skipped = 0

    for d in range(args.repeats):
        seed = args.seed + 1000 + d
        t_points, _ = synthetic_points(
            args.timing_n_clusters, args.max_per_cluster, args.jitter,
            args.timing_extent, seed)
        t_priority = np.random.default_rng(seed + 1).uniform(size=len(t_points))
        n_t = len(t_points)

        t0 = time.perf_counter()
        kept_b = merge_anchors(t_points, args.radius, priority=t_priority)
        times['B'].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        kept_a = merge_within_radius_grid(t_points, args.radius, priority=t_priority)
        times['A'].append(time.perf_counter() - t0)
        _, frac_ba = _mismatch(kept_b, kept_a, n_t)
        mismatch_ba.append(frac_ba)

        if n_t <= args.skip_original_above:
            t0 = time.perf_counter()
            kept_o = merge_within_radius_original(t_points, args.radius,
                                                  priority=t_priority)
            times['original'].append(time.perf_counter() - t0)
            _, frac_ob = _mismatch(kept_o, kept_b, n_t)
            mismatch_ob.append(frac_ob)
        else:
            n_skipped += 1

    def _stats(name: str) -> str:
        arr = np.asarray(times[name])
        return f'mean {arr.mean():.4f}s (std {arr.std():.4f}s)' if len(arr) else 'n/a'

    b_mean = float(np.mean(times['B']))
    a_mean = float(np.mean(times['A']))
    print(f'\n[timing set]  {args.repeats} independent draws, '
         f'~{args.timing_n_clusters} clusters each, radius {args.radius:g}, '
         f'clustering re-drawn (different seed) every repeat')
    print(f'  B (array buffer, retired) -> {_stats("B")}')
    print(f'  A (grid, production)         -> {_stats("A")}   '
         f'({(b_mean / a_mean if a_mean > 0 else float("inf")):.1f}x vs B, '
         f'mean over draws)')
    n_disagree_ba = sum(1 for x in mismatch_ba if x > 0)
    print(f'  B vs A mismatch: mean {np.mean(mismatch_ba):.2%} of kept '
         f'points, disagreement in {n_disagree_ba}/{args.repeats} draws')

    agree = (n_disagree_ba == 0)
    if times['original']:
        o_mean = float(np.mean(times['original']))
        b_mean_paired = float(np.mean(times['B'][:len(times['original'])]))
        print(f'  original                      -> {_stats("original")}   '
             f'({(o_mean / b_mean_paired if b_mean_paired > 0 else float("inf")):.1f}'
             f'x slower than B)')
        n_disagree_ob = sum(1 for x in mismatch_ob if x > 0)
        print(f'  B vs original mismatch: mean {np.mean(mismatch_ob):.2%} of '
             f'kept points, disagreement in {n_disagree_ob}/{len(mismatch_ob)} draws')
        agree = agree and (n_disagree_ob == 0)
    if n_skipped:
        print(f'  original skipped on {n_skipped}/{args.repeats} draws -- '
             f'point count exceeded --skip-original-above '
             f'{args.skip_original_above}')

    return agree


def _draw_merge_grid_figure(ax, points: np.ndarray, kept_idx: np.ndarray,
                            radius: float, extent: float, title: str) -> None:
    ax.set_title(title, fontsize=9)
    dropped = np.setdiff1d(np.arange(len(points)), kept_idx)
    ax.scatter(points[dropped, 0], points[dropped, 1], s=14, c='lightgray',
              label=f'merged away ({len(dropped)})')
    ax.scatter(points[kept_idx, 0], points[kept_idx, 1], s=28, c='crimson',
              marker='*', label=f'kept ({len(kept_idx)})', zorder=3)
    for x, y in points[kept_idx]:
        ax.add_patch(plt.Circle((x, y), radius, fill=False, color='crimson',
                                alpha=0.25, lw=0.8))
    n_cells = int(np.ceil(extent / radius)) + 1
    for k in range(n_cells + 1):
        ax.axvline(k * radius, color='steelblue', lw=0.4, alpha=0.35)
        ax.axhline(k * radius, color='steelblue', lw=0.4, alpha=0.35)
    ax.set_xlim(-radius, extent + radius)
    ax.set_ylim(-radius, extent + radius)
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper right')


def _run_merge_grid(args, out_dir: str) -> int:
    # ── small set: figure + correctness ──────────────────────────────────────
    points, _cluster_id = synthetic_points(
        args.n_clusters, args.max_per_cluster, args.jitter, args.extent,
        args.seed)
    priority = np.random.default_rng(args.seed + 1).uniform(size=len(points))

    kept_orig = merge_within_radius_original(points, args.radius, priority=priority)
    kept_ref = merge_anchors(points, args.radius, priority=priority)
    kept_grid = merge_within_radius_grid(points, args.radius, priority=priority)

    n_diff_ob, frac_ob = _mismatch(kept_orig, kept_ref, len(points))
    n_diff_ba, frac_ba = _mismatch(kept_ref, kept_grid, len(points))
    agree_small = (n_diff_ob == 0 and n_diff_ba == 0)
    print(f'[figure set]  {len(points)} raw points, {args.n_clusters} true '
         f'clusters, radius {args.radius:g}')
    print(f'  original -> {len(kept_orig)} kept')
    print(f'  B (array buffer, retired) -> {len(kept_ref)} kept')
    print(f'  A (grid, production)  -> {len(kept_grid)} kept')
    print(f'  mismatch  orig-vs-B: {n_diff_ob} pts ({frac_ob:.2%})   '
         f'B-vs-A: {n_diff_ba} pts ({frac_ba:.2%})')
    print(f'  all three agree: {agree_small}')

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    _draw_merge_grid_figure(axes[0], points, kept_orig, args.radius, args.extent,
                           f'original (list rebuild)\n{len(kept_orig)} anchors')
    _draw_merge_grid_figure(axes[1], points, kept_ref, args.radius, args.extent,
                           f'B: production (array buffer)\n{len(kept_ref)} anchors')
    _draw_merge_grid_figure(axes[2], points, kept_grid, args.radius, args.extent,
                           f'A: grid\n{len(kept_grid)} anchors, agree={agree_small}')
    fig.suptitle('_merge_within_radius: three implementations, same input',
                fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = os.path.join(out_dir, 'merge_grid_demo.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out_path}')

    # ── large set: timing distribution + disagreement rate, not plotted ─────
    agree_large = _timing_multi_draw(args)

    # ── real data: timing + agreement, plotted only if asked ────────────────
    agree_real = True
    if args.checkpoint:
        print(f'\n[real data]  {args.wsi_name}  C-tree {args.tree_index}  '
             f'detecting...', flush=True)
        r_points, r_radius, r_label = _load_real_points(args)
        n_r = len(r_points)
        print(f'[real data]  {r_label}  {n_r} raw detections, '
             f'radius {r_radius:g} (net.cfg.nms_radius), '
             f'best of {args.repeats}')

        kept_b_r, dt_b_r = _time_it(
            lambda p, r, priority: merge_anchors(p, r, priority=priority),
            r_points, r_radius, None, args.repeats)
        print(f'  B (array buffer, retired) -> {len(kept_b_r)} kept   '
             f'{dt_b_r:.4f}s')

        kept_grid_r, dt_grid_r = _time_it(merge_within_radius_grid, r_points,
                                          r_radius, None, args.repeats)
        speedup = f'{dt_b_r / dt_grid_r:.1f}x vs B' if dt_grid_r > 0 else 'too fast to time'
        print(f'  A (grid, production)         -> {len(kept_grid_r)} kept   '
             f'{dt_grid_r:.4f}s   ({speedup})')

        n_diff_ba_r, frac_ba_r = _mismatch(kept_b_r, kept_grid_r, n_r)
        agree_real = (n_diff_ba_r == 0)
        print(f'  B vs A mismatch: {n_diff_ba_r} pts ({frac_ba_r:.2%})')

        if n_r <= args.skip_original_above:
            kept_orig_r, dt_orig_r = _time_it(
                merge_within_radius_original, r_points, r_radius, None,
                args.repeats)
            slower = (f'{dt_orig_r / dt_b_r:.1f}x slower than B'
                     if dt_b_r > 0 else 'n/a')
            print(f'  original                      -> {len(kept_orig_r)} '
                 f'kept   {dt_orig_r:.4f}s   ({slower})')
            n_diff_ob_r, frac_ob_r = _mismatch(kept_orig_r, kept_b_r, n_r)
            print(f'  B vs original mismatch: {n_diff_ob_r} pts ({frac_ob_r:.2%})')
            agree_real = agree_real and (n_diff_ob_r == 0)
            print(f'  all three agree: {agree_real}')
        else:
            print(f'  original skipped -- {n_r} points exceeds '
                 f'--skip-original-above {args.skip_original_above}')

        if args.plot_real:
            fig_r, axes_r = plt.subplots(1, 2, figsize=(11, 5.5))
            _draw_scatter_only(axes_r[0], r_points, kept_b_r,
                              f'B: production\n{len(kept_b_r)} anchors')
            _draw_scatter_only(axes_r[1], r_points, kept_grid_r,
                              f'A: grid\n{len(kept_grid_r)} anchors, '
                              f'agree={agree_real}')
            fig_r.suptitle(f'real data -- {r_label}', fontsize=11)
            fig_r.tight_layout(rect=(0, 0, 1, 0.94))
            real_out = os.path.join(out_dir, 'merge_grid_demo_real.png')
            fig_r.savefig(real_out, dpi=150, bbox_inches='tight')
            plt.close(fig_r)
            print(f'Saved  {real_out}')

    # ── real flow: deploy each implementation, compare the aggregated result ─
    agree_flow = True
    if args.real_flow:
        print(f'\n[real flow]  {args.wsi_name}  up to '
             f'{args.real_flow_n_trees} C-trees  (survival_alpha_analysis.py'
             f'\'s own C-axis flow, each implementation deployed in turn)')
        flow_results = _run_real_flow_comparison(args)
        ref_bucket, ref_dt, ref_n, ref_anchors = flow_results['B']
        print(f'  B (retired)     -> {ref_n} ChainStacks   {ref_dt:.4f}s   '
             f'{sum(len(a) for a in ref_anchors)} anchors total')
        for name in ('original', 'A'):
            bucket, dt, n_trees, anchors_per_tree = flow_results[name]
            if dt is None:
                print(f'  {name:8s}        -> skipped (see above)')
                continue
            speed_note = f'{ref_dt / dt:.1f}x vs B' if dt > 0 else 'n/a'
            print(f'  {name:8s}        -> {n_trees} ChainStacks   {dt:.4f}s'
                 f'   ({speed_note})')

            anchor_mismatch = _anchor_mismatch(ref_anchors, anchors_per_tree)
            print(f'    anchor mismatch vs B: mean {anchor_mismatch:.2%} '
                 f'across {len(ref_anchors)} trees')

            if ref_bucket is None or bucket is None:
                same = (ref_bucket is None) == (bucket is None)
                print(f'    aggregated curves match B: {same}')
            else:
                keys = ('match_rate_mean', 'decoy_rate_mean', 'gap_mean',
                       'margin_mean')
                max_diffs = {k: float(np.nanmax(np.abs(ref_bucket[k] - bucket[k])))
                            for k in keys}
                same = all(np.allclose(ref_bucket[k], bucket[k], equal_nan=True)
                          for k in keys)
                print(f'    aggregated curves match B: {same}   max abs diff: '
                     + ', '.join(f'{k.replace("_mean", "")}={v:.2e}'
                                for k, v in max_diffs.items()))
            agree_flow = agree_flow and same and (anchor_mismatch == 0.0)
        print(f'  all three match: {agree_flow}')

        anchors_by_candidate = {name: flow_results[name][3][0]
                                for name in ('original', 'B', 'A')
                                if flow_results[name][3]}
        curves_by_candidate = {name: flow_results[name][0]
                               for name in ('original', 'B', 'A')}
        if anchors_by_candidate:
            order = sorted(float(r) for r in args.c_rungs)
            flow_out = os.path.join(out_dir, 'merge_grid_demo_real_flow.png')
            _plot_real_flow(anchors_by_candidate, curves_by_candidate, order,
                           f'{args.wsi_name}, first C-tree', flow_out)
            print(f'Saved  {flow_out}')

    return 0 if (agree_small and agree_large and agree_real and agree_flow) else 1


# =============================================================================
#  dispatch
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parts', nargs='+', choices=['chains_stack', 'merge_grid'],
                    default=['chains_stack', 'merge_grid'],
                    help='which demo(s) to run -- both by default')

    # ── shared ────────────────────────────────────────────────────────────
    ap.add_argument('--wsi-name', default=None,
                    help='resolved via AccessDatasets -- chains_stack '
                         f'defaults to {DEFAULT_WSI_NAME} if unset; '
                         'merge_grid needs it only with --checkpoint/'
                         '--real-flow')
    ap.add_argument('--tiles-root', default=None,
                    help='default: prepare_chain_stack.DEFAULT_TILES_ROOT')
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--c-rungs', type=float, nargs='+',
                    default=[1.0, 2.0, 4.0, 8.0, 16.0],
                    help="chains_stack: C's mother is always the coarsest "
                         'of these. merge_grid: which rungs a real C-tree '
                         'covers')
    ap.add_argument('--cache-root', default=None,
                    help='default: ChainStack.DEFAULT_CACHE_ROOT')
    ap.add_argument('--out', default=None,
                    help='output DIRECTORY (default: '
                         "job_result_dir('DemoSurvivalAnalysis')) -- shared "
                         'root for both parts, each keeps its own filenames')

    # ── chains_stack only ────────────────────────────────────────────────
    ap.add_argument('--wsi', default=None,
                    help='[chains_stack] bypass AccessDatasets, use this '
                         'path directly (needs --wsi-stem too)')
    ap.add_argument('--wsi-stem', default=None)
    ap.add_argument('--rungs', type=float, nargs='+',
                    default=[1.0, 2.0, 4.0, 8.0, 16.0],
                    help='[chains_stack] F chain completeness + R output '
                         "rungs -- must match prepare_chain_stack.py's own "
                         '--rungs')
    ap.add_argument('--lineage-index', type=int, default=0,
                    help='[chains_stack]')

    # ── merge_grid only ───────────────────────────────────────────────────
    ap.add_argument('--n-clusters', type=int, default=40,
                    help='[merge_grid] figure only -- kept small enough to plot')
    ap.add_argument('--max-per-cluster', type=int, default=6,
                    help='[merge_grid]')
    ap.add_argument('--jitter', type=float, default=1.5, help='[merge_grid]')
    ap.add_argument('--extent', type=float, default=100.0, help='[merge_grid]')
    ap.add_argument('--radius', type=float, default=6.0, help='[merge_grid]')
    ap.add_argument('--seed', type=int, default=0, help='[merge_grid]')
    ap.add_argument('--timing-n-clusters', type=int, default=800,
                    help='[merge_grid] timing/agreement only -- not plotted')
    ap.add_argument('--timing-extent', type=float, default=400.0,
                    help='[merge_grid]')
    ap.add_argument('--skip-original-above', type=int, default=20_000,
                    help='[merge_grid] point count above which the O(n^2) '
                         'original is skipped in the timing comparison')
    ap.add_argument('--repeats', type=int, default=3, help='[merge_grid]')
    ap.add_argument('--checkpoint', default=None,
                    help='[merge_grid] enables the real-data comparison: a '
                         'trained KeypointNet .pt, loaded to detect on one '
                         'C-tree')
    ap.add_argument('--tree-index', type=int, default=0,
                    help='[merge_grid] which C-tree in the forest to read')
    ap.add_argument('--real-rung', type=float, default=None,
                    help='[merge_grid] which c-rung to pull detections from '
                         '(default: the finest, min(--c-rungs))')
    ap.add_argument('--score-threshold', type=float, default=None,
                    help="[merge_grid] defaults to the checkpoint's own "
                         'cfg.detection_threshold')
    ap.add_argument('--plot-real', action='store_true',
                    help='[merge_grid] also save a scatter of the real '
                         'point cloud')
    ap.add_argument('--real-flow', action='store_true',
                    help='[merge_grid] enables the FOURTH comparison: '
                         'deploy each of the three implementations into '
                         "survival_alpha_analysis.py's own C-axis flow. "
                         'Requires --checkpoint and --wsi-name')
    ap.add_argument('--real-flow-n-trees', type=int, default=2,
                    help='[merge_grid] how many C-trees to run the flow on')
    args = ap.parse_args()

    out_dir = args.out or job_result_dir('DemoSurvivalAnalysis')
    os.makedirs(out_dir, exist_ok=True)

    status = 0
    if 'chains_stack' in args.parts:
        print('======== [chains_stack] ========')
        _run_chains_stack(args, out_dir)
    if 'merge_grid' in args.parts:
        print('\n======== [merge_grid] ========')
        status = max(status, _run_merge_grid(args, out_dir))
    return status


if __name__ == '__main__':
    sys.exit(main())
