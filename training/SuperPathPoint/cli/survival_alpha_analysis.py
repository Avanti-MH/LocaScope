#!/usr/bin/env python3
"""Calibrate alpha (`tau = max(tau_floor, alpha * ds)`) against a decoy,
one bucket (F/R/C) at a time. plan.md 2.2①.

    python training/SuperPathPoint/cli/survival_alpha_analysis.py \
        --checkpoint <.pt> --wsi <path> --wsi-stem <stem> --axes C

ORCHESTRATION ONLY. Every number is computed in `SurvivalAnalysis/
SurvivalProcess.py` (real measurement) and `SurvivalAnalysis/
AlphaCalibration.py` (decoy, curves, cross-ChainStack aggregation) -- this
file loads the detector, collects each axis's ChainStacks from `ChainStack.py`,
calls those two modules per ChainStack, and plots the result.

STATISTICS NEVER CROSS A BUCKET. F/R/C are three independent runs through
the same per-ChainStack steps; nothing here averages across axes.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
from typing import Callable, Dict, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..', 'utilities'))
import _paths                                                     # noqa: E402
_paths.setup_import_paths('SuperPathPoint')

from cli import (add_chainstack_args, add_corpus_args,  # noqa: E402
                 chainstack_root, job_result_dir)


import matplotlib                                                # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                                # noqa: E402
import torch                                                       # noqa: E402

import AccessDatasets                                                # noqa: E402
from SafeSlide import SafeSlide                                    # noqa: E402
from SurvivalAnalysis import ChainStack, SurvivalProcess           # noqa: E402
from SurvivalAnalysis.AlphaCalibration import (                    # noqa: E402
    aggregate_curves, aggregate_offset_quantiles, aggregate_pattern_curves,
    alpha_curve, merge_anchors, offset_quantiles_of, pattern_curve,
    probability_map_curve, probability_map_pattern_curve, probe_decoy)
from SurvivalAnalysis.Patterns import ASCII_NAMES, PATTERNS            # noqa: E402
from SurvivalAnalysis import AliveCandidates                          # noqa: E402

#: Fixed per-pattern colours, one hand-picked hue per `PATTERNS` entry rather
#: than matplotlib's `C0`-`C5` cycle -- matches the "六態存活譜" reference
#: figure (claude.ai/code/artifact/501f65f3-bd59-4a04-92ca-958602f85270) so a
#: pattern reads as the same colour there and in every plot this file draws.
PATTERN_COLORS = {
    '一直存活': '#1D4E77',   # alive-everywhere
    '細部存活': '#2B7A8C',   # fine-only
    '晚生型':   '#4E9C6C',   # late-born
    '只在一階': '#94BD3E',   # one-rung-only
    '中間帶':   '#CE9A24',   # mid-band
    '不連續':   '#C0453A',   # flicker
}
import prepare_chain_stack                                          # noqa: E402
from reeval_density import load_arm                                  # noqa: E402

FStack, RStack, CStack = ChainStack.FStack, ChainStack.RStack, ChainStack.CStack


# ── decoy_shift factories -- CLI-facing strategy choices, not core detection ──

def decoy_shift_fixed(offset_xy: Tuple[float, float]
                      ) -> Callable[[np.ndarray], np.ndarray]:
    """Same `(dx, dy)` added to every anchor, every call."""
    offset = np.asarray(offset_xy, np.float64)

    def _shift(anchors: np.ndarray) -> np.ndarray:
        return anchors + offset
    return _shift


def decoy_shift_random(min_mag: float, max_mag: float,
                       rng: np.random.Generator
                       ) -> Callable[[np.ndarray], np.ndarray]:
    """Each anchor shifted independently: magnitude uniform in
    `[min_mag, max_mag]`, direction uniform over the circle.
    """
    def _shift(anchors: np.ndarray) -> np.ndarray:
        n = len(anchors)
        magnitude = rng.uniform(min_mag, max_mag, size=n)
        angle = rng.uniform(0.0, 2 * np.pi, size=n)
        offset = np.stack([magnitude * np.cos(angle),
                          magnitude * np.sin(angle)], axis=1)
        return anchors + offset
    return _shift


def decoy_shift_rotate(angle_range: Tuple[float, float],
                       rng: np.random.Generator
                       ) -> Callable[[np.ndarray], np.ndarray]:
    """Every anchor ROTATED around the anchor set's own centroid
    (`anchors.mean(axis=0)`, the "同心軸") -- but EACH anchor its OWN angle,
    drawn independently from `angle_range`, not one shared angle for the
    whole set. `angle_range` straddling 0 means each anchor's turn is
    independently clockwise or counter-clockwise, not the same direction
    for all of them.

    THIS IS A ROTATION, NOT A SHIFT, AND THAT HAS A CONSEQUENCE
    `decoy_shift_fixed`/`_random` DO NOT: a point exactly at the centroid
    does not move at all, and one near it barely does (a rotation by angle
    `t` moves a point `2 * r * sin(t / 2)`, `r` its distance from the
    centre) -- the decoy is weak for anchors clustered near the middle of
    the set and only meaningful for ones spread away from it. No
    `magnitude` parameter here because there is none to give: the
    displacement is entirely a function of where an anchor already sits
    relative to the centroid.
    """
    def _shift(anchors: np.ndarray) -> np.ndarray:
        centre = anchors.mean(axis=0)
        angle = rng.uniform(*angle_range, size=len(anchors))
        c, s = np.cos(angle), np.sin(angle)
        rel = anchors - centre
        rotated = np.stack([c * rel[:, 0] - s * rel[:, 1],
                           s * rel[:, 0] + c * rel[:, 1]], axis=1)
        return rotated + centre
    return _shift


def _decoy_shift_per_rung(kind: str, magnitude_ref: float, order, axis: str,
                          rng: np.random.Generator) -> Dict[float, Callable]:
    """One `decoy_shift` callable per rung, magnitude scaled by that rung's
    own `rung_shrink` -- the same quantity `tau` itself scales by, so the
    decoy sits at a consistent multiple of tau's own scale at every rung.
    """
    out = {}
    for ds in order:
        mag = float(magnitude_ref) * ChainStack.rung_shrink(ds, axis)
        if kind == 'fixed':
            step = mag / np.sqrt(2.0)
            out[ds] = decoy_shift_fixed((step, step))
        elif kind == 'random':
            out[ds] = decoy_shift_random(0.5 * mag, 1.5 * mag, rng)
        elif kind == 'rotate':
            # No magnitude to scale by rung here -- a rotation's displacement
            # is set by each anchor's own distance from the set's centroid,
            # not by a chosen step size (`decoy_shift_rotate`'s docstring).
            out[ds] = decoy_shift_rotate((0.0, 2 * np.pi), rng)
        else:
            raise ValueError(f"decoy kind must be 'fixed'/'random'/'rotate', "
                             f"got {kind!r}")
    return out


# ── per-ChainStack: detect, build anchors, probe, curve ──────────────────────

def _one_f_chain(chain, net, *, tile: int, rungs, score_threshold: float,
                 decoy_kind: str, decoy_magnitude: float, rng):
    order = sorted(float(r) for r in rungs)
    stack = FStack.read(chain, tile=tile)
    origins = {d: (float(chain.members[d].meta.x), float(chain.members[d].meta.y))
              for d in order}
    scales = {d: ChainStack.rung_scale(d, 'F') for d in order}
    detections, per_rung, maps = SurvivalProcess.detect_all_rungs(
        stack, net, rungs=order, origins=origins, scales=scales,
        score_threshold=score_threshold)
    # merge_radius_l0=0: F's `order` entries ARE the real level-0 scale, so
    # the default rung_scale (identity) is correct here -- see anchors_of's
    # own docstring and spec.md "同一個點的定義".
    anchors, source_rung = SurvivalProcess.anchors_of(detections, order, 0)
    dist, score, rival, _ = SurvivalProcess.probe_real(
        anchors, per_rung, order, maps, origins=origins, scales=scales,
        nms_radius=net.cfg.nms_radius)
    decoy_shift = _decoy_shift_per_rung(decoy_kind, decoy_magnitude, order,
                                        'F', rng)
    decoy_dist, decoy_score = probe_decoy(anchors, per_rung, order, decoy_shift)
    return order, anchors, source_rung, dist, score, decoy_dist, decoy_score


def _one_r_tile(corpus, tile_of, net, *, tile: int, rungs,
                score_threshold: float, decoy_kind: str,
                decoy_magnitude: float, rng):
    """Same shape as `_one_f_chain`: `RStack.from_own(...)[i]` degrades one
    real tile into `Dict[ds, image]`, exactly what `FStack.read` returns.
    The only difference is the origin -- 'R''s footprint never moves, so
    every rung shares the SAME corner (the tile's own `(x, y)`), and
    `rung_scale('R', ...) == 1.0` at every rung.
    """
    order = sorted(float(r) for r in rungs)
    image = ChainStack._read_store_tile(corpus, tile_of, int(tile))
    stack = RStack.from_tile(image, float(tile_of.meta.ds), order, tile=tile)
    origin = (float(tile_of.meta.x), float(tile_of.meta.y))
    origins = {d: origin for d in order}
    scales = {d: ChainStack.rung_scale(d, 'R') for d in order}
    detections, per_rung, maps = SurvivalProcess.detect_all_rungs(
        stack, net, rungs=order, origins=origins, scales=scales,
        score_threshold=score_threshold)
    # merge_radius_l0=net.cfg.nms_radius, NOT 0: R has no downsampling
    # quantisation (rung_scale('R', ds) is 1.0 at every rung -- R's own
    # rung labels are degrade_resolution amounts relative to whichever real
    # WSI pyramid level the base image happened to come from, not scale
    # factors), so the quantisation term always collapses to 0 -- unlike
    # F/C, nothing else is left to absorb the actual disagreement source
    # (degrade_resolution's own blur shifting a peak) if base is ALSO 0.
    # base stays the pre-redesign value; only rung_scale is overridden
    # (spec.md "同一個點的定義"). The labels still separate
    # R's own rungs from each other; only the added slop collapses to 0.
    anchors, source_rung = SurvivalProcess.anchors_of(
        detections, order, net.cfg.nms_radius, rung_scale=lambda ds: 1.0)
    dist, score, rival, _ = SurvivalProcess.probe_real(
        anchors, per_rung, order, maps, origins=origins, scales=scales,
        nms_radius=net.cfg.nms_radius)
    decoy_shift = _decoy_shift_per_rung(decoy_kind, decoy_magnitude, order,
                                        'R', rng)
    decoy_dist, decoy_score = probe_decoy(anchors, per_rung, order, decoy_shift)
    return order, anchors, source_rung, dist, score, decoy_dist, decoy_score


def _one_f_chain_probability_map(chain, net, *, tile: int, rungs,
                                 score_threshold: float, decoy_kind: str,
                                 decoy_magnitude: float, rng):
    """Candidates 1/2's own version of `_one_f_chain` -- SEPARATE function,
    `_one_f_chain` itself untouched. SIMPLER than the C-axis version:
    `detect_all_rungs` already returns `maps[ds]` unconditionally (no
    `keep_maps` opt-in needed, F only ever has one tile per rung so there is
    nothing to stitch -- `AliveCandidates.assemble_generation_map` is not
    needed at all here).

    `origins`/`scales` ARE `footprint_origin`/`rung_scale`
    (`_probability_map_alive`'s own docstring on why these must be per-rung,
    not one shared value) -- the SAME dicts `probe_real` already needed,
    not a second computation of the same thing.
    """
    order = sorted(float(r) for r in rungs)
    stack = FStack.read(chain, tile=tile)
    origins = {d: (float(chain.members[d].meta.x), float(chain.members[d].meta.y))
              for d in order}
    scales = {d: ChainStack.rung_scale(d, 'F') for d in order}
    detections, per_rung, maps = SurvivalProcess.detect_all_rungs(
        stack, net, rungs=order, origins=origins, scales=scales,
        score_threshold=score_threshold)
    # source rung discarded -- this path has no offset_quantiles_of
    # equivalent to feed it to (main()'s own comment on why).
    anchors, _ = SurvivalProcess.anchors_of(detections, order, 0)
    combined_maps = {ds: maps[ds] for ds in order}
    decoy_shift = _decoy_shift_per_rung(decoy_kind, decoy_magnitude, order,
                                        'F', rng)
    return order, anchors, combined_maps, origins, scales, decoy_shift


def _one_r_tile_probability_map(corpus, tile_of, net, *, tile: int,
                                rungs, score_threshold: float,
                                decoy_kind: str, decoy_magnitude: float, rng):
    """Candidates 1/2's own version of `_one_r_tile` -- SEPARATE function,
    `_one_r_tile` itself untouched. Same shape as `_one_f_chain_probability_
    map`; the only difference (same one `_one_r_tile` itself has) is that
    `origins` is one constant repeated and `scales` is `1.0` at every rung
    (`rung_scale('R', ds) == 1.0`, spec.md "同一個點的定義") -- `anchors_of`
    needs the `rung_scale=lambda ds: 1.0` override for the same reason
    `_one_r_tile` does.
    """
    order = sorted(float(r) for r in rungs)
    image = ChainStack._read_store_tile(corpus, tile_of, int(tile))
    stack = RStack.from_tile(image, float(tile_of.meta.ds), order, tile=tile)
    origin = (float(tile_of.meta.x), float(tile_of.meta.y))
    origins = {d: origin for d in order}
    scales = {d: ChainStack.rung_scale(d, 'R') for d in order}
    detections, per_rung, maps = SurvivalProcess.detect_all_rungs(
        stack, net, rungs=order, origins=origins, scales=scales,
        score_threshold=score_threshold)
    # source rung discarded -- this path has no offset_quantiles_of
    # equivalent to feed it to (main()'s own comment on why).
    anchors, _ = SurvivalProcess.anchors_of(
        detections, order, net.cfg.nms_radius, rung_scale=lambda ds: 1.0)
    combined_maps = {ds: maps[ds] for ds in order}
    decoy_shift = _decoy_shift_per_rung(decoy_kind, decoy_magnitude, order,
                                        'R', rng)
    return order, anchors, combined_maps, origins, scales, decoy_shift


def _one_c_tree(mother, mother_image, groups_by_ds, images_by_ds, net, *,
               c_rungs, score_threshold: float,
               decoy_kind: str, decoy_magnitude: float, rng):
    order = sorted(float(r) for r in c_rungs)
    per_rung_tiles, per_rung = SurvivalProcess.detect_all_generations(
        mother, mother_image, groups_by_ds, images_by_ds, net,
        score_threshold=score_threshold)
    anchors, source_rung = SurvivalProcess.anchors_of_generations(
        per_rung_tiles, order, net.cfg.nms_radius)
    dist, score, rival, _ = SurvivalProcess.probe_real(anchors, per_rung, order)
    decoy_shift = _decoy_shift_per_rung(decoy_kind, decoy_magnitude, order,
                                        'C', rng)
    decoy_dist, decoy_score = probe_decoy(anchors, per_rung, order, decoy_shift)
    return order, anchors, source_rung, dist, score, decoy_dist, decoy_score


def _one_c_tree_probability_map(mother, mother_image, groups_by_ds, images_by_ds,
                                net, *, c_rungs, score_threshold: float,
                                map_overlap_mode: str, decoy_kind: str,
                                decoy_magnitude: float, rng):
    """Candidates 1/2's own version of `_one_c_tree` -- SEPARATE function,
    `_one_c_tree` itself untouched (zero regression risk to baseline/
    exp_decay). Builds the per-rung probability fields
    (`AliveCandidates.assemble_generation_map`) instead of `probe_real`'s
    `dist`/`score`, because that is what `probability_map_curve`/
    `probability_map_pattern_curve` need instead.

    `footprint_origin`/`footprint_shape`: every C-tree generation covers
    EXACTLY the mother tile's own rectangle (spec.md 3.2's `C` axis
    section) -- `mother.x, mother.y` (constant across rungs) and
    `mother.size_px` (the whole footprint's level-0 span, also constant)
    are already on the `mother` `PatchInfo`, not re-derived here. Still
    returned as a PER-RUNG dict (one value repeated), not a bare tuple --
    `_probability_map_alive` takes the same shape from every axis (F's own
    origin genuinely does vary per rung; C's happens not to, but the caller
    should not have to know that).
    `map_overlap_mode` is `assemble_generation_map`'s own union/intersection
    (probability-field max/min) -- unrelated to `anchors_of_generations`' point-list overlap handling.
    """
    order = sorted(float(r) for r in c_rungs)
    per_rung_tiles, per_rung, per_rung_maps = SurvivalProcess.detect_all_generations(
        mother, mother_image, groups_by_ds, images_by_ds, net,
        score_threshold=score_threshold, keep_maps=True)
    # source rung discarded -- this path has no offset_quantiles_of
    # equivalent to feed it to (main()'s own comment on why).
    anchors, _ = SurvivalProcess.anchors_of_generations(
        per_rung_tiles, order, net.cfg.nms_radius)
    mother_origin = (float(mother.x), float(mother.y))
    combined_maps = {
        ds: AliveCandidates.assemble_generation_map(
            *per_rung_maps[ds], scale=ds, footprint_origin=mother_origin,
            footprint_shape=(round(mother.size_px / ds),
                            round(mother.size_px / ds)),
            overlap_mode=map_overlap_mode)
        for ds in order}
    footprint_origin = {ds: mother_origin for ds in order}
    rung_scale = {ds: ds for ds in order}       # C: label IS the real scale
    decoy_shift = _decoy_shift_per_rung(decoy_kind, decoy_magnitude, order,
                                        'C', rng)
    return order, anchors, combined_maps, footprint_origin, rung_scale, decoy_shift


def _apply_merge_radius_2nd(anchors, source_rung, dist, score, decoy_dist,
                            decoy_score, radius: float):
    """Re-merge the anchor list at `radius` (module docstring: a
    sensitivity check, not a value to tune) and slice every per-anchor
    column the same way -- `source_rung`/`dist`/`score`/`decoy_dist`/
    `decoy_score` are all one row per anchor (`source_rung` is `[N]`, the
    rest `[N, L]`), so one index array applies to all five. `source_rung`
    has to be carried through this re-merge like the others: this can drop
    anchors, and `offset_quantiles_of`'s self-match exclusion needs
    `source_rung[i]` to still name the SAME anchor `dist[i]` does.
    `radius <= 0` is a no-op (`merge_anchors` itself returns every index).
    """
    keep = merge_anchors(anchors, radius, priority=score.max(axis=1)
                        if len(score) else None)
    return (anchors[keep], source_rung[keep], dist[keep], score[keep],
            decoy_dist[keep], decoy_score[keep])


# ── plotting ──────────────────────────────────────────────────────────────────

def _colours(rungs):
    return plt.cm.viridis(np.linspace(0.0, 0.9, len(rungs)))


def _plot_alpha_curves(agg: Dict[str, np.ndarray], title: str, out_path: str):
    """1D, one figure three panels: (match, decoy) / gap / margin, each
    against alpha, one line per rung, mean ± std shaded.
    """
    rungs, alphas = agg['rungs'], agg['alphas']
    colours = _colours(rungs)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for j, (ds, colour) in enumerate(zip(rungs, colours)):
        m, ms = agg['match_rate_mean'][j], agg['match_rate_std'][j]
        d = agg['decoy_rate_mean'][j]
        axes[0].plot(alphas, m, '-o', color=colour, markersize=3,
                    label=f'ds {ds:g}')
        axes[0].fill_between(alphas, m - ms, m + ms, color=colour, alpha=0.15)
        axes[0].plot(alphas, d, '--', color=colour, linewidth=1, alpha=0.6)

        g, gs = agg['gap_mean'][j], agg['gap_std'][j]
        axes[1].plot(alphas, g, '-o', color=colour, markersize=3)
        axes[1].fill_between(alphas, g - gs, g + gs, color=colour, alpha=0.15)

        mar = agg['margin_mean'][j]
        axes[2].plot(alphas, mar, '-o', color=colour, markersize=3)

    axes[0].set_title('match (solid) / decoy (dashed)')
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_title('gap = match - decoy')
    axes[1].axhline(0.0, color='0.5', linewidth=0.8)
    axes[2].set_title('margin = match / decoy')
    axes[2].set_yscale('log')
    axes[2].axhline(1.0, color='0.5', linewidth=0.8)
    for ax in axes:
        ax.set_xlabel('alpha  (tau = alpha * ds level-0 px)')
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_heatmaps(agg: Dict[str, np.ndarray],
                   offset_agg: Dict[str, np.ndarray], quantiles,
                   tau_floor: float, title: str, out_path: str):
    """2D: y=ds, x=alpha, for gap and margin (log color scale); a third
    panel for `offset_quantiles` -- which has no alpha axis at all (it is a
    property of the data, not of tau) -- shown as one line per quantile
    against ds, with the alpha sweep overlaid as a fan of `tau = max(tau_floor,
    alpha*ds)` lines (one per alpha, straight through the origin). This is
    the offset-vs-alpha check plan.md's alpha-calibration design calls for:
    a tau line sitting BELOW a quantile's curve at some ds means that
    fraction of real matches gets declared dead by the threshold alone, not
    by an absent detection -- readable directly as "does this alpha's line
    dip under the q0.5/q0.9/q0.99 curve" rather than inferred from match_rate.
    """
    rungs, alphas = agg['rungs'], agg['alphas']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    im0 = axes[0].imshow(agg['gap_mean'], aspect='auto', origin='lower',
                         extent=(alphas[0], alphas[-1], 0, len(rungs)),
                         cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('gap')
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(np.log(np.maximum(agg['margin_mean'], 1e-9)),
                         aspect='auto', origin='lower',
                         extent=(alphas[0], alphas[-1], 0, len(rungs)),
                         cmap='viridis')
    axes[1].set_title('log(margin)')
    fig.colorbar(im1, ax=axes[1])

    for ax in (axes[0], axes[1]):
        ax.set_yticks(np.arange(len(rungs)) + 0.5)
        ax.set_yticklabels([f'{d:g}' for d in rungs])
        ax.set_xlabel('alpha')
        ax.set_ylabel('ds')

    ds_line = np.asarray(rungs, dtype=np.float64)
    for k, alpha in enumerate(alphas):
        axes[2].plot(ds_line, np.maximum(tau_floor, alpha * ds_line), '-',
                    color='0.75', lw=0.6, zorder=1)
    axes[2].plot([], [], '-', color='0.75', lw=1.2,
                label=f'tau, alpha {alphas[0]:g}..{alphas[-1]:g}')

    for k, q in enumerate(quantiles):
        axes[2].plot(rungs, offset_agg['offset_quantile_mean'][:, k], '-o',
                    color=f'C{k}', lw=2, zorder=2, label=f'q{q:g}')
    axes[2].set_title('real offset quantiles vs alpha-swept tau lines')
    axes[2].set_xlabel('ds')
    axes[2].set_ylabel('offset / tau (level-0 px)')
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _stacked_pattern_bars(ax, alphas: np.ndarray, values: np.ndarray,
                          labels: Sequence[str]):
    """100%-stacked bar, one bar per alpha, six segments in `PATTERNS` order
    -- each segment's height IS the fraction, read directly off the y axis,
    which a heatmap's colour never quite lets you do. Same `PATTERN_COLORS`
    colour a pattern gets in the excess line panels below, so a colour means
    the same pattern in every panel of this figure.
    """
    width = 0.8 * float(np.min(np.diff(alphas))) if len(alphas) > 1 else 0.8
    bottom = np.zeros(len(alphas))
    for p, label in enumerate(labels):
        ax.bar(alphas, values[p], width=width, bottom=bottom,
              color=PATTERN_COLORS[PATTERNS[p]], label=label)
        bottom += values[p]
    ax.set_ylim(0, 1)
    ax.set_xlabel('alpha')
    ax.set_ylabel('fraction')
    ax.legend(fontsize=6, ncol=2)


def _plot_patterns(agg: Dict[str, np.ndarray], title: str, out_path: str):
    """Five panels: measured six-pattern composition vs alpha (100%-stacked
    bar, real and decoy side by side), NullModel excess vs alpha (six SIGNED
    lines, real and decoy side by side, shared y-range so a glance says
    which is bigger), and a total-variation summary (`0.5*sum(|excess|)`,
    unsigned, real vs decoy on one axis -- a navigation aid for which alpha
    departs from the null furthest overall, not a substitute for the
    six-line panels it sits next to).

    `gap` (AlphaSelectionNotes.md Q7), cross-checked against the
    keypoint-precision / coarse-rung-error estimate, stays the primary way
    this project picks alpha; whether the excess/NullModel view here ever
    becomes a further criterion is still open, not settled either way.
    """
    alphas = agg['alphas']
    labels = [ASCII_NAMES[p] for p in PATTERNS]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    ax_dist_real, ax_dist_decoy, ax_tvd = axes[0]
    ax_exc_real, ax_exc_decoy, ax_blank = axes[1]
    ax_blank.axis('off')

    _stacked_pattern_bars(ax_dist_real, alphas, agg['measured_real_mean'],
                          labels)
    ax_dist_real.set_title('measured, real')

    _stacked_pattern_bars(ax_dist_decoy, alphas, agg['measured_decoy_mean'],
                          labels)
    ax_dist_decoy.set_title('measured, decoy')

    excess_real = agg['excess_real_mean']
    excess_decoy = agg['excess_decoy_mean']
    y_lo = min(np.nanmin(excess_real), np.nanmin(excess_decoy))
    y_hi = max(np.nanmax(excess_real), np.nanmax(excess_decoy))

    for p, label in enumerate(labels):
        color = PATTERN_COLORS[PATTERNS[p]]
        ax_exc_real.plot(alphas, excess_real[p], '-o', ms=3,
                         color=color, label=label)
        ax_exc_decoy.plot(alphas, excess_decoy[p], '-o', ms=3,
                          color=color, label=label)
    for ax, name in ((ax_exc_real, 'real'), (ax_exc_decoy, 'decoy')):
        ax.axhline(0.0, color='0.5', lw=0.8)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel('alpha')
        ax.set_ylabel('excess (measured - null)')
        ax.set_title(f'excess vs alpha, {name}')
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)

    tvd_real = 0.5 * np.abs(excess_real).sum(axis=0)
    tvd_decoy = 0.5 * np.abs(excess_decoy).sum(axis=0)
    ax_tvd.plot(alphas, tvd_real, '-o', color='crimson', label='real')
    ax_tvd.plot(alphas, tvd_decoy, '-o', color='steelblue', label='decoy')
    ax_tvd.set_xlabel('alpha')
    ax_tvd.set_ylabel('total variation vs null (unsigned)')
    ax_tvd.set_title('total variation summary')
    ax_tvd.legend(fontsize=7)
    ax_tvd.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ── csv ───────────────────────────────────────────────────────────────────────

def _write_curve_csv(bucket: Dict[str, np.ndarray], path: str):
    """One row per (ds, alpha) -- the numbers `_plot_alpha_curves`/
    `_plot_heatmaps` draw, kept so a specific value can be looked up rather
    than measured off a figure. `n_chainstacks` is repeated on every row (a
    scalar, not something with its own ds/alpha axis) so `--from-csv` can
    reconstruct the figure title without a separate sidecar file.
    """
    rungs, alphas = bucket['rungs'], bucket['alphas']
    n = bucket['n_chainstacks']
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['ds', 'alpha', 'n_chainstacks', 'match_rate_mean',
                        'match_rate_std', 'decoy_rate_mean', 'decoy_rate_std',
                        'gap_mean', 'gap_std', 'margin_mean', 'margin_std'])
        for j, ds in enumerate(rungs):
            for k, alpha in enumerate(alphas):
                writer.writerow([
                    f'{ds:g}', f'{alpha:g}', n,
                    bucket['match_rate_mean'][j, k], bucket['match_rate_std'][j, k],
                    bucket['decoy_rate_mean'][j, k], bucket['decoy_rate_std'][j, k],
                    bucket['gap_mean'][j, k], bucket['gap_std'][j, k],
                    bucket['margin_mean'][j, k], bucket['margin_std'][j, k]])


_CURVE_KEYS = ('match_rate', 'decoy_rate', 'gap', 'margin')


def _read_curve_csv(path: str) -> Dict[str, np.ndarray]:
    """Inverse of `_write_curve_csv` -- pivots the (ds, alpha) rows back
    into the `[L, K]` arrays `_plot_alpha_curves`/`_plot_heatmaps` expect.
    """
    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    rungs = sorted({float(r['ds']) for r in rows})
    alphas = sorted({float(r['alpha']) for r in rows})
    j_of = {ds: j for j, ds in enumerate(rungs)}
    k_of = {a: k for k, a in enumerate(alphas)}
    out = {f'{key}_{stat}': np.full((len(rungs), len(alphas)), np.nan)
          for key in _CURVE_KEYS for stat in ('mean', 'std')}
    n_chainstacks = 0
    for row in rows:
        j, k = j_of[float(row['ds'])], k_of[float(row['alpha'])]
        for key in _CURVE_KEYS:
            out[f'{key}_mean'][j, k] = float(row[f'{key}_mean'])
            out[f'{key}_std'][j, k] = float(row[f'{key}_std'])
        n_chainstacks = int(row['n_chainstacks'])
    out['rungs'] = np.asarray(rungs, np.float64)
    out['alphas'] = np.asarray(alphas, np.float64)
    out['n_chainstacks'] = n_chainstacks
    return out


_PATTERN_KEYS = ('measured_real', 'measured_decoy', 'null_real',
                 'null_decoy', 'excess_real', 'excess_decoy')


def _write_pattern_csv(bucket: Dict[str, np.ndarray], path: str):
    """One row per (pattern, alpha) -- `pattern_curve`'s six numbers
    (measured/null/excess, real and decoy), the numbers `_plot_patterns`
    draws.
    """
    alphas = bucket['alphas']
    n = bucket['n_chainstacks']
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['pattern', 'alpha', 'n_chainstacks']
                        + [f'{k}_mean' for k in _PATTERN_KEYS]
                        + [f'{k}_std' for k in _PATTERN_KEYS])
        for p, pattern in enumerate(PATTERNS):
            for k, alpha in enumerate(alphas):
                writer.writerow(
                    [pattern, f'{alpha:g}', n]
                    + [bucket[f'{key}_mean'][p, k] for key in _PATTERN_KEYS]
                    + [bucket[f'{key}_std'][p, k] for key in _PATTERN_KEYS])


def _read_pattern_csv(path: str) -> Dict[str, np.ndarray]:
    """Inverse of `_write_pattern_csv` -- pivots back into `[6, K]` arrays,
    row order fixed to `Patterns.PATTERNS` (not the CSV's own row order,
    which is already that order but this does not rely on it).
    """
    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    alphas = sorted({float(r['alpha']) for r in rows})
    k_of = {a: k for k, a in enumerate(alphas)}
    p_of = {p: i for i, p in enumerate(PATTERNS)}
    out = {f'{key}_{stat}': np.full((len(PATTERNS), len(alphas)), np.nan)
          for key in _PATTERN_KEYS for stat in ('mean', 'std')}
    n_chainstacks = 0
    for row in rows:
        p, k = p_of[row['pattern']], k_of[float(row['alpha'])]
        for key in _PATTERN_KEYS:
            out[f'{key}_mean'][p, k] = float(row[f'{key}_mean'])
            out[f'{key}_std'][p, k] = float(row[f'{key}_std'])
        n_chainstacks = int(row['n_chainstacks'])
    out['alphas'] = np.asarray(alphas, np.float64)
    out['n_chainstacks'] = n_chainstacks
    return out


def _write_offset_csv(offset_bucket: Dict[str, np.ndarray], rungs,
                      quantiles, path: str):
    """One row per (ds, quantile)."""
    n = offset_bucket['n_chainstacks']
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['ds', 'quantile', 'n_chainstacks', 'offset_mean',
                        'offset_std'])
        for j, ds in enumerate(rungs):
            for k, q in enumerate(quantiles):
                writer.writerow([
                    f'{ds:g}', f'{q:g}', n,
                    offset_bucket['offset_quantile_mean'][j, k],
                    offset_bucket['offset_quantile_std'][j, k]])


def _read_offset_csv(path: str) -> Dict[str, np.ndarray]:
    """Inverse of `_write_offset_csv`."""
    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    rungs = sorted({float(r['ds']) for r in rows})
    quantiles = sorted({float(r['quantile']) for r in rows})
    j_of = {ds: j for j, ds in enumerate(rungs)}
    k_of = {q: k for k, q in enumerate(quantiles)}
    mean = np.full((len(rungs), len(quantiles)), np.nan)
    std = np.full((len(rungs), len(quantiles)), np.nan)
    n_chainstacks = 0
    for row in rows:
        j, k = j_of[float(row['ds'])], k_of[float(row['quantile'])]
        mean[j, k] = float(row['offset_mean'])
        std[j, k] = float(row['offset_std'])
        n_chainstacks = int(row['n_chainstacks'])
    return {'offset_quantile_mean': mean, 'offset_quantile_std': std,
           'n_chainstacks': n_chainstacks, 'rungs': np.asarray(rungs),
           'quantiles': np.asarray(quantiles)}


def _run_from_csv(csv_dir: str, axes, quantiles, tau_floor: float,
                  fig_dir: str) -> int:
    """`--from-csv` -- redraws every figure `main()` would have produced,
    reading the three CSVs back instead of recomputing anything. No
    checkpoint, no slide, no torch work happens on this path.
    """
    status = 0
    for axis in axes:
        curve_path = os.path.join(csv_dir, f'alpha_curve_{axis}.csv')
        offset_path = os.path.join(csv_dir, f'offset_quantiles_{axis}.csv')
        pattern_path = os.path.join(csv_dir, f'pattern_curve_{axis}.csv')
        if not os.path.exists(curve_path):
            print(f'  {axis}: no {curve_path}, skipping', flush=True)
            status = 1
            continue

        bucket = _read_curve_csv(curve_path)
        title = f"'{axis}' axis, {bucket['n_chainstacks']} ChainStacks"
        print(f'  {axis}: {bucket["n_chainstacks"]} ChainStacks (from CSV)',
             flush=True)

        _plot_alpha_curves(bucket, title,
                          os.path.join(fig_dir, f'alpha_curves_{axis}.png'))

        if os.path.exists(offset_path):
            offset_bucket = _read_offset_csv(offset_path)
            _plot_heatmaps(bucket, offset_bucket, quantiles, tau_floor,
                          title, os.path.join(fig_dir, f'heatmaps_{axis}.png'))
        else:
            print(f'  {axis}: no {offset_path}, skipping heatmaps_{axis}.png',
                 flush=True)

        if os.path.exists(pattern_path):
            pattern_bucket = _read_pattern_csv(pattern_path)
            _plot_patterns(pattern_bucket, title,
                          os.path.join(fig_dir, f'patterns_{axis}.png'))
        else:
            print(f'  {axis}: no {pattern_path}, skipping patterns_{axis}.png',
                 flush=True)

    print(f'\nfigures -> {fig_dir}/', flush=True)
    return status


def _make_alive_fn(method: str, *, combined_threshold: Optional[float]
                   ) -> Optional[Callable[..., np.ndarray]]:
    """`None` for 'baseline' -- lets `alpha_curve`/`pattern_curve` keep their
    own fast built-in path (`AliveCandidates.alive_absolute_dual_threshold`
    IS that path, just not called through this indirection). Otherwise a
    closure with the uniform `(score, dist, tau)` signature those two
    functions call, binding whichever extra parameter the selected
    `AliveCandidates` method needs -- `AlphaCalibration.py` never has to
    know `combined_threshold` exists.

    ONLY 'baseline'/'exp_decay' GO THROUGH THIS. Candidates 1/2
    ('probability_map'/'scale_extremum') never call this function -- they
    do not fit the `(score, dist, tau)` shape at all (they ignore `score`/
    `dist` and re-probe the probability field instead), so `main()` routes
    them to `probability_map_curve`/`probability_map_pattern_curve`
    directly (`AlphaCalibration.py`'s own module comment above those two
    explains why a shared `alive_fn` could not do this).
    """
    if method == 'baseline':
        return None
    if method == 'exp_decay':
        def _fn(score, dist, tau):
            return AliveCandidates.alive_exp_decay_joint_score(
                score, dist, tau=tau, combined_threshold=combined_threshold)
        return _fn
    raise ValueError(f'unknown --alive-method {method!r} for _make_alive_fn '
                     f'-- probability_map/scale_extremum do not use this '
                     f'function at all, see its own docstring')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wsi-name', default=None,
                    help='look up --wsi/--wsi-stem via AccessDatasets -- '
                         'takes priority over setting them directly')
    ap.add_argument('--wsi', default=None,
                    help="only needed when 'C' is in --axes -- 'C''s "
                         "descendants are read fresh from the slide; F's "
                         "own read (a store) and R's own read (a store, "
                         "degraded) never touch it")
    ap.add_argument('--wsi-stem', default=None)
    ap.add_argument('--checkpoint', default=None,
                    help='required unless --from-csv')
    ap.add_argument('--from-csv', default=None,
                    help='skip detection/computation entirely and redraw '
                         'figures from a previous run\'s CSVs (the same '
                         'directory alpha_curve_<axis>.csv etc. were '
                         'written to) -- no checkpoint, no slide, no torch '
                         'work happens on this path')
    ap.add_argument('--alive-method',
                    choices=('baseline', 'exp_decay', 'probability_map',
                            'scale_extremum'),
                    default='baseline',
                    help="'baseline' = AliveCandidates.alive_absolute_"
                         "dual_threshold (Patterns.alive_from, production); "
                         "'exp_decay' = AliveCandidates."
                         "alive_exp_decay_joint_score (candidate 3) -- "
                         "requires --combined-threshold; 'probability_map' "
                         "(candidate 1) / 'scale_extremum' (candidate 2) -- "
                         "F/R/C all supported, re-probes the raw probability "
                         "field every alpha instead of thresholding dist/"
                         "score; --map-overlap-mode ('C' only, main+overlap "
                         "tile fold) and (scale_extremum only) "
                         "--extremum-margin apply")
    ap.add_argument('--combined-threshold', type=float, default=None,
                    help='required when --alive-method exp_decay -- operates '
                         'on score*exp(-dist/tau), a different scale from '
                         '--score-threshold, not filled in by default on '
                         'purpose (ClaudeRules 8: calibration, not guessing)')
    ap.add_argument('--map-overlap-mode', choices=('union', 'intersection'),
                    default='union',
                    help="'probability_map'/'scale_extremum' only -- "
                         "AliveCandidates.assemble_generation_map's own "
                         "main/overlap probability-field fold (max/min); "
                         "unrelated to the anchors_of_generations point-list "
                         "overlap handling (removed 2026-09-11)")
    ap.add_argument('--extremum-margin', type=float, default=0.0,
                    help="'scale_extremum' only -- how far a neighbouring "
                         "rung's peak_value is allowed to exceed this "
                         "rung's before it counts as beaten "
                         "(AliveCandidates.alive_scale_local_extremum)")
    ap.add_argument('--sample-step', type=float, default=0.5,
                    help="'probability_map'/'scale_extremum' only -- "
                         "probe_via_probability_map's own sub-pixel search "
                         "grid spacing, in each rung's own pixel units; its "
                         "own docstring calls this 'a real accuracy/speed "
                         "knob, not yet tuned' -- finer catches a sharper "
                         "peak more precisely at more compute per anchor "
                         "per rung")
    add_corpus_args(ap, corpus=False)
    prepare_chain_stack.add_axis_corpus_args(ap)
    ap.add_argument('--rungs', type=float, nargs='+',
                    default=[1.0, 2.0, 4.0, 8.0, 16.0])
    ap.add_argument('--c-rungs', type=float, nargs='+',
                    default=[1.0, 2.0, 4.0, 8.0, 16.0])
    ap.add_argument('--axes', nargs='+', default=['C'], choices=['F', 'R', 'C'])
    ap.add_argument('--score-threshold', type=float, default=None,
                    help='defaults to the checkpoint\'s own '
                         'cfg.detection_threshold -- the value its labels '
                         'were cut at, not a borrowed constant')
    ap.add_argument('--alphas', type=float, nargs=3, default=[0.5, 0.25, 4.0],
                    metavar=('MIN', 'STEP', 'MAX'))
    ap.add_argument('--tau-floor', type=float, default=0.0)
    ap.add_argument('--merge-radius-2nd', type=float, default=0.0)
    ap.add_argument('--decoy-kind', choices=('fixed', 'random', 'rotate'),
                    default='fixed')
    ap.add_argument('--decoy-magnitude', type=float, default=8.0,
                    help='decoy shift, in units of ds -- matches how tau '
                         'itself scales')
    ap.add_argument('--decoy-seed', type=int, default=0)
    add_chainstack_args(ap, 'SurvivalAlphaAnalysis', on=True)
    ap.add_argument('--quantiles', type=float, nargs='+',
                    default=[0.5, 0.9, 0.99])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.from_csv:
        out_dir = args.out or job_result_dir('SurvivalAlphaAnalysis')
        fig_dir = os.path.join(out_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        return _run_from_csv(args.from_csv, args.axes, args.quantiles,
                             args.tau_floor, fig_dir)

    if not args.checkpoint:
        ap.error('--checkpoint is required unless --from-csv')
    if args.alive_method == 'exp_decay' and args.combined_threshold is None:
        ap.error('--combined-threshold is required when '
                 '--alive-method exp_decay')
    probability_map_method = args.alive_method in ('probability_map',
                                                    'scale_extremum')
    if probability_map_method and args.merge_radius_2nd > 0.0:
        ap.error(f"--merge-radius-2nd is not supported with --alive-method "
                 f"{args.alive_method!r} yet -- candidates 1/2 have no "
                 f"score to rank anchors by at merge time "
                 f"(_apply_merge_radius_2nd needs score.max(axis=1))")

    if args.wsi_name:
        entry = AccessDatasets.locate(args.wsi_name)
        args.wsi, args.wsi_stem = entry.path, entry.name
    if not args.wsi_stem:
        ap.error('--wsi-stem (or --wsi-name) is required')

    wsi_needed = 'C' in args.axes
    if wsi_needed and not args.wsi:
        ap.error("--wsi (or --wsi-name) is required when 'C' is in --axes")

    out_dir = args.out or job_result_dir('SurvivalAlphaAnalysis')
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    rng = np.random.default_rng(args.decoy_seed)

    alpha_min, alpha_step, alpha_max = args.alphas
    alphas = np.arange(alpha_min, alpha_max + alpha_step / 2, alpha_step)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, identity, _ = load_arm(args.checkpoint, device)
    threshold_from_cfg = args.score_threshold is None
    if threshold_from_cfg:
        args.score_threshold = float(net.cfg.detection_threshold)
    print(f'detector {args.checkpoint}  {net.identity_id()}', flush=True)
    print(f'score threshold {args.score_threshold:g}'
         f'{" (from cfg.detection_threshold)" if threshold_from_cfg else ""}',
         flush=True)
    print(f'alive method {args.alive_method}'
         f'{f"  combined-threshold {args.combined_threshold:g}" if args.alive_method == "exp_decay" else ""}'
         f'{f"  map-overlap-mode {args.map_overlap_mode}  sample-step {args.sample_step:g}" if probability_map_method else ""}'
         f'{f"  extremum-margin {args.extremum_margin:g}" if args.alive_method == "scale_extremum" else ""}',
         flush=True)
    # probability_map/scale_extremum never call _make_alive_fn -- they do not
    # fit its (score, dist, tau) shape at all (_make_alive_fn's own
    # docstring); alive_fn stays None and is simply not used on that path.
    alive_fn = (None if probability_map_method else
               _make_alive_fn(args.alive_method,
                             combined_threshold=args.combined_threshold))

    corpus = {axis: prepare_chain_stack.axis_corpus(axis, args)
              for axis in args.axes}

    slide_ctx = SafeSlide(args.wsi) if wsi_needed else contextlib.nullcontext(None)
    with slide_ctx as wsi:
        for axis in args.axes:
            print(f'\n==== bucket {axis} ====', flush=True)
            curves = []
            offsets = []
            pattern_curves = []
            order = None

            if axis == 'F':
                own = FStack.from_own(corpus['F'], args.wsi_stem,
                                      tile=args.tile, rungs=args.rungs)
                chainstacks = [(own.chains[cid],) for cid in own]
                if probability_map_method:
                    per_chainstack = lambda args_tuple: _one_f_chain_probability_map(  # noqa: E731
                        *args_tuple, net, tile=args.tile, rungs=args.rungs,
                        score_threshold=args.score_threshold,
                        decoy_kind=args.decoy_kind,
                        decoy_magnitude=args.decoy_magnitude, rng=rng)
                else:
                    per_chainstack = lambda args_tuple: _one_f_chain(      # noqa: E731
                        *args_tuple, net, tile=args.tile, rungs=args.rungs,
                        score_threshold=args.score_threshold,
                        decoy_kind=args.decoy_kind,
                        decoy_magnitude=args.decoy_magnitude, rng=rng)

            elif axis == 'C':
                forest = CStack.from_own(
                    corpus['C'], args.wsi_stem, args.c_rungs, wsi,
                    tile=args.tile,
                    cache_root=chainstack_root(args))
                chainstacks = [forest[i] for i in forest]
                if probability_map_method:
                    per_chainstack = lambda args_tuple: _one_c_tree_probability_map(  # noqa: E731
                        *args_tuple, net, c_rungs=args.c_rungs,
                        score_threshold=args.score_threshold,
                        map_overlap_mode=args.map_overlap_mode,
                        decoy_kind=args.decoy_kind,
                        decoy_magnitude=args.decoy_magnitude, rng=rng)
                else:
                    per_chainstack = lambda args_tuple: _one_c_tree(       # noqa: E731
                        *args_tuple, net, c_rungs=args.c_rungs,
                        score_threshold=args.score_threshold,
                        decoy_kind=args.decoy_kind,
                        decoy_magnitude=args.decoy_magnitude, rng=rng)

            else:  # 'R'
                own = RStack.from_own(corpus['R'], args.wsi_stem,
                                      args.rungs, tile=args.tile,
                                      cache_root=None)
                chainstacks = [(own.corpus, t) for t in own.items]
                if probability_map_method:
                    per_chainstack = lambda args_tuple: _one_r_tile_probability_map(  # noqa: E731
                        *args_tuple, net, tile=args.tile, rungs=args.rungs,
                        score_threshold=args.score_threshold,
                        decoy_kind=args.decoy_kind,
                        decoy_magnitude=args.decoy_magnitude, rng=rng)
                else:
                    per_chainstack = lambda args_tuple: _one_r_tile(       # noqa: E731
                        *args_tuple, net, tile=args.tile, rungs=args.rungs,
                        score_threshold=args.score_threshold,
                        decoy_kind=args.decoy_kind,
                        decoy_magnitude=args.decoy_magnitude, rng=rng)

            n_chainstacks = len(chainstacks)
            is_probability_map_axis = probability_map_method
            for i, one in enumerate(chainstacks):
                print(f'  ChainStack {i + 1}/{n_chainstacks}...', flush=True)
                if is_probability_map_axis:
                    order, anchors, combined_maps, footprint_origin, \
                        rung_scale, decoy_shift = per_chainstack(one)
                    print(f'    -> {len(anchors)} anchors', flush=True)
                    # merge_radius_2nd already refused at arg-parse time for
                    # this method (no score to rank anchors by at merge time).
                    curves.append(probability_map_curve(
                        combined_maps, anchors, decoy_shift, rungs=order,
                        alphas=alphas, tau_floor=args.tau_floor,
                        threshold=args.score_threshold,
                        footprint_origin=footprint_origin,
                        rung_scale=rung_scale,
                        kind=args.alive_method,
                        extremum_margin=args.extremum_margin,
                        sample_step=args.sample_step))
                    pattern_curves.append(probability_map_pattern_curve(
                        combined_maps, anchors, decoy_shift, rungs=order,
                        alphas=alphas, tau_floor=args.tau_floor,
                        threshold=args.score_threshold,
                        footprint_origin=footprint_origin,
                        rung_scale=rung_scale,
                        kind=args.alive_method,
                        extremum_margin=args.extremum_margin,
                        sample_step=args.sample_step))
                    # no offset_quantiles_of equivalent -- probability_map/
                    # scale_extremum never compute a `dist` in probe_real's
                    # sense (AlphaCalibration.py's probability_map_curve
                    # module comment); offsets/heatmaps skipped below.
                    continue
                order, anchors, source_rung, dist, score, decoy_dist, \
                    decoy_score = per_chainstack(one)
                print(f'    -> {len(anchors)} anchors', flush=True)
                if args.merge_radius_2nd > 0.0:
                    anchors, source_rung, dist, score, decoy_dist, \
                        decoy_score = _apply_merge_radius_2nd(
                            anchors, source_rung, dist, score, decoy_dist,
                            decoy_score, args.merge_radius_2nd)
                curves.append(alpha_curve(
                    dist, score, decoy_dist, decoy_score, rungs=order,
                    alphas=alphas, tau_floor=args.tau_floor,
                    threshold=args.score_threshold, alive_fn=alive_fn))
                offsets.append(offset_quantiles_of(
                    dist, rungs=order, source_rung=source_rung,
                    quantiles=args.quantiles))
                pattern_curves.append(pattern_curve(
                    dist, score, decoy_dist, decoy_score, rungs=order,
                    alphas=alphas, tau_floor=args.tau_floor,
                    threshold=args.score_threshold, alive_fn=alive_fn))

            if not curves:
                print(f'  0 ChainStacks for {axis}, skipping', flush=True)
                continue

            bucket = aggregate_curves(curves)
            pattern_bucket = aggregate_pattern_curves(pattern_curves)
            print(f'  {bucket["n_chainstacks"]} ChainStacks aggregated', flush=True)

            _write_curve_csv(bucket, os.path.join(out_dir, f'alpha_curve_{axis}.csv'))
            _write_pattern_csv(pattern_bucket,
                              os.path.join(out_dir, f'pattern_curve_{axis}.csv'))

            _plot_alpha_curves(
                bucket, f"'{axis}' axis, {bucket['n_chainstacks']} ChainStacks",
                os.path.join(fig_dir, f'alpha_curves_{axis}.png'))
            _plot_patterns(
                pattern_bucket,
                f"'{axis}' axis, {pattern_bucket['n_chainstacks']} ChainStacks",
                os.path.join(fig_dir, f'patterns_{axis}.png'))

            if is_probability_map_axis:
                print(f'  {axis}: alive-method {args.alive_method} has no '
                     f'offset_quantiles/heatmap equivalent, skipped',
                     flush=True)
                continue

            offset_bucket = aggregate_offset_quantiles(offsets)
            _write_offset_csv(offset_bucket, order, args.quantiles,
                             os.path.join(out_dir, f'offset_quantiles_{axis}.csv'))
            _plot_heatmaps(
                bucket, offset_bucket, args.quantiles, args.tau_floor,
                f"'{axis}' axis, {bucket['n_chainstacks']} ChainStacks",
                os.path.join(fig_dir, f'heatmaps_{axis}.png'))

    print(f'\nfigures -> {fig_dir}/', flush=True)
    print(f'tables  -> {out_dir}/alpha_curve_<axis>.csv, '
         f'offset_quantiles_<axis>.csv, pattern_curve_<axis>.csv', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
