#!/usr/bin/env python3
"""Does stage 1 get better if the KNN runs in the scale subspace instead of 1536D?

Steps 1-2 of FeatureSubspaceDecomposition (bench_feature_axes.py) measured where
mpp lives in a slide's feature space and found it low dimensional: 45-64 of the
1536 directions beat a parallel-analysis null, but 90% of log mpp is explained by
2 to 10 of them, while random subspaces of the same size explain 0.11-0.16 at
r=1. Those are correlations. They do not say whether a nearest-neighbour search
run in that subspace classifies better, worse, or the same -- and that is the
only question with a consequence, because it is the one that would change
`GigaPathKnnEstiMpp`.

What the first run already settled, on BRACS_1228
-------------------------------------------------
Both gates passed (`pinned` exact, `shuffled` 0.378 against a real 0.990), so
the numbers below are evidence rather than plumbing.

    ranking by |corr with log mpp| is dead.  `variance` beat `scale` at every r
        (r=2: 0.933 against 0.794; r=10: 0.997 against 0.966). The mechanism is
        in subspace_knn_selected.csv: after projecting and renormalising, every
        kept direction votes equally, so the rule's second pick -- component 5,
        variance ratio 0.038 -- carries the same weight as PC1 at 0.228. Its
        third pick, component 3, correlates 0.618 with BACKGROUND. The selection
        rule walked straight into the confusion corr_white exists to detect.

    the premise of the whole case was wrong.  "A few thousand tiles in 1536
        dimensions is a badly conditioned place for a nearest-neighbour search"
        -- measured, it is 0.990 on clean tiles. There was no ill-conditioning
        waiting to be fixed.

    the dimension cut was never given a fair test.  On arm B the loss came from
        CENTRING: the `centred` row keeps all 1530 components and still falls
        from 0.680 to 0.400. Everything with a smaller r was measured downstream
        of a break that had nothing to do with r.

So this version adds the setting that isolates it.

Four settings, because "1536 -> r" was two changes at once
---------------------------------------------------------
    production   x                     uncentred, full dimension
    centred      V^T (x - mu)          all components, mean removed
    centred+cut  V_r^T (x - mu)        mean removed AND cut to r
    uncentred    V_r^T x               cut to r, mean KEPT

The last two differ by exactly V_r^T mu, one constant vector added to every
point. Cosine is not translation invariant, so that constant is not cosmetic:
L2-normalised deep features share a large common component pointing along mu,
uncentred cosine is dominated by it, and it is the part a domain gap does not
destroy. Subtracting the REFERENCE mean from a query does not centre the query
-- it shifts it by the wrong vector and leaves a residual where the domain gap
dominates. (If mu happened to be orthogonal to span(V_r) the two would coincide;
whether it is, is measurable rather than assumed.)

`uncentred` uses the eigenvalue ordering, not the correlation ordering, because
the correlation ordering is the thing the first run buried.

Which r components, and the decoys
----------------------------------
    scale       top r by |corr(component, log mpp)|. Kept for continuity with
                steps 1-2, no longer a candidate.
    variance    top r by eigenvalue. Plain PCA.
    random      a uniformly random r-dimensional subspace, averaged over
                RANDOM_DRAWS draws. One draw per r made the decoy unreadable:
                r=2 scored 0.681 and r=3 scored 0.532, which is two rolls of a
                die and not a trend.

Two arms, the cheap one first
-----------------------------
    A  tile -> tile   reference and test are both grid tiles from the same
                      store. No domain gap. Production sits at 0.990 here, so
                      this arm has no headroom left -- it is a gate, not a
                      comparison.

    B  photo -> tile  the query stores: synthesised FoV renders carrying colour
                      temperature, vignetting, distortion, defocus, noise and
                      JPEG. Query tiles carry `fov_id`, so production's
                      median-of-medians is reproduced per FoV. This is the arm
                      with the decision in it.

The query is projected through the reference's basis -- it never gets a PCA of
its own. mu and V are fitted once, on the reference bank, and both sides pass
through them. A second fit would put the two sides in unrelated coordinate
systems: eigenvectors are defined only up to sign and their order can differ, so
"component 1 against component 1" would be the inner product of two unrelated
axes. It would still return a number.

That is also what deployment looks like: `LocaScopePipeline.build()` already
holds this slide's reference tiles, so fitting a per-slide basis there costs one
eigendecomposition and no extra reading.

Why arm A cannot be split at random
-----------------------------------
`ReferenceSampler` lays its grid with `overlap=True`, so a reference store holds
main positions at (256i, 256j) AND overlap positions half a tile off. Those two
share half their pixels. A random split puts a tile and its 50%-overlapping
neighbour on opposite sides, and the nearest neighbour of a test tile is then a
near copy of itself.

So the split is a contiguous band in x, per level: the top quartile by x is
held out. The random split is ALSO reported, because the gap between the two is
the size of the near-duplicate inflation.

Background is confounded with level, on BOTH arms
-------------------------------------------------
Measured from the stores (inspect_feature_store --sampling): the quota sampler's
background fraction rises steeply with level -- median 0.000 at L0 on all seven
slides, 0.288 at L2 on BRACS_1228, 0.626 at L3 on S1151088. An L3 tile covers
64x the level-0 area, so tissue-dense positions do not exist in the numbers the
quota asks for and the allocation falls through to the emptier buckets.

Any direction that separates levels therefore picks up background to the extent
the two are confounded. `--white-max` repeats BOTH arms with every tile above
the threshold dropped from the REFERENCE bank (and, on arm A, from the test set
too). Arm B was left out of this control in the first version on the grounds
that query stores hold no white_frac -- true, and beside the point: the
confound lives on the reference side, which is exactly the side that can be
filtered.

The subset is rebalanced per level after filtering. Skipping that step made the
first run's low-background bank 342/330/191, which moves the chance rate to
0.394 and lets the leading component encode "which level has the most tiles" --
the trap load_slide_balanced's docstring names.

Three gates in front of the run
-------------------------------
A wrong answer here looks like "no subspace improves accuracy", which reads as a
finding rather than a bug, so the gates are cheap and none is a tolerance:

    pinned          the `production` setting must reproduce
                    `KnnClassifier.predict` on the same tensors, exactly.
    full_rank       `uncentred` with ALL components must reproduce `production`
                    exactly. A full-rank projection is an orthogonal rotation
                    and no mean is removed, so cosine cannot have moved. This
                    one is free and it is the only check on the projection
                    arithmetic itself: `centred` is SUPPOSED to differ from
                    production, so a transposed matrix would hide there.
    shuffled        with the level labels permuted, accuracy must fall to
                    chance.

On the sample size of arm B
---------------------------
75 FoVs on BRACS_1228, so a single accuracy carries a standard error near 0.054
and a 20-point gap is visible while a 10-point one is not. The comparison is
PAIRED -- every setting is scored on the same FoVs -- and
subspace_knn_arm_b_fovs.csv records each FoV's prediction under each setting, so
"production right, uncentred wrong" can be counted directly instead of being
inferred from two overlapping intervals. More FoVs cost a re-dump with a GPU and
the slide; that is worth paying only if the verdict comes out a tie.

Outputs, all under result/SubspaceKnn/
--------------------------------------
    subspace_knn_scores.csv        one row per (slide, arm, split, subset, rule, r)
    subspace_knn_arm_b_fovs.csv    one row per (setting, FoV): what it predicted
    subspace_knn_selected.csv      which components each rule picked, and their
                                   correlation with log mpp and with background
    subspace_knn_gates.csv         the three gates, per slide
    subspace_knn_definitions.csv   every name used above, and what it computes
    subspace_knn_accuracy.png      accuracy against r, arm A beside arm B
    subspace_knn_confusion.png     arm B, true level against predicted level
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _directory in ('utilities', 'aiNNModel', 'utilities/test_modules',
                   '1_estimate_query_mpp'):
    _path = str(_ROOT / _directory)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import matplotlib                                                   # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402

import FeatureStore as FeatureStoreModule                           # noqa: E402
from GigaPathFunc import pool_tokens                                # noqa: E402
from GigaPathKnnEstiMpp import KnnClassifier                        # noqa: E402
from _paths import job_result_dir                                   # noqa: E402
from bench_feature_axes import load_slide_balanced                  # noqa: E402

#: Neighbours, fixed at production's value. This bench asks about the space, not
#: about k; sweeping both would make every row answer two questions at once.
K_NEIGHBOURS = 5

#: Dimensions swept. r_90pct measured in step 2 was 5, 2, 5, 2, 10, 6, 7 across
#: the seven slides, so the list brackets the whole range and the per-slide value
#: is read off the same sweep rather than given a column of its own.
R_VALUES = (1, 2, 3, 5, 10, 20, 50)

#: The two r values every table and figure reports at. NOT derived from step 2's
#: r_90pct, which is invalid here twice over: it counted components in the
#: |corr(log mpp)| ORDER, which `variance` has since beaten, and it measured
#: them in the CENTRED projection, which is the step that breaks arm B.
#:
#:   2   the smallest dimension at which a cosine carries anything. L2
#:       normalising a scalar leaves only its sign, so at r=1 every query
#:       collapses to one of two values -- which is why every rule scores
#:       exactly chance there. 2 is a floor, not a choice, and reporting it
#:       asks "is the cheapest possible setting already enough?"
#:   10  one order of magnitude up, to show whether the curve is flat between
#:       them. It coincides with max(r_90pct); that is not the reason.
R_REPORT_POINTS = (2, 10)

#: Draws averaged for the random subspace. One draw per r made the decoy jump
#: 0.681 -> 0.532 between r=2 and r=3, which is sampling noise wearing the
#: costume of a trend.
RANDOM_DRAWS = 10

#: Fraction of each level held out for testing, as the top quartile by x.
TEST_FRACTION = 0.25

#: (rule, centre). The flag is the whole point of the `uncentred` row: it and
#: `variance` select the SAME directions, so any difference between them is the
#: mean removal and nothing else.
RULES = (('scale', True), ('variance', True), ('random', True),
         ('uncentred', False))

#: Written out on every run so the names in the figures cannot drift away from
#: what the code does. Required by ClaudeRules section 12.
DEFINITIONS = [
    ('production', 'cosine on the raw 1536-D L2-normalised features: no '
                   'projection, no mean removed. What KnnClassifier does today.'),
    ('centred', 'mean removed, ALL components kept. An orthogonal rotation does '
                'not move a cosine, so this row is the mean removal measured '
                'alone.'),
    ('scale', 'the r components with the largest |corr(component, log mpp)|, '
              'mean removed. Ranked on the fit set only.'),
    ('variance', 'the r components with the largest eigenvalue, mean removed. '
                 'Plain PCA.'),
    ('random', f'a uniformly random r-dimensional subspace, mean removed, '
               f'averaged over {RANDOM_DRAWS} draws. The decoy for "any r '
               f'directions would do".'),
    ('uncentred', 'the same directions as `variance`, but projected WITHOUT '
                  'removing the mean. Differs from variance by the constant '
                  'V_r^T mu, which a cosine is not invariant to.'),
    ('chance', 'the largest class share in the test set: what always predicting '
               'one level would score.'),
    ('r', 'how many dimensions are kept. NOT "the r-th component".'),
    ('r = 1', 'meaningless by construction and reported only as evidence of it: '
              'L2 normalising a scalar leaves its sign, so every query collapses '
              'to one of two values and every rule scores exactly chance.'),
    ('wins_vs_production', 'test items this setting got right and production got '
                           'wrong. Paired -- the same items go through both.'),
    ('losses_vs_production', 'test items production got right and this setting '
                             'got wrong. A margin of +8/-7 and one of +8/-1 look '
                             'the same on an accuracy axis and are not the same '
                             'evidence.'),
    ('level accuracy', 'the predicted mpp mapped to the nearest level in LOG '
                       'mpp equals the true level.'),
    ('arm A', 'tile -> tile. Reference and test are both grid tiles from the '
              'reference store. No domain gap.'),
    ('arm B', 'photo -> tile. Queries are synthesised FoV renders carrying the '
              'domain gap, scored per FoV by median-of-medians.'),
    ('split band', 'the test set is the top quartile by x within each level. '
                   'Spatially separated from the reference bank.'),
    ('split random', 'the test set is drawn at random. Leaky, because the store '
                     'holds overlap positions sharing half their pixels with a '
                     'main position. Reported so the size of the leak is '
                     'visible.'),
    ('subset white<T', 'every tile whose own background fraction is at or above '
                       'T is dropped from the reference bank, and the levels '
                       'are rebalanced afterwards.'),
    ('coarse share of errors', 'among the wrong answers, the fraction that '
                               'predicted a COARSER level. 0.0 means every '
                               'error went the other way.'),
]


# ══════════════════════════════════════════════════════════════════════════════
#  The basis
# ══════════════════════════════════════════════════════════════════════════════

def fit_basis(features: torch.Tensor, mpp_labels: np.ndarray) -> dict:
    """Fit mu and V on ONE set of tiles, and rank the components by scale.

    `features` are already L2-normalised, which is what production compares, so
    the centring is applied to the normalised vectors rather than to raw ones.

    The correlation is measured on these same tiles, which is why the caller
    must pass the FIT set. Ranking components by a correlation computed on the
    tiles that are about to be scored is how a subspace scores well on data it
    has already seen.
    """
    matrix = features.numpy().astype(np.float64)
    mean = matrix.mean(axis=0)
    centred = matrix - mean

    # SVD of the centred matrix rather than an eigendecomposition of a 1536x1536
    # covariance: n is a few thousand, so this is both faster and better
    # conditioned, and the right singular vectors ARE the components.
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
    """The r directions this rule proposes, as [r, dim] orthonormal rows.

    'random' does NOT pick r random components: that would still be a subspace
    of whatever the leading components span. It draws a uniformly random
    r-dimensional subspace of the full space, which is the decoy that actually
    denies "any r directions would do".

    'uncentred' returns the same directions as 'variance' on purpose -- the two
    differ only in whether the caller removes the mean, so any gap between their
    scores is the mean removal and cannot be anything else.
    """
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
    """Centre (or not), project, renormalise.

    The renormalisation is not cosmetic: the thing being replaced is a cosine,
    so the subspace version has to be a cosine of the projected coordinates.
    That is also why `centre` matters at all -- a cosine is not invariant to the
    constant shift that removing the mean applies.
    """
    matrix = features.numpy().astype(np.float64)
    if centre:
        matrix = matrix - basis['mean']
    projected = torch.from_numpy(matrix @ directions.T).float()
    return torch.nn.functional.normalize(projected, dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
#  The classifier, and the score
# ══════════════════════════════════════════════════════════════════════════════

def knn_labels(reference: torch.Tensor, reference_mpp: np.ndarray,
               query: torch.Tensor, k: int = K_NEIGHBOURS) -> np.ndarray:
    """Per-query median of its k nearest neighbours' mpp labels.

    Identical in form to KnnClassifier.predict, stopping one median short: that
    class returns the median over all patches of one FoV, and here the grouping
    differs between the arms, so it is done by the caller.
    """
    k = min(k, reference.shape[0])
    similarity = query @ reference.T
    neighbours = similarity.topk(k, dim=1).indices.numpy()
    return np.median(reference_mpp[neighbours], axis=1)


def nearest_level(mpp: np.ndarray, level_mpp_values: np.ndarray) -> np.ndarray:
    """Which level an mpp belongs to, in LOG space.

    The levels are geometrically spaced: on a 4x pyramid a prediction sitting
    between L0 and L1 is 0.5 from one and 2.0 from the other, and a linear
    nearest would call that L1 every time.
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

    # Which way the errors go, because the cost is asymmetric: BenchLocaScope
    # measured that routing one level coarse is survivable (those shots still
    # refine to single-digit micrometres) while routing fine is not.
    wrong = ~correct
    coarse_share = (float((predicted_mpp[wrong] > true_mpp[wrong]).mean())
                    if wrong.any() else float('nan'))

    _, counts = np.unique(true_level, return_counts=True)
    return dict(n_test=int(len(predicted_mpp)),
                level_accuracy=float(correct.mean()),
                mpp_error_relative_p50=float(np.median(relative_error)),
                coarse_share_of_errors=coarse_share,
                chance_accuracy=float(counts.max() / counts.sum()))


# ══════════════════════════════════════════════════════════════════════════════
#  Splitting and balancing
# ══════════════════════════════════════════════════════════════════════════════

def split_band(coords: np.ndarray, level_index: np.ndarray) -> np.ndarray:
    """Hold out the top quartile by x, WITHIN each level.

    Per level, because the levels do not cover the same x range once a deep
    level's grid thins out; one global threshold would hand a whole level to the
    test set and leave it unrepresented in the reference bank, which is not a
    hard split but a missing class.
    """
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
    """Cut a filtered subset back to the same count at every level.

    Filtering on background does not remove tiles evenly: the first run's
    low-background subset came out 342 / 330 / 191, which moves the chance rate
    from 0.333 to 0.394 and hands the leading component "which level has the
    most tiles" to encode. load_slide_balanced's docstring names this trap for
    the unfiltered case; the filter reintroduces it.
    """
    rng = np.random.default_rng(seed)
    levels = np.unique(level_index)
    take = min(int((keep & (level_index == level)).sum()) for level in levels)
    balanced = np.zeros(len(keep), dtype=bool)
    for level in levels:
        rows = np.flatnonzero(keep & (level_index == level))
        balanced[rng.choice(rows, size=take, replace=False)] = True
    return balanced


# ══════════════════════════════════════════════════════════════════════════════
#  The gates
# ══════════════════════════════════════════════════════════════════════════════

def gate_pinned(reference: torch.Tensor, reference_mpp: np.ndarray,
                query: torch.Tensor) -> dict:
    """The production setting must BE production, not a re-implementation.

    KnnClassifier.predict returns one number for a whole set of patches, so the
    comparison is against the median of what knn_labels returns over the same
    set -- the same two medians, in the same order. Exact equality, not a
    tolerance: both paths run the same two np.median calls on the same topk
    indices, so any difference at all means a different computation.
    """
    classifier = KnnClassifier(reference, reference_mpp, k=K_NEIGHBOURS)
    theirs = float(classifier.predict(query))
    ours = float(np.median(knn_labels(reference, reference_mpp, query)))
    return dict(gate='pinned', theirs=theirs, ours=ours,
                difference=abs(theirs - ours), passed=bool(theirs == ours))


def gate_full_rank(reference: torch.Tensor, reference_mpp: np.ndarray,
                   query: torch.Tensor, basis: dict) -> dict:
    """`uncentred` at full rank must equal `production`, exactly.

    A full-rank projection is an orthogonal rotation and no mean is removed, so
    the cosine cannot have moved. This is the only check on the projection
    arithmetic itself -- `centred` is SUPPOSED to differ from production, so a
    transposed matrix or a mixed-up axis would sit there unnoticed and look
    like a finding about centring.
    """
    directions = basis['components']
    theirs = float(np.median(knn_labels(reference, reference_mpp, query)))
    ours = float(np.median(knn_labels(
        project(reference, basis, directions, centre=False), reference_mpp,
        project(query, basis, directions, centre=False))))
    return dict(gate='full_rank', theirs=theirs, ours=ours,
                difference=abs(theirs - ours), passed=bool(theirs == ours))


def gate_shuffled(reference: torch.Tensor, reference_mpp: np.ndarray,
                  query: torch.Tensor, query_mpp: np.ndarray,
                  level_mpp_values: np.ndarray, real_accuracy: float,
                  seed: int) -> dict:
    """Permuted labels must fall to chance.

    Scored against the real run rather than against a threshold: a margin over a
    decoy survives a change in how hard the slide is, where a fixed cutoff is a
    guess about that difficulty.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
#  The settings, evaluated
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_settings(fit_features: torch.Tensor, fit_mpp: np.ndarray,
                      test_features: torch.Tensor, test_mpp: np.ndarray,
                      level_mpp_values: np.ndarray, basis: dict,
                      group_ids: np.ndarray | None, seed: int) -> tuple:
    """Every setting on one (fit, test) pair.

    Returns (score_rows, per_group_rows). `group_ids` is the FoV each test row
    belongs to, or None when one row is one query. When present the per-query
    medians are collapsed by group, which is the second median in production's
    median-of-medians, and each group's prediction is recorded so a wrong answer
    can be traced back to the FoV that produced it.
    """
    rng = np.random.default_rng(seed)
    score_rows, group_rows = [], []
    groups = np.unique(group_ids) if group_ids is not None else None
    production_correct = None

    def run_once(reference, query):
        """One setting, scored. Returns (score, predicted, truth, correct)."""
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
        """How this setting and production disagree, item by item.

        An accuracy difference alone cannot tell +8/-7 from +8/-1, and those are
        a coin flip and a real margin. The comparison is paired by construction
        -- every setting is scored on the same test items -- so the counts are
        free and they are the part that carries the evidence.
        """
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

            # Averaged over draws, and its spread reported: a single random
            # subspace is far too noisy to be read as a level.
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


# ══════════════════════════════════════════════════════════════════════════════
#  Loading the query store for arm B
# ══════════════════════════════════════════════════════════════════════════════

def load_query_level(store_root, wsi_stem, level, pooling):
    """One level's synthesised FoV tiles: features and the FoV each came from."""
    path = FeatureStoreModule.find_one(
        store_root, what=f'query store for {wsi_stem} L{level}',
        wsi_stem=wsi_stem, level=level, pooling='query_tokens')
    meta = FeatureStoreModule.load_meta(path)
    tensors, _ = FeatureStoreModule.load(path, keys=['features', 'fov_id'])

    token_spec = {'dim': meta.dim, 'token_grid': meta.token_grid,
                  'num_prefix': meta.num_prefix}
    slots = pool_tokens(tensors['features'].float(), pooling, token_spec)[0]
    features = torch.nn.functional.normalize(
        slots.reshape(slots.shape[0], -1), dim=-1)
    return features, tensors['fov_id'].numpy().astype(np.int64), float(meta.mpp)


# ══════════════════════════════════════════════════════════════════════════════
#  One slide
# ══════════════════════════════════════════════════════════════════════════════

def analyse_slide(store_root, wsi_stem, args) -> tuple:
    print(f'\n{"=" * 78}\n{wsi_stem}   pooling {args.pooling}\n{"=" * 78}',
          flush=True)

    data = load_slide_balanced(store_root, wsi_stem, args.pooling,
                               args.per_level, seed=args.seed)
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
        # A level that cannot field a bank is not a harder level, it is a
        # missing class -- every test tile of it would be scored against
        # neighbours that cannot be its own, which reads as a collapse.
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
                arm_b = run_arm_b(store_root, wsi_stem, args, data,
                                  features[rows], mpp_labels[rows],
                                  level_mpp_values, subset_name)
                score_rows.extend(arm_b[0])
                group_rows.extend(arm_b[1])
            except Exception as exc:                        # noqa: BLE001
                # Reported, not swallowed: arm B needs query stores that a slide
                # may simply not have, and that is a different fact from arm B
                # failing.
                print(f'  arm B  UNAVAILABLE: {type(exc).__name__}: {exc}',
                      flush=True)

    return score_rows, group_rows, selected_rows, gate_rows


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


def run_arm_b(store_root, wsi_stem, args, data, features, mpp_labels,
              level_mpp_values, subset_name) -> tuple:
    """Reference is the (possibly filtered) bank; queries are the FoV renders.

    The background filter reaches arm B through `features` -- the reference side
    -- and not through the queries, which carry no white_frac because they are
    not grid positions. That is the side the confound is on.
    """
    query_blocks, group_blocks, query_mpp_blocks = [], [], []
    group_offset = 0
    for level in data['levels']:
        block_features, fov_id, level_mpp = load_query_level(
            store_root, wsi_stem, level, args.pooling)
        query_blocks.append(block_features)
        # FoV ids restart per level, so they are offset to keep one group per
        # (level, fov) -- without this the vote would pool tiles photographed at
        # different magnifications into one answer.
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


def report(prefix: str, rows: list) -> None:
    """Every rule at the SAME fixed r values, so nothing is compared against a
    best-of-seven. Both report points are printed because the uncentred curve
    was flat but noisy between them, and one number cannot show that. The full
    sweep is in the CSV for anyone who wants the maximum."""
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


def record_selection(wsi_stem, pooling, subset_name, basis, white_fraction,
                     fit_features) -> list:
    """Which components the scale rule picked, and what else they track.

    corr_white is here for the same reason it is in step 2: on these banks the
    background fraction rises with level (0.000 at L0 to 0.626 at L3), so a
    component can separate levels by being a background detector. The first run
    caught the selection rule doing exactly that at rank 3.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
#  Output
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(rows, path) -> None:
    if not rows:
        print(f'  (nothing to write to {path.name})')
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, restval='')
        writer.writeheader()
        writer.writerows(rows)
    print(f'  {path}  ({len(rows)} rows)')


#: Legend entries stay short so they do not cover the data; the long form goes
#: under the figure. ClaudeRules section 12: the reader who never saw this
#: conversation has to be able to read the panel.
CURVE_STYLE = (('variance', '--s', 'tab:orange'),
               ('uncentred', '-D', 'tab:green'),
               ('scale', '-o', 'tab:blue'),
               ('random', ':^', 'tab:grey'))

FIGURE_CAPTION = (
    'production = cosine on the raw 1536-D features, no projection, no mean '
    'removed (what KnnClassifier does today).    centred = all components kept, '
    'mean removed.\n'
    'variance = top-r components by eigenvalue, mean removed.    uncentred = '
    'the SAME directions, projected without removing the mean.\n'
    'scale = top-r by |corr(component, log mpp)|, mean removed.    random = a '
    f'uniformly random r-dim subspace, mean of {RANDOM_DRAWS} draws (band = '
    'std).\n'
    'chance = the largest class share in the test set.    r = how many '
    'dimensions are kept, not "the r-th component".')


def plot_accuracy(score_rows, wsi_stem, path) -> None:
    """Accuracy against r: arm A beside arm B, both on the band split and the
    unfiltered subset. The verdict is arm B's `uncentred` curve against the
    `production` line -- everything else in the panel is context."""
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


def plot_confusion(group_rows, wsi_stem, path) -> None:
    """Arm B, true level against predicted level, in FoV counts.

    Mass in one column is a collapse; a diagonal that leaks to one side is a
    bias. The first run could only infer which of those it was, from every error
    sharing a direction.
    """
    rows = [r for r in group_rows
            if r['wsi_stem'] == wsi_stem and r['subset'] == 'all']
    if not rows:
        return
    # centred is here to settle a question the first run could only infer: every
    # one of its errors went the same way, which is the signature of a collapse
    # to one answer, but "all errors are fine-ward" and "everything is predicted
    # finest" are different claims and only the matrix separates them.
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


def plot_versus_baseline(score_rows, wsi_stem, path) -> None:
    """Which recipes beat production, and on how much evidence.

    The accuracy difference alone is not readable at this sample size: +1.3
    points came from 7 wins against 8 losses on arm B, and the same +1.3 on the
    background-matched bank came from 8 against 1. Those sit at the same place
    on the axis and are not the same result, so every row carries its paired
    counts next to it.
    """
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
                '+w/-l = paired counts on the same test items: w this setting '
                'got right and production did not, l the other way. +8/-7 and '
                '+8/-1 land in the same place and are not the same evidence.',
                fontsize=7.2, va='bottom', family='monospace', linespacing=1.5)
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(f'  {path}')


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Does the KNN do better in the scale subspace?')
    parser.add_argument('slides', nargs='+', help='wsi_stem of each slide')
    parser.add_argument('--stores', default='result/cache/features')
    parser.add_argument('--pooling', default='cls')
    parser.add_argument('--per-level', type=int, default=1000,
                        help='tiles per level, balanced (default 1000)')
    parser.add_argument('--white-max', type=float, default=0.15,
                        help='also run BOTH arms with reference tiles at or '
                             'above this background fraction dropped; negative '
                             'to skip')
    parser.add_argument('--min-subset', type=int, default=100,
                        help='a level with fewer tiles than this in the '
                             'background-matched subset disables that subset')
    parser.add_argument('--skip-arm-b', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    if args.white_max is not None and args.white_max < 0:
        args.white_max = None

    out_dir = Path(args.out or job_result_dir('SubspaceKnn'))
    out_dir.mkdir(parents=True, exist_ok=True)
    store_root = Path(args.stores)

    all_scores, all_groups, all_selected, all_gates, failed = [], [], [], [], []
    for slide in args.slides:
        try:
            scores, groups, selected, gates = analyse_slide(store_root, slide,
                                                            args)
            all_scores.extend(scores)
            all_groups.extend(groups)
            all_selected.extend(selected)
            all_gates.extend(gates)
        except Exception as exc:                            # noqa: BLE001
            print(f'\n{slide}: {type(exc).__name__}: {exc}')
            traceback.print_exc()
            failed.append(slide)

    print(f'\n{"=" * 78}\nwriting to {out_dir}')
    write_csv(all_scores, out_dir / 'subspace_knn_scores.csv')
    write_csv(all_groups, out_dir / 'subspace_knn_arm_b_fovs.csv')
    write_csv(all_selected, out_dir / 'subspace_knn_selected.csv')
    write_csv(all_gates, out_dir / 'subspace_knn_gates.csv')
    write_csv([dict(term=term, means=means) for term, means in DEFINITIONS],
              out_dir / 'subspace_knn_definitions.csv')
    for slide in {row['wsi_stem'] for row in all_scores}:
        stem = slide.replace(',', '_')
        plot_accuracy(all_scores, slide,
                      out_dir / f'subspace_knn_accuracy__{stem}.png')
        plot_confusion(all_groups, slide,
                       out_dir / f'subspace_knn_confusion__{stem}.png')
        plot_versus_baseline(all_scores, slide,
                             out_dir / f'subspace_knn_vs_baseline__{stem}.png')

    failed_gates = [g for g in all_gates if not g['passed']]
    if failed_gates:
        print(f'\n{len(failed_gates)} GATE FAILURE(S) -- the scores above do '
              f'not mean what they say:')
        for gate in failed_gates:
            print(f'  {gate["wsi_stem"]}  {gate["gate"]}  {gate}')
    if failed:
        print(f'\n{len(failed)} slide(s) failed: {", ".join(failed)}')
    return 1 if (failed or failed_gates) else 0


if __name__ == '__main__':
    sys.exit(main())
