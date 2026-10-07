# PrototypicalRoutingHead — spec

Same task as `training/MppRoutingHead`: route an unknown tile/photo to the
`DsLadder` rung (equivalently, the mpp) it was taken at. Different mechanism —
this is the "ultimate goal" `MppRoutingHead/spec.md` names and defers: given a
handful of prototype tiles each labelled with their own scale, tell an unknown
tile which one it is closest to, so that a NEW WSI (a new `base_mpp`, a new
scanner) needs a fresh set of prototypes, not a fresh training run.

`MppRoutingHead` is the classifier-head baseline this has to beat, not a step
towards it. Both sit on the same `stage1_compare` scorecard as `KnnEstMpp`
(`utilities/bench_modules/bench_stage1_mpp.py`) — see that
file's own module docstring for the comparison shape every stage-1 candidate
is measured on.

**Why this earns a separate project rather than living inside
`MppRoutingHead`.** The forward pass takes TWO kinds of input at once — a
support set (to build a prototype from) and a query (to classify against
it) — and training is episodic, not epoch-over-manifest. Every training-loop,
checkpoint and data-loading assumption `MppRoutingHead/Runtime.py`,
`cli/train.py` and `Datasets.py` make is built around ONE input per forward
pass. Retrofitting that shape to carry two would be the same mistake
`common/Head.py`'s own docstring already names for a different case:
forcing something task-specific to fit a signature that only accidentally
matched it. What genuinely IS generic (encoder access, the reduction
primitives, precision handling, checkpoint idioms) lives in
`aiNNModel/models/` already and is reused from there, per "Files" below.

## Task

Unchanged from `MppRoutingHead/spec.md`: 6-way classification over
`DsLadder.DEFAULT_RUNGS = (1, 2, 4, 8, 16, 32)`. Not mpp regression, not
ordinal regression as the label shape — but SEE "Ordinal-aware loss" below:
`MppRoutingHead/spec.md`'s own late addition (the class-imbalance-weighted +
soft-ordinal-target loss) is adopted here from the start, not retrofitted,
and rung 32's training scarcity (roughly 1% of rung 1's supply, same
`RICHNESS`-driven shortfall `MppRoutingHead` measured) is accepted as a fact
about the data rather than something to fix by resampling — this project's
own bet is that the loss shape, not the data supply, is what has to close
that gap. If it does not, that is a finding, not a blocker.

**This is the DEPLOYMENT task, not the per-episode training sub-task.**
Metric-based meta-learning trains on N-WAY episodes, N < 6 (see "Episode
construction" under Stage 4) — the model never sees the full 6-way task
during that training. `stage1_compare` scoring is still the full 6-way
task; see Stage 4 for why training on a strict subset of it is deliberate,
not a simplification of it.

## Architecture — four stages, each a real decision point

Two axes run through this whole design, not four independent choices:
**which arm is the MAIN LINE** (build and score first, alone) and **which
arms are OBSERVATION ARMS** (built one at a time, after the main line has a
number, never assumed better for having more parameters — `MppRoutingHead
2-5`/`3-2` already found that more head capacity does not reliably help
cross-dataset generalisation, and `2-3` ArcFace was the worst performer
despite being the most geometrically sophisticated frozen-feature head).
See "Working order" in `plan.md` for which arm is built when.

### Stage 1 — encoding

| | |
|---|---|
| **Main line** | ONE frozen foundation model (`gigapath` / `uni2` / `conch_vit`, `TileEncoderFunc`'s registry) |
| Side arm | mixed foundation models |
| Side arm | fine-tuned CNN |

No new code here. `Features.encode_raw` (`aiNNModel/models/common/
Features.py`) is the entry point, unchanged — it already chunks a batch of
tiles through a frozen encoder without building a graph, which is exactly
what BOTH the support set and the query need (many tiles in, one raw
`[N, L, D]` tensor out, per tile). The two side arms are not started; if
either is ever built, it still produces the same `[N, L, D]` shape into
Stage 2, so nothing downstream has to know which arm produced it.

### Stage 2 — dynamic prototype generator

Everything here reads `raw` tiles (encoder output, not yet pooled) for the
support set and produces either (a) one prototype per rung, or (b) for the
Matching-Net arm, no fixed prototype at all — see that arm's own entry.

**The pooling step is shared between support and query — this is not
optional.** `pooled_view`/`grid_view`/`AttentionPoolHead`
(`aiNNModel/models/common/Head.py`, `Heads.py`) already implement the
Avg(GAP)/Attn choice for a SINGLE tile's raw exit; Max and passthrough
(no pooling, keep the token grid) are the two this project adds. Support
tiles and the query tile MUST go through the same pooling choice with the
same weights (for `Attn`, literally the same `AttentionPoolHead` instance) —
two different poolers would put support and query in two different
embedding spaces, and no distance computed after that means anything. This
was a real gap in the first draft of this design (caught in review before any
code existed): the query's own path to becoming a vector has to be drawn
through the SAME module as `POOL_OPT`, not assumed.

| | |
|---|---|
| **Main line** | Set Transformer encoder over the pooled support set (set-to-set self-attention, THEN collapse to one prototype per rung) |
| Side arm | statistical aggregation — mean / median, no parameters |
| Side arm | shared MLP, applied per-tile before the per-rung collapse |
| Side arm (Matching Net) | **no fixed prototype** — see below |

**Why Set Transformer is the main line and not a side arm.** Plain
mean-pooling cannot let one noisy or off-target support tile (a tile that
happened to sample a region edge, or unusually little tissue) be down-weighted
by the OTHER support tiles of its own rung before it enters the prototype;
self-attention over the set can. This is a real, precedented mechanism (Ye et
al., FEAT — "Few-shot Embedding Adaptation with Transformer"), not
speculative capacity. Still an observation arm's worth of caution applies:
build it, score it, and only then decide whether the extra parameters earned
their keep on THIS task's small per-rung support counts.

**Matching Net arm, defined precisely (2026-09-19, settled with the user):**
cross-attention REPLACES the k-NN vote `KnnEstMpp` does today — the query
attends over every individual support-tile embedding (not first collapsed to
a per-rung mean), and the predicted class distribution is the
attention-weighted mix of the attended tiles' OWN rung labels (Vinyals et
al., Matching Networks). No per-rung prototype is ever materialised for this
arm — it is a genuinely different shape from the other three, structurally
closer to `KnnEstMpp` (a live reference bank, differentiable attention
standing in for the KNN vote) than to Prototypical Networks proper. Kept as
a fair, comparable evolution of the CURRENT production baseline, not
compared head-to-head against the prototype-producing arms as if it were one
of them. Shares its cross-attention machinery with Stage 3's Learnable Arm
2 — see that entry; this is one module, not two.

### Stage 3 — prototype-class routing head

Takes the query's own pooled vector (Stage 2's shared pooling, applied to the
query) and either the per-rung prototypes (main line, most arms) or the raw
attended-support mix (Matching Net, feeding straight in from Stage 2).

| | |
|---|---|
| **Main line** | adaptive-temperature cosine routing — L2-normalise both sides, `logit_k = tau * cos(q, p_k)`, `tau` a learned scalar |
| Non-learnable | negative squared Euclidean `-\|\|q - p_k\|\|^2` |
| Non-learnable | fixed-temperature cosine (`tau` a constant, not learned) |
| Non-learnable | multi-prototype soft-min (more than one prototype per rung, `-min_j \|\|q - p_{k,j}\|\|^2`) |
| Learnable Arm 1 | scale-prior bias routing — cosine/`tau` + a learned per-class bias `b_k` |
| Learnable Arm 2 | cross-attention metric routing — `W_q`/`W_k` projections, **the same module as Stage 2's Matching Net arm** |

**Gaussian/Mahalanobis and the hyperbolic Poincaré-ball head are explicitly
OUT of scope (settled with the user 2026-09-19), not deferred-and-forgotten:**
Mahalanobis needs a covariance estimate accumulated by a fit-once streaming
pass beside gradient descent — the exact plumbing gap `MppRoutingHead
spec.md`'s own "Status" section already names as unbuilt for its own 2-4.
Poincaré solves a problem this label space does not have: hyperbolic
geometry earns its keep on HIERARCHICAL (tree-shaped) label sets, and
rung 1→2→4→8→16→32 is a total order, not a tree — there is no hierarchy for
the extra geometry to represent here.

### Stage 4 — training and loss

Not in `aiNNModel/models/` — training loops and losses are task-specific by
this repo's own convention (`MppRoutingHead/Runtime.py`'s own docstring: "task
specific so it stays here rather than moving to `aiNNModel/models/`"). Lives
in this project's own `cli/train.py` + a `Losses.py`.

**Loss — adopted from `MppRoutingHead/spec.md`'s "Ordinal-aware loss" section
verbatim, from the start:**

```
K=6 rungs, ell_c = log2(rung_c) = (0,1,2,3,4,5)
w_c = N / (K * n_c)                              (class-imbalance weight, Datasets.class_weights' formula)

L_bal   = -w_y * log p_y                          (weighted CE -- today's MppRoutingHead default)
L_ord_A = L_bal + lambda * sum_i w_i (E_c[ell_c] - ell_i)^2 / sum_i w_i
                                                  (regression penalty on the expected log-rung,
                                                   class-weighted like L_bal)
L_ord_B = w_y * (-sum_c q_c(y) log p_c)             (soft target: Gaussian kernel over ell_c - ell_y)
```

Both `lambda` (A) and `sigma` (B, the kernel bandwidth) are new hyperparameters,
default to the value that recovers plain `L_bal` (`lambda=0`, `sigma->0`) so a
smoke run without them is comparable to a `MppRoutingHead` run. Which of A/B
wins is an empirical question this project answers itself — it is not
inherited as a decision, only as a formula.

**Episode construction: the mixed pool and the overlap rule.** Support and
query are drawn from ONE pool per rung -- every position of that rung across
every training WSI (`Episodes.pool_by_rung`). Which WSI a position came from
is not part of the task: a scale question on one slide is the same question
on another with the same base mpp and rungs (user, 2026-09-23). What the WSI
still decides is the **overlap rule**: two positions on the same WSI whose
level-0 footprints overlap -- across rungs too -- never land on opposite
sides of a draw (`--max-overlap`, default 0: any shared area). Positions on
the same side may overlap; every position in one draw is distinct.

**Episode reuse (`--episode-reuse`, `--reuse-k`).** One DRAW is `ks` support
batches and `kq` query batches for one rung combination; one optimizer step
is one (support batch, query batch) pair:

| mode | per draw | draws per epoch | steps per epoch |
|---|---|---|---|
| `none`   | 1 x 1 (nothing held) | 39 x K | 39 x K |
| `hold_s` | 1 x K (support held, K query batches through it) | 39 | 39 x K |
| `hold_q` | K x 1 (query held, K support batches against it) | 39 | 39 x K |

The frozen encoder runs once per batch; the pairs reuse its tokens, so only
the trainable part (pooling, G, F, Collapse, head) runs per pair.

**K and the episode count are measured, not guessed.** `--reuse-k auto` is
`K_max`, the largest K for which the real draw succeeds on every training
combination for BOTH held shapes (`Episodes.max_feasible_k`). All three modes
use it, so their step counts match. `--episodes-per-epoch auto` is the table
above. An integer is used as given and refused if it cannot be drawn. Every
run prints "given -> used" for all three, and `--supply-only` prints the
per-rung supply and what `auto` resolves to, without training.

**Cross-domain (`--cross-domain-dataset`).** A combination that fits inside
`BRACS_RUNGS` may put `bracs/train` on one side and the training dataset on
the other (a coin per draw), forcing the comparison across a staining domain
neither side can shortcut through.

**Episode construction: which RUNGS a draw routes among.** Every 3-, 4- and
5-of-6 rung combination except the held-out ones -- 19 + 14 + 6 = 39
(`Episodes.training_combos`) -- each ONCE per epoch, in a fresh order every
epoch (`Episodes.epoch_schedule`; K rounds under `none`). 6-way is never
trained: only validation and `stage1_compare` ask the model to route among
all six at once.

**Held-out combinations** (`Episodes.HELD_OUT_COMBOS`): `{1, 4, 16, 32}`,
`{1, 2, 4}` and the full 6-way. Never trained on, at any WSI. They test
whether the metric space generalises to a rung COMBINATION never asked about
jointly -- every rung in them still appears in plenty of training
combinations. A third evaluation axis alongside the held-out WSIs
(the recorded split, `make_split.py`) and the full-6-way deployment task.

**What changes in the loss for a variable N.** `log2_rungs`
(`MppRoutingHead spec.md`'s ordinal-loss formulas, adopted here) must be
built from THIS EPISODE's own `N` chosen rungs, not the fixed
`DsLadder.DEFAULT_RUNGS` 6-tuple — and `target` must be the LOCAL index
into that episode's own rung list (`0..N-1`), not the global rung index
(`0..5`), since `CosineTauHead` returns `[query_N, N]` logits shaped by
however many prototypes it was actually handed.

Support and query render through `Camera` the same way `MppRoutingHead/
Datasets.py`'s own `CameraBank`/`render_row` already do (`Episodes.
render_draw`): on train each capture draws its augmentation from a generator
seeded off the episode sampler, so a resumed run renders the photos the
uninterrupted one would have; on eval each row renders from its own
identity, the same photo every time. Both names were renamed public
in `MppRoutingHead/Datasets.py` first (were `_CameraBank`/`_render_row`) so
this reuse is legitimate rather than reaching past a no-stability-promise
underscore — every reference to the old names, including comments in
`cli/train.py`/`Runtime.py`/`test_camera.py`, was updated
in the same change.

**Validation, K x K** (user, 2026-09-24). For each eval dataset and each
held-out combination: K support batches and K query batches, every one of
the K x K pairs scored (`train.score_kxk` / `kxk_report`):

```
rung in a combo   mean over the K x K pairs of that rung's accuracy
combo             mean of its rungs
dataset           mean of its combos
total             mean of the datasets
```

`--val-reuse-k auto` is the largest K every dataset supplies for every
held-out combination. The val draw uses `random.Random(--seed)` afresh every
epoch, so the questions stay fixed and only the model changes. Every epoch
prints each combination's accuracy and its per-rung row, each dataset's, and
the total. Checkpoints: `_best.pt` on the total, `_6rung.pt` on the full
6-way combination (mean over datasets), `_native.pt` on the mean over the
combinations whose every query rendered natively.

**Test** (`cli/evaluate.py`) writes two tables: the ORIGINAL one (random full
6-way episodes, accuracy pooled per combination -- unchanged, comparable with
earlier runs and tile by tile with `MppRoutingHead`) and the K x K one above
on the test split.

**Resume** (`--resume-dir`, `aiNNModel/models/common/Resume.py`). Unset:
train from scratch, write nothing. Set: write `<run>_resume.pt` there every
epoch -- trainable modules, optimizer, scheduler, epoch, best scores, the val
rows so far, and every RNG (python, numpy, torch, cuda, the episode
sampler's) -- and continue from it when it is already there. `--epochs` is
the total, so a finished run resumes into nothing. A file written under a
different identity (any argument that changes what is trained, plus the
resolved K and episode count) is refused by name.

### Stage 4's OWN three-way comparison: how the embedding gets trained

Settled 2026-09-20: this is a THIRD axis, orthogonal to Stage 1-3's own
architecture choices (encoder / generator / head) — the same Pooling ->
SetTransformerPrototype -> CosineTauHead stack (or whichever Stage 2/3 arm is
under test) can be trained by any of the three, and a result has to name
BOTH which architecture and which training arm produced it.

| Arm | Training | Inner-loop grad steps | Cost |
|---|---|---|---|
| **Baseline prototype learning** | ordinary, non-episodic supervised training of the embedding (identical shape to `MppRoutingHead`'s own baseline 2: every training tile classified directly, e.g. through `CosineTauHead` against a FIXED learned weight matrix, or any existing `Heads.py` classifier). Prototypes are computed POST-HOC at inference by averaging a support set's embeddings in this trained space — Chen et al. 2019's "Baseline" category for few-shot classification, and the closest thing to `MppRoutingHead spec.md`'s still-unbuilt 2-2 (NCM), except the embedding space is now one this project trained on purpose rather than a frozen foundation model's raw exit | 0 | cheapest — identical per-step cost to `MppRoutingHead` |
| **Metric-based meta-learning** (built, plan.md 1.1-1.3) | episodic; the "adaptation" to a new support set is a forward-pass COMPUTATION (mean, or `SetTransformerPrototype`'s self-attention) — no gradient steps at all. ONE backward pass per episode updates the embedding. This is Prototypical Networks' own category (Snell et al.), and what plan.md has been building | 0 (no inner loop) | current main line's cost |
| **Optimization-based meta-learning** (NEW arm, not yet designed in code) | episodic; an INNER loop takes `K` real SGD steps on a COPY of the embedding's parameters, using a loss computed FROM the support set itself, before the OUTER loss (query, scored with the ADAPTED copy) backprops to update the ORIGINAL parameters — MAML's own shape (Finn et al. 2017) | `K` (typically 1-5) | `K+1` forward/backward passes per episode; full (second-order) MAML backprops THROUGH the inner loop's own gradients (Hessian-vector products); first-order approximations (FOMAML, Reptile) drop that term and are cheaper and more stable at some fidelity cost |

**Not the same "meta-learning."** Colloquial "meta-learning" usually means
the optimization-based kind; Prototypical Networks (the metric-based arm
already built) is meta-learning too, in the sense the few-shot-learning
literature uses the word, but has NO inner loop — its "adaptation" is a
plain forward pass. The distinction matters here specifically because the
optimization-based arm is meaningfully more expensive and structurally
different (needs a differentiable inner-loop step, e.g. `torch.func`'s
functional-parameter machinery, or manual parameter copying), not a small
variation on what is already built.

**Full MAML vs FOMAML vs Reptile — the actual mechanics, and the current
lean (2026-09-20, not yet a final decision).** The inner loop (`K` real SGD
steps on a copy of the parameters, using the support set's own loss) is
IDENTICAL across all three; they differ only in how the outer update is
computed from it:

- **Full MAML (second-order).** The inner loop's own gradient computation
  is kept differentiable (`create_graph=True`) so the adapted parameters
  `theta'` remain a function of `theta` in the autograd graph. Backprop-ing
  the outer (query) loss through `theta'` back to `theta` therefore
  differentiates THROUGH a gradient -- a Hessian-vector product per inner
  step. Most faithful to the actual meta-objective; the most memory (every
  inner step's graph stays alive) and the most numerically fragile.
- **FOMAML (first-order).** Same inner loop, but the outer gradient is
  computed AS IF `theta'` did not depend on `theta` at all -- the gradient
  of the outer loss w.r.t. `theta'` is applied directly as the update to
  `theta`, dropping the second-order term entirely. The original MAML
  paper's own ablation found this costs little in practice.
- **Reptile (Nichol et al. 2018).** No outer gradient computed at all: run
  the inner loop to get `theta'`, then move `theta <- theta + epsilon *
  (theta' - theta)`. Justified by a Taylor-expansion argument (approximates
  FOMAML's expected update direction) rather than by differentiating an
  outer loss. Needs no differentiable-inner-loop machinery whatsoever --
  plain sequential SGD steps, then a parameter interpolation.

Current lean: **build Reptile first.** It needs no second-order gradients
and no functional-parameter machinery at all, which matters given this
codebase's fp16-under-Adam NaN history (`MppRoutingHead spec.md`'s Status
section) -- the fastest path to a real answer on whether optimization-based
training helps here at all, with FOMAML as the next step up in fidelity if
Reptile's result looks worth sharpening, and full MAML last. Not yet a
final decision.

## Open design decisions

1. **Checkpoint format — RESOLVED, built 2026-09-20.**
   `Checkpoints.save_checkpoint`/`build_from_checkpoint`
   (`aiNNModel/models/common/Checkpoints.py`) are shaped around ONE encoder +
   ONE `Head` and have real callers today (`MppRoutingHead`,
   `stage1_estimation/ClassifierEstMpp.py`) that must not break —
   untouched. Added `save_prototype_checkpoint`/`build_prototype_from_
   checkpoint` in the SAME file instead: three trained modules (pooling,
   generator, head) rather than one `Head`, `pooling_cfg`/`generator_cfg`
   carried the same way `head_cfg` already is, `extra: dict` reused
   unchanged as the task-specific escape hatch. `build_prototype_from_
   checkpoint` lazily imports `Pooling`/`SetTransformerPrototype`/
   `PrototypeRoutingHeads` from `aiNNModel/models/` (never from a training
   package — see decision 4 below for why that direction matters).
2. **Stage 2/3 shared cross-attention module's home.** One class, used as
   Stage 2's Matching Net AND Stage 3's Learnable Arm 2 — goes in
   `aiNNModel/models/` (it is encoder/head plumbing, not training logic) as
   its own file, tentatively `aiNNModel/models/CrossAttentionMatch.py`. Not
   yet built (plan.md step 5).
3. **Set Transformer's own home — RESOLVED, built.** `aiNNModel/models/
   PrototypeGenerators.py` (merged there 2026-09-21 with the other Stage 2
   arms, decision 8), same reasoning as 2.
4. **`Pooling.py`'s home — RESOLVED, moved 2026-09-20.** Built first under
   `training/PrototypicalRoutingHead/`, moved to `aiNNModel/models/` once
   decision 1's checkpoint code needed to rebuild it: `Checkpoints.py`
   lives in the GENERIC layer (`aiNNModel/models/common/`), and having it
   import a class from a TASK-SPECIFIC training package would invert the
   dependency direction CLAUDE.md's repo-layout section establishes — the
   generic layer is what task packages depend on, never the reverse.
   `Pooling` is encoder/head plumbing exactly like `SetTransformerPrototype`
   (nothing about Avg/Max/Attn/passthrough is specific to this one task), so
   it belongs where that already lives, not where it happened to be written
   first.
5. **Multi-arm training infrastructure — RESOLVED, built 2026-09-20.** The
   eventual shape (user's own framing): three INDEPENDENT axes stacked
   together — Stage 4 training regime (baseline / metric-based /
   optimization-based) × Stage 2 generator (Set Transformer / shared MLP /
   Matching Net) × Stage 3 routing head (Cosine+τ / cross-attention). Built
   the SEAM, not the cross product — none of the non-main-line arms exist
   yet (plan.md step 5 is still "one at a time"), so there is nothing to
   select between besides today's one entry per axis. Concretely:
   - `Runtime.py` (new): `GENERATOR_CHOICES`/`ROUTING_HEAD_CHOICES` — see
     the file's own docstring for why each entry is a BUILDER FUNCTION
     rather than a `(class, config_class)` tuple `MppRoutingHead.
     HEAD_CHOICES`-style: a future arm's config shape (Matching Net's own
     W_q/W_k dims) is not yet known, and forcing one shape now would be
     guessing it before a second entry exists to test it against.
     `ROUTING_HEAD_CHOICES` still has one entry (`cosine_tau`).
     `GENERATOR_CHOICES` gained three MORE (2026-09-20, plan.md step 5.1
     moved first — none of `mean`/`median`/`shared_mlp` needs
     `CrossAttentionMatch.py` to exist, unlike Matching Net):
     - `mean`/`median` (`aiNNModel/models/PrototypeGenerators.py` --
       merged there 2026-09-21 with the other Stage 2 arms, all THREE
       generator files consolidated into one, matching how
       `PrototypeRoutingHeads.py` already puts every Stage 3 arm in one
       file rather than one per file) — the zero-parameter floor for this
       axis (Stage 4's own floor, `cli/train_baseline.py`, was deleted
       2026-09-22, its inference-time positioning never settled).
     - `shared_mlp` (same file) — Deep Sets
       (Zaheer et al. 2017): the same small MLP applied to each support
       member independently, no cross-member interaction, then mean-pooled.
       The direct comparison point for "does `set_transformer`'s
       cross-member self-attention buy anything over a per-member learned
       transform". Built with comparable depth (2) and BOTH an outer and
       inner residual, deliberately NOT narrower (`width_mult=1.0`, not
       0.5) — see that file's own docstring: a narrower/non-residual MLP
       would confound "less capacity"/"no residual anchoring" with "no
       attention", muddying the one axis this arm exists to isolate. A
       narrower variant is a legitimate SECOND question, a separate
       registry entry once this one has a real number.
   - `cli/train.py` gained `--generator`/`--routing-head`, resolved through
     that registry; checkpoint filenames and `extra` now record both
     choices, so a second arm's checkpoint is at least identifiable on disk
     before `build_prototype_from_checkpoint` is made registry-aware (real
     work for whenever that second arm exists, not done now).
   - `cli/train_baseline.py` did NOT read this registry as of 2026-09-20 --
     ITS OWN Stage-3 choice went through `Heads._CLASSIFIER_REGISTRY`
     instead, because `common.Head.Head`'s classifier shape (`features`
     alone) and a routing head's shape (`query, prototypes` explicit) are
     not interchangeable through `Head`'s own `classifier(cfg)`
     construction contract. SUPERSEDED 2026-09-21 (Open design decisions,
     7): that file stopped going through `common.Head.Head` at all, so the
     shape mismatch that justified two separate registries no longer
     applies to it -- it now reads `Runtime.ROUTING_HEAD_CHOICES` directly,
     the SAME registry `cli/train.py` uses. `Heads._CLASSIFIER_REGISTRY`
     lost its `'prototype_cosine'` entry once nothing built through `Head`
     needed it any more.
   - Stage 4 (training regime) stays THREE SEPARATE CLI scripts, not one
     merged entry point — their data flow shapes genuinely differ (one
     encoder pass / episodic support-query / episodic + inner-loop
     gradients) and their CLI argument sets barely overlap
     (`n_support`/`n_query`/`--episode-reuse` mean nothing to the baseline arm).
     Same principle `MppRoutingHead/cli/train.py`'s own `run_baseline2`/
     `run_baseline3` split uses (one file, two functions, because the two
     shapes differ) — this project just needed the split one level up, at
     the file boundary, because even the CLI surface diverges, not only the
     inner loop.
   - `cli/train.py` imports `training.PrototypicalRoutingHead.Runtime`
     despite `training/MppRoutingHead/Runtime.py` sharing the bare name
     `Runtime.py` — RESOLVED by construction: `training/`,
     `training/MppRoutingHead/` and `training/PrototypicalRoutingHead/`
     are real Python packages (each with its own `__init__.py`), so
     every cross-file import within these two packages is fully-qualified
     (`from training.MppRoutingHead.Datasets import ...`, `from training.
     PrototypicalRoutingHead.Episodes import ...`) and resolves by import
     path alone — no `sys.path` ordering involved, and no future same-named
     file in either package can collide again. `SuperPathPoint` still uses
     the `add_training_package` mechanism (it has no top-level bare file
     that collides with anything), see that function's own docstring.
6. **Held-out validation for the metric-based main line — RESOLVED, built
   2026-09-21.** Both axes needed at once, each resolved by reusing
   something that already existed rather than a new design:
   - **Held-out WSI**: `--eval-datasets` (`bracs/test`/`ki67_with_photo`,
     same defaults) × `wsi_split` pointed at `MppRoutingHead`'s own cache
     (`result/cache/mpp_routing_head/`) — the EXACT same mechanism
     `MppRoutingHead/cli/train.py`'s own `val_rows` uses, so both scripts
     read the IDENTICAL held-out WSI names under shared defaults, by
     construction (same file on disk via `wsi_split`'s "EXISTING WINS"
     rule), not by two independent derivations happening to agree.
   - **Held-out rung combination**: `Episodes.HELD_OUT_COMBOS` gained a
     THIRD entry — the full 6-way tuple `(1.0, 2.0, 4.0, 8.0, 16.0, 32.0)`,
     spec.md's own actual DEPLOYMENT task, never rehearsed by ANY training
     episode (`n_choices` never contains 6 either) — and a new
     `Episodes.sample_val_episode` draws the episode's rung subset FROM
     `HELD_OUT_COMBOS` (all three entries) instead of excluding them, via a
     shared `_draw_episode` helper factored out of `sample_episode` so the
     WSI/position-slicing mechanics are not duplicated between the two.
   - Val episodes are drawn with a FRESH `random.Random(args.seed)` every
     epoch (not the training loop's own advancing `rng`), so the val SET
     stays identical epoch to epoch — same reasoning `MppRoutingHead.
     predict`'s own `epoch_seed=0` gives.
   - The scoring formula (`score`/`rescore`) is kept in sync with
     `MppRoutingHead/cli/train.py`'s and `cli/train_baseline.py`'s own —
     these were ADDED to this package's own `Runtime.py` (a duplicate of
     `MppRoutingHead/Runtime.py`'s same-named functions, same bare-name-
     collision reasoning as decision 5's wandb section), not written a
     third, differently-shaped way.
   - EVOLVED THROUGH THREE SHAPES the same day before settling: (1) a
     `val_report` producing a dataset-level "all, n-weighted"/"all,
     dataset-avg" summary blended across every combo, plus a SEPARATE
     `rung_report` pooling every combo's contribution to each rung
     together; (2) `combo_report` added alongside those two, per-combo
     accuracy kept separate; (3) `val_report`/`rung_report` REMOVED
     entirely once it was clear ANY number blended across combos hides
     whether accuracy holds up on the actual 6-way DEPLOYMENT task
     specifically (a rung, or a whole dataset, scored alongside 2 other
     candidates and the SAME thing scored alongside 5 are different
     questions) — `combo_report` (`val_scores_per_combo.csv`/`val_scores_
     per_rung.csv`) is now the ONLY val report: for EACH combo (3-way /
     4-way / the full 6-way), its own overall accuracy (with native/
     resampled) AND its own per-rung breakdown, nested, computed within
     that combo's own query examples only. Per-combo's own overall
     accuracy is POOLED within the combo, not rung-averaged — a combo's
     query examples are already balanced across its own rungs by
     construction (`--n-query` is the same for every rung an episode
     drew), so there is no tile-count skew here for rung-averaging to
     correct (that skew is `RICHNESS`'s own coarse-rung TRAINING
     shortfall, with no equivalent at the per-episode level).
   - CHECKPOINT SELECTION reads TWO scalars derived directly from `combo_
     report`'s own rows, not a blended summary: `_best.pt` on `_deployment
     _accuracy` (the full 6-way combo's own accuracy, n-weighted across
     the two eval datasets — the actual deployment task, never rehearsed
     by training, so the checkpoint most representative of "does it do
     the job" rather than of the easier compositional-generalisation
     probes); `_best_diagnostic.pt` (renamed from `_best_unweighted.pt`)
     on `_diagnostic_accuracy` (the plain mean across every combo AND
     dataset — a broader signal, kept as a SEPARATE file specifically so
     a checkpoint that only excels at the deployment-scale probe is
     visible as a difference between the two saved checkpoints, not
     hidden inside one blended number). Via a new `save_tagged`/`_prototype
     _weight_filename` pair in `cli/train.py` itself — the prototype-
     checkpoint format (`save_prototype_checkpoint`, three modules) has no
     `weight_filename`-equivalent of its own yet, so this is a LOCAL
     naming convention (`<encoder>_frozen_<pooling>_<generator>_
     <routing_head>_<tag>.pt`), not a reuse of `Checkpoints.weight_
     filename` (shaped for one `Head`).
   - Still NOT solved: the actual GENERALISATION QUALITY question (does
     the model do well on held-out combos/WSIs) has no answer yet — this
     decision built the MEASUREMENT, not a result. `evaluate.py` (test
     split, spec.md's Files section) is still not built either.
7. **Pooling/RoutingHead unified across Stage 4 arms; `arm`→
   `training_framework` — RESOLVED, built 2026-09-21.** User's own framing,
   settled after several rounds: Pooling/Generator/RoutingHead are
   properties of the MODEL, independent of Stage 4 (which regime trained
   it) — a generator's absence in the BASELINE arm is not a limitation to
   work around, it is Chen et al.'s own definition of "Baseline" (no
   support set at training time, so nothing for a generator to act on).
   - `cli/train_baseline.py` gained `--pooling`/`--routing-head`, reading
     the SAME `Pooling`/`Runtime.ROUTING_HEAD_CHOICES` `cli/train.py`
     already uses — no `--generator`, and none is planned for this arm.
   - This meant leaving `common.Head.Head`/`Heads._CLASSIFIER_REGISTRY`:
     `Head`'s classifier contract (`classifier(cfg)`, one positional arg)
     cannot carry a `--routing-head` CHOICE through construction.
     `LearnedPrototypeClassifier` (`aiNNModel/models/
     PrototypeRoutingHeads.py`) is now `routing_head`-injected instead of
     hardcoding `CosineTauHead` — it wraps WHICHEVER `ROUTING_HEAD_CHOICES`
     entry was chosen around its own learned weight matrix, rather than
     duplicating a distance function. `Heads._CLASSIFIER_REGISTRY` lost its
     `'prototype_cosine'` entry (dead once nothing built through `Head`
     needed it) and the import that supported it.
   - `Checkpoints.save_prototype_checkpoint` gained an OPTIONAL
     `generator`/`generator_cfg` (default `None`) rather than getting a
     fourth, parallel checkpoint-saving function written for the
     two-module (pooling + classifier, no generator) shape — `cli/
     train_baseline.py` calls it with both left `None`.
     `build_prototype_from_checkpoint` still rebuilds the three-module
     shape unconditionally on the READ side; making it handle a
     `None`-generator checkpoint is unstarted, same "not registry-aware
     yet" caveat decision 1 already names.
   - `cli/train_baseline.py`'s local `score`/`rescore`/`rescore_by_rung`/
     `wandb_init`/`wandb_log`/`wandb_finish` copies were REMOVED in favour
     of importing them from this package's own `Runtime.py` — the
     bare-name-collision concern that justified the duplicates no longer
     applies (verified: a script IN this package importing `Runtime`
     resolves to this package's own file), it just had not been revisited
     since `Runtime.py` was built.
   - `arm` renamed to `training_framework` in both files'
     `_IDENTITY_FIELDS` — `arm` is this project's own term (spec.md's
     Stage 4 comparison table), but is ALSO used loosely throughout
     spec.md's prose for Stage 2/3 observation arms, so as a COLUMN NAME
     specifically identifying "which Stage 4 regime" it was ambiguous;
     `training_framework` names that one axis precisely. Chosen over
     `training_arch` on purpose: "architecture" is what Pooling/Generator/
     RoutingHead now ARE (this decision's whole point), so naming the
     Stage 4 axis "arch" would contradict the distinction just established.
8. **Stage 2's generator files consolidated into one — RESOLVED, built
   2026-09-21.** `SetTransformerPrototype.py`/`TrivialPrototype.py`/
   `SharedMlpPrototype.py` merged into `aiNNModel/models/
   PrototypeGenerators.py` -- three separate files was never a deliberate
   choice for this axis, just how they happened to get written one at a
   time; `PrototypeRoutingHeads.py` already puts every Stage 3 arm in ONE
   file, so Stage 2 having three was the inconsistent one. `Runtime.py`'s
   own import and `Checkpoints.build_prototype_from_checkpoint`'s lazy
   import both updated; class names (`SetTransformerPrototype`,
   `TrivialPrototype`, `SharedMlpPrototype`) are unchanged, only which
   file they live in moved. Matching Net (plan.md step 5.1) goes in this
   same file when it is built.

## Files

```
aiNNModel/models/                    UNCHANGED except three new files
    Pooling.py                        NEW -- the shared support/query pooling
                                     module (built): Avg/Max/Attn/
                                     passthrough, Attn wraps Heads.
                                     AttentionPoolHead. MOVED here from
                                     training/PrototypicalRoutingHead/
                                     2026-09-20 -- it is generic encoder/head
                                     plumbing, same kind as
                                     PrototypeGenerators.py, and
                                     Checkpoints.py (below) rebuilding it on
                                     load would otherwise mean the GENERIC
                                     layer importing FROM a task-specific
                                     training package, backwards
    PrototypeGenerators.py            NEW -- Stage 2, every generator arm in
                                     ONE file (merged 2026-09-21 from three
                                     separate files -- SetTransformerPrototype.py/
                                     TrivialPrototype.py/SharedMlpPrototype.py --
                                     which was the odd one out, not a
                                     deliberate choice: PrototypeRoutingHeads.py
                                     already put every Stage 3 arm in one
                                     file, so Stage 2 now matches):
                                     SetTransformerPrototype (main line,
                                     built), TrivialPrototype (`mean`/
                                     `median`, the zero-parameter floor,
                                     built 2026-09-20), SharedMlpPrototype
                                     (Deep-Sets-style observation arm, no
                                     cross-member attention, built
                                     2026-09-20). Matching Net (plan.md step
                                     5.1) belongs here too when it exists
    PrototypeRoutingHeads.py          NEW -- Stage 3, query-vs-prototypes
                                     heads (`forward(query, prototypes)`,
                                     the shape every arm here shares).
                                     `CosineTauHead` (the main line, built)
                                     plus `LearnedPrototypeClassifier`
                                     (built 2026-09-20, GENERALISED
                                     2026-09-21) -- the exception to
                                     "prototypes as an explicit argument"
                                     from the CALLER's side: Stage 4's
                                     BASELINE arm, a FIXED learned weight
                                     matrix used as a `routing_head`-
                                     INJECTED module's own `prototypes`
                                     argument (any `Runtime.
                                     ROUTING_HEAD_CHOICES` entry, not
                                     hardcoded `CosineTauHead` any more).
                                     No longer registered in
                                     `Heads._CLASSIFIER_REGISTRY` / built
                                     through `common.Head.Head` -- see Open
                                     design decisions, 7. The non-learnable
                                     block, bias arm and cross-attention arm
                                     are plan.md step 5, not yet written
    CrossAttentionMatch.py            NEW, not yet built -- shared by Stage 2's
                                     Matching Net and Stage 3's Learnable Arm 2
    Heads.py                         REUSED, `_CLASSIFIER_REGISTRY` lost its
                                     `'prototype_cosine'` entry 2026-09-21
                                     (Open design decisions, 7) -- otherwise
                                     unchanged (AttentionPoolHead for the
                                     Attn pooling choice)
    common/
        Head.py                     REUSED unchanged (pooled_view/grid_view)
        Features.py                  REUSED unchanged (encode_raw)
        Checkpoints.py                EXTENDED -- `save_prototype_checkpoint`
                                     gained an OPTIONAL `generator`/
                                     `generator_cfg` 2026-09-21 (Open design
                                     decisions, 7); `save_checkpoint`/
                                     `build_from_checkpoint` (the ORIGINAL,
                                     pre-existing pair) still untouched (see
                                     Open design decisions, 1)

training/PrototypicalRoutingHead/
    spec.md                          this file
    plan.md                          working order, staged
    Episodes.py                      (built) the mixed pool, the overlap
                                     rule, the K-batch draw, the 39 training
                                     combinations and their epoch schedule,
                                     max_feasible_k; plus the ORIGINAL
                                     per-episode draw cli/evaluate.py's
                                     first table still uses. Draws from a
                                     manifest built by MppRoutingHead.
                                     Datasets.build_manifest -- REUSED
                                     (imported), not copied, since nothing
                                     about "which positions exist for a WSI
                                     at a rung" is specific to how a caller
                                     trains on them. NOT named Datasets.py --
                                     MppRoutingHead already has a top-level
                                     module by that name; even though both
                                     packages are now real Python packages
                                     with fully-qualified imports (2026-09-22,
                                     see this file's "Open design decisions"
                                     5), a second Datasets.py here would
                                     still read as confusable with this
                                     file's own `from training.MppRoutingHead.
                                     Datasets import ...` line, so it keeps
                                     its distinct name regardless.
                                     sample_val_episode (built 2026-09-21)
                                     draws FROM HELD_OUT_COMBOS (now 3
                                     entries, including the full 6-way
                                     DEPLOYMENT combo) instead of excluding
                                     them -- shares _draw_episode's WSI/
                                     position mechanics with sample_episode
                                     rather than duplicating them; see Open
                                     design decisions, 6
    Runtime.py                       (built 2026-09-20) GENERATOR_CHOICES
                                     (Stage 2: set_transformer, mean,
                                     median, shared_mlp) / ROUTING_HEAD_
                                     CHOICES (Stage 3: cosine_tau only) --
                                     see Open design decisions, 5. Also
                                     holds wandb_init/wandb_log/wandb_finish
                                     AND score/rescore/rescore_by_rung
                                     (added 2026-09-21 for cli/train.py's
                                     val loop) -- all duplicates of
                                     MppRoutingHead/Runtime.py's own,
                                     100% generic, same bare-name-collision
                                     reason as decision 5's own Runtime.py
                                     note
    Losses.py                        (built) L_bal / L_ord_a / L_ord_b, each
                                     taking THIS EPISODE's own rung subset
                                     (episode_class_weights recomputes w_c
                                     per episode -- no fixed 6-way table)
    cli/
        train.py                     (built) episodic training -- the
                                     metric-based line. Resolves K and the
                                     episode count (given -> used, printed),
                                     trains one step per (support, query)
                                     batch pair, then the K x K held-out val
                                     block (per combination, its per-rung
                                     row, per dataset, total). Writes
                                     val_scores_per_combo.csv / _per_rung.csv
                                     and *_best.pt (total) / *_6rung.pt /
                                     *_native.pt; --resume-dir; --supply-only
        train_baseline.py             DELETED 2026-09-22 -- Stage 4's own
                                     BASELINE arm, built 2026-09-20,
                                     generalised 2026-09-21, never had its
                                     inference-time positioning settled (what
                                     a "Baseline" checkpoint should actually
                                     do against a genuinely new WSI). Will be
                                     redesigned from scratch, not resumed
                                     from that version.
        evaluate.py                   (built) scores checkpoints on the TEST
                                     half of the recorded split into two tables:
                                     the original (random full 6-way
                                     episodes, pooled) and K x K (train.py's
                                     val definition), reusing train.py's own
                                     scoring functions
jobscripts/PrototypicalRoutingHead/
    PrototypicalRoutingHead.sh        one job: the nine Stage 2/3 model
                                     combinations x none/hold_s/hold_q (27
                                     runs), the Matching-Net combination
                                     with --cross-domain-dataset (1 run),
                                     then cli/evaluate.py. RESUME_DIR
                                     defaults to $OUT/resume -- resubmit the
                                     same command to continue. Reads
                                     MppRoutingHead's mask/sampler caches.
                                     SUPPLY_ONLY=1 prints supply and the
                                     auto K / episodes and stops
```
