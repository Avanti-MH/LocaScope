'''Every setting of a CoAE run (spec.md), one dataclass a part; a field with several values is a tuple. An arm has two names, both made
from the config alone (the tables LOSS_TERMS, NAME_PARTS, DIR_PARTS) and each read back to the config (config_of_arm, config_of_dir):
the CODE `<local><global><sources>[.<modifier>...]` (CoaeConfig.arm): the letter of the local branch's loss, the letters of the global
branch's losses, the number of simulated sources, then each setting off its default (the FM is not one: the code is the same for every FM,
stage 2 says the FM in the prefix of `<Fm>CoAE+<arm>`, config_of_arm takes it as `encoder`); and the DIRECTORY name
`<level>-<mechanism>-<variant>[-<part>...]` (CoaeConfig.dir_name, the FM among the parts): where the run's results and log are (result/<job>/<dir>, log/<job>_<dir>).

    cfg = CoaeConfig()                                     # the defaults: arm RP3 (gigapath frozen, point prediction)
    cfg = replace(cfg, data=replace(cfg.data, level='light', mechanism='ind'), head=replace(cfg.head, variant='vdist'))   # RP3.ind.vdist.lightperturb
'''
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

#: Real photos (ki67_with_photo) only validate: no slide of it is trained on (DataConfig.forbidden), no setting here is chosen from
#: them, and nothing of the training flow reads them. The validation photos: scratch/diag/coae_val_fovs.py.

FRAMES: Tuple[str, ...] = ('scan', 'input')
SIGNALS: Tuple[str, ...] = ('localphase', 'congruency', 't_weighted')
#: the side of each FM's patch-token grid on its own input (Model.FmTokens checks it against the encoder: crop_size // patch_size)
FM_GRIDS: Dict[str, int] = {'gigapath': 14, 'uni2': 16}
#: Co-figure cells along a token's side: a cell is field_px / (CELLS_A_TOKEN x grid) px -- 8 px for both encoders, phasetiles' cell
CELLS_A_TOKEN: int = 2
#: How the views of a place share their targets: `dep` the Co-figure and cls' targets only among the twins of a slot; `ind` both one
#: for the whole place (the views unaligned); `mix` the Co-figure as `dep`, cls' as `ind`.
MECHANISMS: Tuple[str, ...] = ('dep', 'ind', 'mix')
#: How a target is made of the views that share it: `vref` the ref's own Co-figure; `vavg` the mean of the sharers; `vdist` an NLL
#: against each sharer's own Co-figure.
VARIANTS: Tuple[str, ...] = ('vref', 'vavg', 'vdist')


@dataclass(frozen=True)
class Level:
    """A perturbation level: each quantity is drawn uniformly in +- its value (Sims.draw_geometry)."""
    shift_px: float          # shift on both axes, px of the 256 px tile
    turn_deg: float          # in-plane turn
    scale:    float          # scale 1 +- this
    k1:       float          # radial distortion


LEVELS: Dict[str, Level] = {'0': Level(0.0, 0.0, 0.0, 0.0), 'light': Level(8.0, 3.0, 0.03, 0.05), 'large': Level(64.0, 6.0, 0.20, 0.10)}


@dataclass(frozen=True)
class FmConfig:
    """The foundation model and its layer mix: the patch tokens of `layers`, each LayerNorm'd, summed with softmax weights whose start is
    `init` (relative, normalised; empty = equal) and whose share is capped by `cap` (per layer; empty = none). `layers` names the
    blocks as `locascope.core.networks.Heads.resolve_encoder_layers` reads them: a FLOAT is a share of the encoder's depth in (0, 1]
    (block round(r x depth), 1-based), an INT an absolute 0-based block index (negatives from the end); resolved against the encoder's
    own depth in `Model.FmTokens` (`blocks`), so one spec serves every encoder. They must come in ascending block order: `init` and
    `cap` pair with them by position. `lora` adds rank-`lora_rank` adapters to every block's attention qkv (the arms 3, 4); else the FM
    is frozen (arms 1, 2)."""
    encoder:   str               = 'gigapath'
    # NO real photo sets anything here (the user, 2026-11-07): the layers, their start and their caps may come only from scans and their
    # simulations. Until such a measurement exists: every tenth of the depth, equal start, no cap.
    # every tenth of the depth: gigapath (40 blocks) -> blocks 4, 8 .. 40 (1-based), uni2 (24) -> 2, 5, 7, 10, 12, 14, 17, 19, 22, 24
    layers:    Tuple[Union[int, float], ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    init:      Tuple[float, ...] = ()
    cap:       Tuple[float, ...] = ()
    lora:      bool              = False
    lora_rank: int               = 8
    lora_alpha: float            = 16.0
    # the FM's weight dtype: 'bf16' (the weights cast to bfloat16, no autocast in the FM) or 'fp32' (float32 weights under the caller's bf16
    # autocast). absent = fp32 (every checkpoint before 2026-11-15); bf16 is given explicitly by `--fm-dtype` (CoaeTrain.sh does it for the
    # experiment arms). LoRA needs float32 weights.
    weights_dtype: str           = 'fp32'

    def __post_init__(self) -> None:
        if self.weights_dtype not in ('fp32', 'bf16'):
            raise ValueError(f"fm.weights_dtype must be 'fp32' or 'bf16', got {self.weights_dtype!r}")
        if self.lora and self.weights_dtype == 'bf16':
            raise ValueError("fm.lora needs fm.weights_dtype 'fp32': the adapters train on float32 weights")


@dataclass(frozen=True)
class ModelConfig:
    """The AE: one Swin stage on the FM's own token grid g x g (window g / 2, the second block shifted; width swin_dim), a 2 x 2 patch
    merging to the latent s x s x latent_dim (s = g / 2) with a residual from the pooled tokens; the ConvNeXt decoder s -> 2s -> 4s =
    cofig.out_cells (no skip connection); attn_pool (one learned query) and the cross-attention (q from attn_pool, K and V the latent's
    cells, no residual from q) to cls' (`cls_dim`, 0 = the FM's size), and the source discriminator (domain confusion) on the latent
    and on cls'."""
    latent_dim:     int  = 128                                    # channels of a latent cell
    swin_dim:       int  = 384                                    # the width of the Swin stage on the token grid
    swin_depth:     int  = 2                                      # blocks of the one stage; odd ones (0-based) shifted: 2 = Swin's pair W-MSA, SW-MSA
    swin_heads:     int  = 8
    decoder_depth:  int  = 2                                      # ConvNeXt blocks a resolution
    decoder_dim:    Tuple[int, ...] = (384, 192, 96)              # the channels at s, 2s, 4s (s = the latent's side)
    pool_heads:     int  = 8
    cls_dim:        int  = 0                                      # cls' size; 0 = the FM's own (gigapath 1536): no projection of its CLS
    cross_heads:    int  = 8
    disc_hidden:    int  = 512
    disc_on_cls:    bool = True
    cls_residual:   bool = False                                  # cls' = the FM's own CLS, untouched + gain x the attention branch (gain from 0): no W, no alpha
    noise:          float = 0.5                                   # relative isotropic Gaussian noise on the latent and on cls' in training, a regulariser of where their power goes (0 = off)


@dataclass(frozen=True)
class CoFigureConfig:
    """The Co-figure: the signals of a canvas at every pixel (phasetiles' settings). Training: each view's OWN Co-figure on the FM's
    field of its centre tile, `out_cells` x `out_cells` cells (CoFigure.field_maps); no warp, no average over geometries. `cell`,
    `align` and `frame` belong to the aligned path (CoFigure.place_targets), which training does not take now: `cell` px cells of the
    tile, aligned to the scan's frame by each version's known geometry (`align`; False = the blurred variant, the versions averaged
    unaligned); `frame`: where a version's target is read off the place's average -- 'scan': the tile at the version's shift, the
    scan's axes and scale; 'input': the cells of the version's own input tile, mapped back by its whole known geometry."""
    signals:   Tuple[str, ...]   = SIGNALS
    freqs:     Tuple[float, ...] = (8.0, 24.0)                   # local phase, cycles / 256 px
    pc_scales: Tuple[float, ...] = (4.0, 8.0, 16.0, 32.0)        # phase congruency
    lg_sigma:  float             = 0.55
    taper_px:  int               = 32
    t_band:    Tuple[int, int]   = (1, 8)                        # T weighted: rings (cycles / 256 px)
    cell:      int               = 8
    tile:      int               = 256
    align:     bool              = True
    frame:     str               = 'scan'
    out_cells: int               = 28                                # the side of the targets and of the decoder's output: CELLS_A_TOKEN x the FM's token grid (CoaeConfig checks), spread over the FM's field


@dataclass(frozen=True)
class DataConfig:
    """The batch: `places` places of the training slides' scans, no two of the whole job's batch overlapping (in level-0 px, across
    GPUs and levels), each as `slots` groups of views (slot 0 canonical, slot 1 perturbed at `level`; `level` '0' = only slot 0), a
    view a source (the scan and `sims` simulated, photometric only) in each group: places x slots x (1 + sims) tiles a step. The
    views of a group are twins (one geometry, one perturbation drawn for all of them) when `mechanism` is `dep` or `mix`; with `ind`
    each source of the perturbed group draws its own. The training slides: the `stain` slides of `dataset` (the virtual
    `@pure-train`: ki67_pure's train split, 100 Ki67 slides); a slide of `forbidden` (ki67_with_photo: its photos are the validation)
    is refused. `scale`, `turn_deg` and `shift_px` are the OUTER FRAME: the footprint of a place and `read_px` follow them, and no
    perturbation level (LEVELS) may exceed them.

    THE PLACES (Supply): every place `sampler` (a TileSampler recipe) takes on the disjoint lattice of each rung ds 1, 2, 4 ..
    `max_ds` of each slide, inside the `mask` recipe's tissue; a place's footprint is the tile and the largest shift on both sides
    (256 + 2 x 64 = 384 px at its ds), so a shifted version never reaches a neighbour's place, and its richness is scored over that
    footprint, not the centre tile. The draws are cached in `draw_cache`. THE POOL: a (slide, rung) cell keeps at most `pool_cap` of
    its drawn places for the whole training (0 = every place drawn), spread over its tissue (`Supply.pool_runs`, the offset seeded by
    `pool_seed`), each one screened at start: its reference read lies on the slide and its centre tile is not blank (grey sd >=
    `blank_sd`) (`Supply.pool_runs`, `Supply.screened_pool`); a cell keeps as many as pass, none is an error; training takes places
    from the pool only.
    AN EPOCH: each (slide, rung) cell lists ceil(its pool / `cell_parts`) of its places (at most `cell_epoch`), the least seen so far
    first (seen = among a step's places, counted across epochs; ties by the longest unseen, then at random). A step lists `places`; a cell's
    budget is spread evenly over the epoch's sum(budgets) // that steps (a running credit, floor + largest remainder, ties by the
    step's seed), so every step has the same expected mix -- the pool's, as the cells' budgets give it; the rungs are NOT evened out by
    the sampling (ds 32 has fewer places than a step holds), the loss weights do that. No two places of a step on one slide meet (the
    coarse rungs list first, a meeting place goes back to its cell's queue). A step that cannot be filled ends the epoch. An epoch is
    the shuffle's boundary and the unit of training length (`TrainConfig.epochs`).
    A rung's few places are not repeated to make up its share: its loss weighs more instead (`level_weights`: in each process's step,
    a place's weight goes as 1 / sqrt(the places of its rung in that step), mean 1 over the step's places -- a rung's share of the
    step's loss goes as sqrt(its places there); the rungs absent from the step take nothing)."""
    dataset:     str               = '@pure-train'
    stain:       str               = 'Ki67'
    forbidden:   Tuple[str, ...]   = ('ki67_with_photo',)
    slides:      int               = 0                           # the first this many training slides (0 = all); a test checks the flow on a few
    max_ds:      int               = 32
    mask:        str               = 'uni2_pca'                  # TissueMaskConfig.MASK_RECIPES: where tissue may be
    sampler:     str               = 'reference-bank'            # TileSampler.SAMPLER_RECIPES: its richness gate (background < 50 %)
    cell_epoch:  int               = 1000                        # an epoch lists at most this many places of a (slide, rung) cell, the least seen so far first; a cell with fewer lists them all
    draw_cache:  str               = 'CoaeTrain'                 # whose cache the draws are in
    level_weights: bool            = True                        # a place's reconstruction loss x M / (sqrt(its rung's places in this process's step) x the sum of the step's rungs' sqrt(places))
    pool_cap:    int               = 128                         # a (slide, rung) cell keeps at most this many of its drawn places for the whole training, spread over its tissue (Supply.pool_runs, screened at start); 0 = every place drawn. Experiments 128; the formal arm 512
    pool_seed:   int               = 0                           # which such pool (the offset of the pick)
    cell_parts:  int               = 1                           # an epoch lists ceil(a cell's pool / this) of its places (and at most cell_epoch), the least seen first: the pool goes round once in this many epochs
    places:      int               = 128                         # places a step, over all GPUs (128 a GPU); a place is slots x (1 + sims) views
    sims:        Tuple[str, ...]   = ('bench', 'bench-real', 'ad-chroma4')
    level:       str               = 'large'                     # the perturbation level of slot 1 (LEVELS); '0' = no perturbed group
    mechanism:   str               = 'mix'                       # MECHANISMS
    scale:       float             = 0.20                        # the outer frame (see above)
    turn_deg:    float             = 6.0
    shift_px:    float             = 64.0
    blank_sd:    float             = 3.0                         # grey sd of a place's centre tile below this = background: screened out of the pool at start (ExpCountBlankTiles)
    photo_px:    int               = 640                         # a version's photo canvas (the input is its centre tile)
    processes:   int               = 10                          # worker processes: each reads its places' reference squares and draws their views' parameters (>= 1); a process a GPU's CPU, two left for the training process

    @property
    def slots(self) -> int:
        """The groups of views of a place: 1 (canonical only) or 2 (canonical and perturbed)."""
        return 1 if self.level == '0' else 2

    @property
    def twins(self) -> bool:
        """True: the sources of a group share one geometry (`dep`, `mix`); False (`ind`): each source of the perturbed group its own."""
        return self.mechanism != 'ind'

    @property
    def place_shared(self) -> bool:
        """True: cls' has one target for the whole place (`mix`, `ind`); False (`dep`): one a slot."""
        return self.mechanism != 'dep'

    @property
    def dilate_px(self) -> float:
        """How far cls' target's cells are widened: the level's shift when the target is shared by the place, else 0."""
        return LEVELS[self.level].shift_px if self.place_shared else 0.0


@dataclass(frozen=True)
class HeadConfig:
    """The decoder's target (`variant`: vref the ref's own Co-figure, vavg the mean of the views that share it, vdist an NLL against
    each sharer's own Co-figure -- `output` 'dist': a location and a spread a signal), the losses' weights, the confusion loss's ramp
    (0 -> `adv_max` over `adv_ramp_epochs` epochs). By default every output has its content loss: `w_recon` (the Co-figure, > 0) and `w_pool` (a
    linear head on cls' predicts the place descriptor, CoFigure.shared_moments, over `pool_cells` x `pool_cells` cells; MSE; >= 0, and
    0 is the ablation: the loss is still computed and logged, times 0, so cls' has no content loss). `w_nce` (>= 0) is the target
    contrast on the linear head (needs one descriptor a place and `w_pool` > 0); `w_vnce` (>= 0) is the view-to-view contrast on cls',
    for the experiment arms only (spec.md §5), and `w_recon` may be 0 (an experiment arm: the local branch is a contrast or none, `xP2`); `tau` is
    the contrasts' temperature. The substitution arms (experiment arms only, spec.md §5) put a contrast in the place of a content loss:
    `w_hnce` (>= 0) the view-to-view contrast on the linear head's output, `w_dnce` and `w_lnce` (>= 0) the twin contrast on the decoder's
    output blocks and on the latent (`twin_cells` cell positions drawn a step; they need the twins of at least two sources).
    `w_adv` and `w_koleo` are the source-directed losses: 0 = the discriminators are not built, the losses not computed."""
    variant:      str   = 'vref'
    w_recon:      float = 1.0
    w_koleo:      float = 0.0
    w_adv:        float = 0.0
    w_pool:       float = 1.0                                     # cls' -> the place descriptor
    w_nce:        float = 0.0     # target contrast on the linear head: a view's head against the place descriptors (its own = the positive; the batch's other places and the bank's places of its (slide, rung) cell = the negatives)
    w_vnce:       float = 0.0     # EXPERIMENT ARMS ONLY (spec.md §5): view-to-view contrast on cls' (the place's other views = the positives)
    tau:          float = 0.1     # the temperature of the contrasts
    w_hnce:       float = 0.0     # EXPERIMENT ARMS ONLY: view-to-view contrast on the linear head's output (the place's other views = the positives): the descriptor loss's place, a contrast instead
    w_dnce:       float = 0.0     # EXPERIMENT ARMS ONLY: twin contrast on the decoder's output, a latent cell's block of it (4 x 4 cells x N channels): the reconstruction's place, a contrast instead
    w_lnce:       float = 0.0     # EXPERIMENT ARMS ONLY: the same twin contrast on the latent itself
    twin_cells:   int   = 8       # cell positions drawn a step for the twin contrasts (the same for the whole batch)
    pool_cells:   int   = 4                                       # the descriptor's side
    adv_max:      float = 1.0
    adv_ramp_epochs: float = 3.0                                  # the confusion loss's ramp 0 -> adv_max, in epochs (x the supply's steps an epoch)
    beta_nll:     float = 0.5                                     # beta-NLL (0 = plain NLL)

    @property
    def output(self) -> str:
        """The decoder's output form: 'dist' for vdist, else 'point'."""
        return 'dist' if self.variant == 'vdist' else 'point'


@dataclass(frozen=True)
class TrainConfig:
    epochs:        int   = 3                                      # an epoch is the supply's steps (Supply.steps_an_epoch: at most 448 at pool_cap 128, cell_parts 1; at most 359 at pool_cap 512, cell_parts 4: fewer where the screen drops places); 0 = build the probe, measure once, stop
    lr:            float = 3e-4
    lr_lora:       float = 1e-4
    weight_decay:  float = 0.05
    warmup_epochs: float = 1.0                                    # linear warm-up, in epochs (a fraction is fine: x Supply.steps_an_epoch, rounded, at least 1 step); train.py --warmup-epochs
    grad_ckpt:     bool  = True                                   # checkpoint the FM's blocks when LoRA trains
    seed:          int   = 0


@dataclass(frozen=True)
class ProbeConfig:
    """The probe (spec.md §6): places of slides that are never trained on and have no real photo, each as fixed views."""
    dataset: str               = '@pure-half-val'
    slides:  int               = 0                       # the first this many of its `data.stain` slides (0 = all: 10 Ki67)
    rungs:   Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    places:  int               = 64                      # a (slide, rung) cell; a cell with fewer is left out
    ood:     Tuple[str, ...]   = (                       # sources no arm trains on (Sims.parse_variant on bench-real's base)
        'ood-heavy|defocus_radius=3|noise_sigma=6.0|chromatic_shift=5|color_temp_range=-0.25..0.25',
        'ood-soft|blur_factor=1.5..2.5|noise_corr=1.5|noise_sigma=2.0')
    seed:    int               = 12345


@dataclass(frozen=True)
class LossTerm:
    """One loss of HeadConfig, as an arm's name sees it. The name and the training share this table."""
    key:      str            # the HeadConfig weight field without 'w_': 'recon', 'pool', 'nce', ...
    letter:   str            # its letter in an arm's name
    branch:   str            # 'local' (latent -> decoder) | 'global' (cls' -> linear head)
    standard: float          # the weight a name does not spell
    rides_on: str = ''       # a term that only exists on top of another: 'nce' rides on 'pool' (together they are written `N`)


#: in the order the letters are written (the local branch's one, then the global branch's)
LOSS_TERMS: Tuple[LossTerm, ...] = (
    LossTerm('recon', 'R', 'local', 1.0), LossTerm('dnce', 'D', 'local', 0.5), LossTerm('lnce', 'L', 'local', 0.5),
    LossTerm('pool', 'P', 'global', 1.0), LossTerm('nce', 'N', 'global', 0.5, rides_on='pool'),
    LossTerm('hnce', 'H', 'global', 0.5), LossTerm('vnce', 'V', 'global', 0.5))

#: the simulated sources an arm of that many has when `data.sims` is not spelt in its name
DEFAULT_SIMS: Dict[int, Tuple[str, ...]] = {3: ('bench', 'bench-real', 'ad-chroma4'), 2: ('bench', 'bench-real'), 0: ()}

#: how `data.level` is written when it is not 'large'
LEVEL_NAMES: Dict[str, str] = {'0': 'noperturb', 'light': 'lightperturb'}


#: the simulators the sources are named by (Sims.sim_labs), as the names spell them
SIM_NAMES: Tuple[str, ...] = ('bench-real', 'bench', 'ad-chroma4')
SIM_LIST: str = '(?:' + '|'.join(SIM_NAMES) + ')(?:\\+(?:' + '|'.join(SIM_NAMES) + '))*'     # `a+b`: a regex for the sources of `sims-<a+b>`
NUMBER: str = '\\d+(?:\\.\\d+)?'

#: what a name part sets when it is read back: (config section, field) -> value
Updates = Dict[Tuple[str, str], Any]


@dataclass(frozen=True)
class NamePart:
    """One optional part of a name (arm or directory), in a table in the order the parts are written: written when `get(cfg)` is not
    `default`; read back by `pattern` (one group: what `put` is given) -> the settings it sets."""
    get:     Callable[[CoaeConfig], Any]      # the value
    default: Any                              # not written when equal
    text:    Callable[[Any], str]             # how it is written
    pattern: str                              # a regex of the written part, one capturing group
    put:     Callable[[str], Updates]         # the group -> the settings


def _weight_of(c: CoaeConfig, key: str) -> float:
    return float(getattr(c.head, f'w_{key}'))


def _weight_part(t: LossTerm) -> NamePart:
    """A loss that is on with a weight other than its standard one is written `weight<letter><w>`; an off loss reads as its standard."""
    return NamePart(lambda c: _weight_of(c, t.key) if _weight_of(c, t.key) > 0 else t.standard, t.standard,
                    lambda v: f'weight{t.letter}{v:g}', f'weight{t.letter}({NUMBER})', lambda g: {('head', f'w_{t.key}'): float(g)})


def _encoder_updates(name: str) -> Updates:
    """An encoder's setting: the Co-figure's cells follow the FM's token grid (as train.make_cfg)."""
    return {('fm', 'encoder'): name, ('cofig', 'out_cells'): CELLS_A_TOKEN * FM_GRIDS.get(name, 0)}


def _sims_of(text: str) -> Updates:
    return {('data', 'sims'): tuple(text.split('+'))}


def _sims_not_default(c: CoaeConfig) -> Optional[Tuple[str, ...]]:
    """The sources when they are not the fixed set of their number (None: not written)."""
    sims: Tuple[str, ...] = tuple(c.data.sims)
    return None if sims == DEFAULT_SIMS.get(len(sims)) else sims


#: the modifiers, in the order they are written
NAME_PARTS: Tuple[NamePart, ...] = (
    NamePart(lambda c: c.data.mechanism, 'mix', str, '(dep|ind)', lambda g: {('data', 'mechanism'): g}),
    NamePart(lambda c: c.head.variant, 'vref', str, '(vdist|vavg)', lambda g: {('head', 'variant'): g}),
    NamePart(lambda c: c.model.cls_residual, False, lambda v: 'fmresidual', '(fmresidual)', lambda g: {('model', 'cls_residual'): True}),
    NamePart(lambda c: c.data.level, 'large', lambda v: LEVEL_NAMES[v], '(noperturb|lightperturb)',
             lambda g: {('data', 'level'): {n: k for k, n in LEVEL_NAMES.items()}[g]}),
    NamePart(lambda c: c.head.pool_cells, 4, lambda v: f'desc{v}', 'desc(\\d+)', lambda g: {('head', 'pool_cells'): int(g)}),
    NamePart(lambda c: c.model.noise, 0.5, lambda v: f'noise{v:g}', f'noise({NUMBER})', lambda g: {('model', 'noise'): float(g)}),
    NamePart(lambda c: c.fm.lora, False, lambda v: 'lora', '(lora)', lambda g: {('fm', 'lora'): True}),
    NamePart(_sims_not_default, None, lambda v: 'sims-' + '+'.join(v), f'sims-({SIM_LIST})', _sims_of),
    *(_weight_part(t) for t in LOSS_TERMS),
    NamePart(lambda c: c.data.pool_cap, 128, lambda v: f'pool{v}', 'pool(\\d+)', lambda g: {('data', 'pool_cap'): int(g)}),
    NamePart(lambda c: c.data.slides, 0, lambda v: f'slides{v}', 'slides(\\d+)', lambda g: {('data', 'slides'): int(g)}),
    NamePart(lambda c: c.train.epochs, 3, lambda v: f'epochs{v}', 'epochs(\\d+)', lambda g: {('train', 'epochs'): int(g)}),
    NamePart(lambda c: c.train.seed, 0, lambda v: f'seed{v}', 'seed(\\d+)', lambda g: {('train', 'seed'): int(g)}))


def _vnce_only(c: CoaeConfig) -> bool:
    """The view contrast on cls' alone: weight 1, no reconstruction, no descriptor loss (the directory's `vnceonly`)."""
    return c.head.w_vnce == 1.0 and c.head.w_recon == 0 and c.head.w_pool == 0


def _dir_weight(key: str) -> Callable[[CoaeConfig], float]:
    """The reconstruction's / the descriptor loss's weight as the directory spells it (`vnceonly` spells both as 0)."""
    return lambda c: 1.0 if _vnce_only(c) else _weight_of(c, key)


def _flag_part(key: str, word: str) -> NamePart:
    """A contrast that is on with its standard weight: the word."""
    standard: float = {t.key: t.standard for t in LOSS_TERMS}[key]
    return NamePart(lambda c: _weight_of(c, key) > 0, False, lambda v: word, f'({word})', lambda g: {('head', f'w_{key}'): standard})


#: the directory's parts (after `<level>-<mechanism>-<variant>`), in the order they are written: the names the training's directories
#: have always had (result/CoaeTrain/<dir>), plus `slides` and `epochs` (an arm's epochs are in the code, so in the directory too)
DIR_PARTS: Tuple[NamePart, ...] = (
    NamePart(lambda c: tuple(c.data.sims), DEFAULT_SIMS[3], lambda v: 'sims-' + '+'.join(v) if v else 'refonly',
             f'(refonly|sims-{SIM_LIST})', lambda g: {('data', 'sims'): () if g == 'refonly' else tuple(g[5:].split('+'))}),
    NamePart(lambda c: c.model.noise, 0.5, lambda v: f'noise{v:g}', f'noise({NUMBER})', lambda g: {('model', 'noise'): float(g)}),
    NamePart(lambda c: c.head.pool_cells, 4, lambda v: f'pool{v}', 'pool(\\d+)', lambda g: {('head', 'pool_cells'): int(g)}),
    NamePart(lambda c: c.fm.encoder, 'gigapath', str, '(' + '|'.join(FM_GRIDS) + ')', _encoder_updates),
    NamePart(lambda c: c.fm.lora, False, lambda v: 'lora', '(lora)', lambda g: {('fm', 'lora'): True}),
    NamePart(lambda c: c.model.cls_residual, False, lambda v: 'fmcls', '(fmcls)', lambda g: {('model', 'cls_residual'): True}),
    NamePart(_dir_weight('recon'), 1.0, lambda v: f'wrecon{v:g}', f'wrecon({NUMBER})', lambda g: {('head', 'w_recon'): float(g)}),
    NamePart(_dir_weight('pool'), 1.0, lambda v: f'wpool{v:g}', f'wpool({NUMBER})', lambda g: {('head', 'w_pool'): float(g)}),
    _flag_part('nce', 'nce'),
    NamePart(lambda c: 'only' if _vnce_only(c) else 'on' if c.head.w_vnce > 0 else 'off', 'off', lambda v: 'vnceonly' if v == 'only' else 'vnce',
             '(vnceonly|vnce)', lambda g: {('head', 'w_vnce'): 1.0, ('head', 'w_recon'): 0.0, ('head', 'w_pool'): 0.0} if g == 'vnceonly'
             else {('head', 'w_vnce'): 0.5}),
    _flag_part('dnce', 'dnce'), _flag_part('lnce', 'lat'), _flag_part('hnce', 'hnce'),
    NamePart(lambda c: c.train.seed, 0, lambda v: f'seed{v}', 'seed(\\d+)', lambda g: {('train', 'seed'): int(g)}),
    NamePart(lambda c: c.data.pool_cap, 128, lambda v: f'cap{v}', 'cap(\\d+)', lambda g: {('data', 'pool_cap'): int(g)}),
    NamePart(lambda c: c.data.slides, 0, lambda v: f'slides{v}', 'slides(\\d+)', lambda g: {('data', 'slides'): int(g)}),
    NamePart(lambda c: c.train.epochs, 3, lambda v: f'epochs{v}', 'epochs(\\d+)', lambda g: {('train', 'epochs'): int(g)}))


def read_parts(parts: Tuple[NamePart, ...], text: str, sep: str) -> Updates:
    """The settings the parts of `text` (each `sep` + a part, in the table's order) set; ValueError for text that is not such parts."""
    updates: Updates = {}
    pos: int = 0
    for p in parts:
        found: Optional[re.Match[str]] = re.compile(re.escape(sep) + p.pattern + '(?=' + re.escape(sep) + '|$)').match(text, pos)
        if found is not None:
            updates.update(p.put(found.group(1)))
            pos = found.end()
    if pos != len(text):
        raise ValueError(f'{text[pos:]!r} is not a part of a name, or it is out of order')
    return updates


def build_config(updates: Updates) -> CoaeConfig:
    """The default config with `updates` set, made once (so the config checks see the whole)."""
    base: CoaeConfig = CoaeConfig()
    sections: Dict[str, Dict[str, Any]] = {}
    for (section, name), value in updates.items():
        sections.setdefault(section, {})[name] = value
    return replace(base, **{s: replace(getattr(base, s), **kw) for s, kw in sections.items()})


@dataclass(frozen=True)
class Unnamed:
    """A setting the name cannot say: it must stay at its default, or the arm has no name."""
    what:    str
    get:     Callable[[CoaeConfig], Any]
    default: Any


UNNAMED: Tuple[Unnamed, ...] = (
    Unnamed('head.tau', lambda c: c.head.tau, HeadConfig.tau), Unnamed('head.twin_cells', lambda c: c.head.twin_cells, HeadConfig.twin_cells),
    Unnamed('head.w_koleo', lambda c: c.head.w_koleo, HeadConfig.w_koleo), Unnamed('head.w_adv', lambda c: c.head.w_adv, HeadConfig.w_adv))


@dataclass(frozen=True)
class CoaeConfig:
    fm:     FmConfig       = field(default_factory=FmConfig)
    model:  ModelConfig    = field(default_factory=ModelConfig)
    cofig:  CoFigureConfig = field(default_factory=CoFigureConfig)
    data:   DataConfig     = field(default_factory=DataConfig)
    head:   HeadConfig     = field(default_factory=HeadConfig)
    train:  TrainConfig    = field(default_factory=TrainConfig)
    probe:  ProbeConfig    = field(default_factory=ProbeConfig)

    def __post_init__(self) -> None:
        if self.data.level not in LEVELS:
            raise ValueError(f'data.level must be one of {tuple(LEVELS)}, got {self.data.level!r}')
        if self.data.mechanism not in MECHANISMS:
            raise ValueError(f'data.mechanism must be one of {MECHANISMS}, got {self.data.mechanism!r}')
        if self.head.variant not in VARIANTS:
            raise ValueError(f'head.variant must be one of {VARIANTS}, got {self.head.variant!r}')
        for name, lv in LEVELS.items():
            if lv.shift_px > self.data.shift_px or lv.turn_deg > self.data.turn_deg or lv.scale > self.data.scale:
                raise ValueError(f'level {name!r} {lv} exceeds the outer frame (data.shift_px, turn_deg, scale = '
                                 f'{self.data.shift_px}, {self.data.turn_deg}, {self.data.scale})')
        weights: Tuple[float, ...] = (self.head.w_recon, self.head.w_pool, self.head.w_nce, self.head.w_vnce, self.head.w_hnce,
                                      self.head.w_dnce, self.head.w_lnce)
        if not (all(w >= 0 for w in weights) and self.head.tau > 0):
            raise ValueError(f'head.w_recon, w_pool, w_nce, w_vnce, w_hnce, w_dnce, w_lnce must be >= 0 (w_pool 0 = the descriptor-loss '
                             f'ablation: cls\' then has no content loss) and head.tau > 0, got {weights}, {self.head.tau}')
        if not any(w > 0 for w in weights):
            raise ValueError(f'at least one of head.w_recon, w_pool, w_nce, w_vnce, w_hnce, w_dnce, w_lnce must be > 0: no loss, nothing to train, got {weights}')
        if self.head.w_nce > 0 and not (self.data.place_shared and self.head.w_pool > 0):
            raise ValueError('head.w_nce needs one descriptor a place (mechanism mix or ind) and the descriptor loss on (w_pool > 0: it sets the head\'s units)')
        if (self.head.w_dnce > 0 or self.head.w_lnce > 0) and not (self.data.twins and len(self.data.sims) >= 1):
            raise ValueError('head.w_dnce and w_lnce need twins of at least two sources (mechanism dep or mix, data.sims not empty)')
        if self.head.w_dnce > 0 and self.head.output != 'point':
            raise ValueError("head.w_dnce needs the decoder's point output (variant vref or vavg): its blocks are contrasted")
        if self.fm.encoder not in FM_GRIDS:
            raise ValueError(f'fm.encoder must be one of {tuple(FM_GRIDS)}, got {self.fm.encoder!r}')
        if self.token_grid % 2:
            raise ValueError(f'the token grid must be even: the latent is its 2 x 2 merging, got {self.token_grid}')
        if not 1 <= self.head.twin_cells <= self.latent_side ** 2:
            raise ValueError(f'head.twin_cells must be in 1..{self.latent_side ** 2} (the latent\'s cells), got {self.head.twin_cells}')
        if self.cofig.out_cells != CELLS_A_TOKEN * self.token_grid:
            raise ValueError(f'cofig.out_cells {self.cofig.out_cells} must be {CELLS_A_TOKEN} x the token grid {self.token_grid} of '
                             f'{self.fm.encoder} (= {CELLS_A_TOKEN * self.token_grid}): a token is {CELLS_A_TOKEN} x {CELLS_A_TOKEN} cells')
        if self.latent_side * 2 ** (len(self.model.decoder_dim) - 1) != self.cofig.out_cells:
            raise ValueError(f'model.decoder_dim {self.model.decoder_dim}: one width a resolution s, 2s .. cofig.out_cells '
                             f'(s = {self.latent_side}, out_cells {self.cofig.out_cells})')
        if self.model.swin_dim % self.model.swin_heads:
            raise ValueError(f'model.swin_dim {self.model.swin_dim} must be a multiple of model.swin_heads {self.model.swin_heads}')
        if self.cofig.out_cells % self.head.pool_cells != 0:
            raise ValueError(f'cofig.out_cells {self.cofig.out_cells} must be a multiple of head.pool_cells {self.head.pool_cells}')
        if self.model.noise < 0:
            raise ValueError(f'model.noise must be >= 0, got {self.model.noise}')
        if self.head.variant == 'vdist':
            sharers: int = (1 if self.data.twins else self.data.slots) * (1 + len(self.data.sims))
            if sharers < 2:
                raise ValueError(f'head.variant vdist needs at least 2 views sharing a target, got {sharers} '
                                 f'(mechanism {self.data.mechanism}, level {self.data.level}, sims {self.data.sims})')
        bad = [s for s in self.cofig.signals if s not in SIGNALS]
        if bad:
            raise ValueError(f'unknown Co-figure signals {bad}; known {SIGNALS}')
        if self.cofig.frame not in FRAMES:
            raise ValueError(f'cofig.frame must be one of {FRAMES}, got {self.cofig.frame!r}')
        if self.cofig.frame == 'input' and not self.cofig.align:
            raise ValueError("cofig.frame 'input' needs cofig.align: the unaligned average has no known frame to map back from")
        if not self.fm.layers:
            raise ValueError('fm.layers is empty: the layer mix needs at least one block')
        if self.fm.init and len(self.fm.init) != len(self.fm.layers):
            raise ValueError(f'fm.init needs one value a layer: {len(self.fm.init)} for {len(self.fm.layers)} layers')
        if self.fm.cap and len(self.fm.cap) != len(self.fm.layers):
            raise ValueError(f'fm.cap needs one value a layer: {len(self.fm.cap)} for {len(self.fm.layers)} layers')
        if (self.train.epochs < 0 or self.train.warmup_epochs < 0 or self.head.adv_ramp_epochs < 0 or self.data.cell_epoch <= 0):
            raise ValueError(f'train.epochs, warmup_epochs and adv_ramp_epochs must be >= 0 and data.cell_epoch > 0, got '
                             f'{self.train.epochs}, {self.train.warmup_epochs}, {self.head.adv_ramp_epochs}, {self.data.cell_epoch}')
        if self.train.epochs > 0 and self.train.warmup_epochs > self.train.epochs:
            raise ValueError(f'train.warmup_epochs {self.train.warmup_epochs} is longer than the run ({self.train.epochs} epochs): '
                             f'the lr would never reach its peak')
        if self.data.pool_cap < 0 or self.data.cell_parts < 1:
            raise ValueError(f'data.pool_cap must be >= 0 and data.cell_parts >= 1, got {self.data.pool_cap}, {self.data.cell_parts}')
        if self.data.processes < 1:
            raise ValueError(f'data.processes must be >= 1 (the workers read every place), got {self.data.processes}')
        if self.probe.places < 3:
            raise ValueError(f'probe.places must be >= 3 (the sd of a query\'s scores over the other anchors needs 2), got {self.probe.places}')

    @property
    def token_grid(self) -> int:
        """The FM's patch-token grid side g."""
        return FM_GRIDS[self.fm.encoder]

    @property
    def latent_side(self) -> int:
        """The latent's side s = g / 2."""
        return self.token_grid // 2

    @property
    def canvases(self) -> int:
        """The canvases of each slot the targets read on the GPU: those of its first `canvases` sources. The targets are all that read a
        canvas: under `vref` only the ref's (CoFigure.targets_of_views, shared_moments) -> 1; else every source's -> 1 + len(sims).
        A view's input tile is made apart (train.Views.tiles)."""
        return 1 if self.head.variant == 'vref' else 1 + len(self.data.sims)

    @property
    def arm(self) -> str:
        """The arm's name (spec.md): `<local><global><sources>` then the modifiers of NAME_PARTS that are off their default, each after a '.'.
        The local letter is the one loss of the local branch (`x`: none), the global letters are those of the global branch's losses in
        LOSS_TERMS' order (`x`: none; a term that rides on another is written instead of it); the sources are how many simulated sources the
        views have. A setting the name cannot say (UNNAMED) off its default, or two losses on the local branch: no name (ValueError)."""
        self.check_nameable()
        on: Dict[str, LossTerm] = {t.key: t for t in LOSS_TERMS if _weight_of(self, t.key) > 0}
        ridden: Tuple[str, ...] = tuple(t.rides_on for t in on.values() if t.rides_on)
        local: List[str] = [t.letter for t in on.values() if t.branch == 'local']
        glob: List[str] = [t.letter for t in on.values() if t.branch == 'global' and t.key not in ridden]
        if len(local) > 1:
            raise ValueError(f'an arm name has one loss on the local branch, got {local}')
        sources: str = str(len(self.data.sims))
        mods: List[str] = [f'.{p.text(v)}' for p in NAME_PARTS for v in (p.get(self),) if v != p.default]
        return (local[0] if local else 'x') + (''.join(glob) or 'x') + sources + ''.join(mods)

    @property
    def dir_name(self) -> str:
        """The run's directory name (result/<job>/<dir>, log/<job>_<dir>): `<level>-<mechanism>-<variant>` then the parts of DIR_PARTS that are
        off their default, each after a '-'. A loss weight the directory cannot spell (a contrast off its standard weight, but for the view
        contrast alone) and the settings of UNNAMED: no name (ValueError)."""
        self.check_nameable()
        for t in LOSS_TERMS:
            if t.key not in ('recon', 'pool') and _weight_of(self, t.key) > 0 and _weight_of(self, t.key) != t.standard and not _vnce_only(self):
                raise ValueError(f'w_{t.key} = {_weight_of(self, t.key)} is not its standard {t.standard}: a directory name cannot say it')
        return f'{self.data.level}-{self.data.mechanism}-{self.head.variant}' + ''.join(
            f'-{p.text(v)}' for p in DIR_PARTS for v in (p.get(self),) if v != p.default)

    @property
    def last_checkpoint(self) -> str:
        """`epoch<k>.pt` of the run's last epoch (train.py writes one an epoch): the one definition of "the run's checkpoint". A run of 0
        epochs (the probe only) has none (ValueError)."""
        if self.train.epochs < 1:
            raise ValueError('a run of 0 epochs writes no checkpoint')
        return f'epoch{self.train.epochs - 1}.pt'

    def check_nameable(self) -> None:
        """ValueError when a setting no name can say (UNNAMED) is off its default."""
        for u in UNNAMED:
            if u.get(self) != u.default:
                raise ValueError(f'{u.what} = {u.get(self)!r} is not its default {u.default!r}: a name cannot say it')


ARM_CORE: 're.Pattern[str]' = re.compile(r'([RDLx])(x|[PN]H?V?|HV?|V)(\d+)')


def config_of_arm(arm: str, encoder: str = 'gigapath') -> CoaeConfig:
    """The config an arm's code names (the inverse of CoaeConfig.arm); ValueError for a text that is not the canonical code of a config.
    The FM is not in the code (`RN3` is the same arm of every FM): the caller gives it as `encoder` (stage 2: the `<Fm>` prefix of
    `<Fm>CoAE+<arm>`), one of FM_GRIDS, else ValueError."""
    if encoder not in FM_GRIDS:
        raise ValueError(f'encoder must be one of {tuple(FM_GRIDS)}, got {encoder!r}')
    found: Optional[re.Match[str]] = ARM_CORE.match(arm)
    if found is None:
        raise ValueError(f'{arm!r}: an arm code starts with <local><global><sources> (e.g. RP3)')
    local, glob, sources = found.groups()
    updates: Updates = {('head', f'w_{t.key}'): 0.0 for t in LOSS_TERMS}
    on: List[LossTerm] = [t for t in LOSS_TERMS if (t.letter == local if t.branch == 'local' else t.letter in glob)]
    for t in on:
        updates[('head', f'w_{t.key}')] = t.standard
        if t.rides_on:
            updates[('head', f'w_{t.rides_on}')] = {u.key: u.standard for u in LOSS_TERMS}[t.rides_on]
    if int(sources) in DEFAULT_SIMS:
        updates[('data', 'sims')] = DEFAULT_SIMS[int(sources)]
    updates.update(read_parts(NAME_PARTS, arm[found.end():], '.'))
    updates.update(_encoder_updates(encoder))
    cfg: CoaeConfig = build_config(updates)
    if cfg.arm != arm:
        raise ValueError(f'{arm!r} is not the code of a config (its canonical code is {cfg.arm!r})')
    return cfg


DIR_BASE: 're.Pattern[str]' = re.compile(r'(0|light|large)-(dep|ind|mix)-(vref|vavg|vdist)')


def config_of_dir(name: str) -> CoaeConfig:
    """The config a run's directory name names (the inverse of CoaeConfig.dir_name); ValueError for a text that is not the name of a config."""
    found: Optional[re.Match[str]] = DIR_BASE.match(name)
    if found is None:
        raise ValueError(f'{name!r}: a directory name starts with <level>-<mechanism>-<variant> (e.g. large-mix-vref)')
    updates: Updates = {('data', 'level'): found.group(1), ('data', 'mechanism'): found.group(2), ('head', 'variant'): found.group(3)}
    updates.update(read_parts(DIR_PARTS, name[found.end():], '-'))
    cfg: CoaeConfig = build_config(updates)
    if cfg.dir_name != name:
        raise ValueError(f'{name!r} is not the directory name of a config (its canonical name is {cfg.dir_name!r})')
    return cfg
