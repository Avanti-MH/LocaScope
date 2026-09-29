'''The checkpoint format one `Head` (plus, for a fine-tuned run, its trunk)
is saved to and rebuilt from. Generic: nothing here knows what the classes a
`Head` predicts MEAN (rungs, or anything else) -- a caller that wants that
recorded passes it through `extra`.
'''
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

_HERE = Path(__file__).resolve().parent
for _dir in (_HERE, _HERE.parent):
    _path = str(_dir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch                                                        # noqa: E402

from Head import Head                                               # noqa: E402
from Heads import HeadConfig, classifier_class, classifier_name     # noqa: E402


def weight_filename(encoder_name: str, frozen: bool, head_name: str,
                    tag: str, loss: str = 'bal') -> str:
    '''`<encoder>_<frozen|finetuned>_<head>_<last|best|best_unweighted>.pt`,
    with a `_<loss>` segment before the tag when `loss` is not `'bal'`
    (2026-09-22).

    The tissue-mask recipe is NOT in the name. It is recorded in
    `ckpt['args']['seg']` and evaluation refuses a mismatch, but two runs that
    differ only in it overwrite each other: at this stage the recipe is not an
    axis being compared, and a name segment would say it was.

    `loss='bal'` gets NO segment, not `_bal`: `bal` was every checkpoint's
    only loss before `cli/train.py`'s `--loss` existed, so leaving it
    untagged keeps every already-trained bal checkpoint's filename exactly
    as it is -- only `ord_a`/`ord_b` runs (which used to need a hand-picked
    `--out` subdirectory to avoid overwriting a bal checkpoint of the same
    encoder+head, see `jobscripts/Benchmarks/Stage1MppBench.sh`'s
    history, formerly MppFeatureDecomposition.sh) now get a real, distinct
    filename in the SAME shared
    weights/ directory. `loss` is a training-time hyperparameter, not an
    architectural choice `build_from_checkpoint` needs to rebuild the
    model correctly (unlike `training/PrototypicalRoutingHead`'s `collapse`/
    `routing_head`) -- it is already recorded in `ckpt['args']['loss']`
    regardless of what this function does, so this change is naming/
    organisation only, not a `build_from_checkpoint` change.

    ENCODER FIRST, because `_paths.encoder_tag(encoder, head)` already spells
    an encoder-plus-head name that way and CLAUDE.md's result paths follow it
    (`conch_vit_attn_pool`); a second ordering for the same pair of facts is a
    second convention. `encoder` is the REGISTRY name (`convnext_v2`, not
    `ConvNeXt`), same rule.

    `frozen|finetuned` is the field that answers "what weights are in here":
    a frozen run trains the head alone, so the file holds `head_state` and
    `trunk_state=None` -- the trunk is the published checkpoint, reachable by
    name. A finetuned run holds BOTH, in one file rather than two, because
    the trunk and the head were trained together and a mismatched pair of
    files is a failure mode worth making impossible. It also keeps two runs
    of the same encoder apart if one is ever used both frozen and fine-tuned.

    `best_unweighted`: `best` is picked on the N-WEIGHTED val accuracy
    across both eval datasets -- each dataset's own number is the plain mean
    of its six rungs' own accuracies (NOT pooled over its tiles: 2026-09-21,
    the caller switched away from that because a tile-pooled average is
    dominated by whichever rungs happen to have the most val tiles), and the
    two datasets' numbers are then combined weighted by each one's own total
    n, so a dataset with more val positions still gets proportionally more
    say (2026-09-18, BRACS had 1094 val positions against Ki67's 847).
    `best_unweighted` is picked on the UNWEIGHTED MEAN of each dataset's own
    accuracy instead, so both datasets get equal say regardless of how many
    positions each happened to contribute. Two separate files, not one flag,
    because they can diverge (the epoch that wins n-weighted is not always
    the epoch that wins balanced), and a caller wanting either one names it
    directly instead of re-deriving which epoch would have won under the
    other's weighting. (This function itself does not compute either number
    -- it only names the file. The formula lives in whichever training
    package's own `val_report` calls it, currently `MppRoutingHead/
    cli/train.py`'s.)
    '''
    if tag not in ('last', 'best', 'best_unweighted'):
        raise ValueError(
            f"tag must be 'last', 'best' or 'best_unweighted', got {tag!r}")
    kind = 'frozen' if frozen else 'finetuned'
    loss_seg = '' if loss == 'bal' else f'{loss}_'
    return f'{encoder_name}_{kind}_{head_name}_{loss_seg}{tag}.pt'


def save_checkpoint(path, *, head: Head, encoder, encoder_name: str,
                    frozen: bool, head_name: str, head_cfg: HeadConfig,
                    epoch: int, val: Dict, run_args: Dict,
                    extra: Optional[Dict[str, Any]] = None,
                    optimizer_state: Optional[Dict[str, Any]] = None) -> Path:
    '''Everything needed to rebuild this model. `build_from_checkpoint`
    rebuilds it; `optimizer_state`/`epoch` ride along for a caller that wants
    to fine-tune further from it. Resuming an interrupted RUN is a different
    file with more in it (RNG, scheduler, every model of the run) --
    `Resume.py`.

    `reduction`/`classifier` are read OFF THE LIVE `head` OBJECT
    (`head.reduction`, `classifier_name(type(head.classify))`), not looked up
    from a registry by `head_name` -- the object already knows what it is,
    and a second lookup that could in principle disagree with what was
    actually built is a failure mode this avoids by construction. `classifier`
    is stored as its REGISTERED NAME (`'arcface'`), not `type(...).__name__`
    (`'ArcFaceHead'`) -- see `Heads.classifier_name`'s own docstring for why.

    `extra` is nested under its own key, not spread into the top level: a
    caller's field could otherwise collide with one of this format's own
    (`epoch`, `val`, ...), silently overwriting it. Whatever a specific task
    needs recorded -- which label space the classes mean, what a render was
    configured with -- goes here rather than being a field this generic
    format has an opinion about.

    `optimizer_state` is `None` when the caller has no optimizer to save
    (or chooses not to).
    '''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        head_state=head.state_dict(),
        trunk_state=None if frozen else encoder.model.state_dict(),
        encoder=encoder_name,
        frozen=frozen,
        head_name=head_name,
        reduction=head.reduction,
        classifier=classifier_name(type(head.classify)),
        head_cfg=asdict(head_cfg),
        epoch=epoch,
        val=val,
        args=run_args,
        extra=extra or {},
        optimizer_state=optimizer_state,
    ), path)
    return path


def build_from_checkpoint(path, device):
    '''`(head, encoder, ckpt)` -- the model this checkpoint describes, on
    `device`, in `eval()` mode.

    The encoder is built from its REGISTRY NAME and then, for a fine-tuned
    run, has its trained weights loaded over the published ones. A frozen run
    loads nothing into the trunk, which is the point of it being frozen: the
    published checkpoint IS the trunk, and storing a copy per head would have
    been several GB saying nothing.

    THE DTYPE THE ENCODER IS BUILT AT MATTERS, and building it at the
    registry's own default before `load_state_dict` is how a fine-tuned trunk
    would silently lose precision: `nn.Module.load_state_dict` casts via
    `Tensor.copy_` with NO error and NO warning on a dtype mismatch, so
    building at whatever a registry defaults to and then loading a
    DIFFERENTLY-typed `trunk_state` on top would downcast (or upcast) every
    trained weight on load, silently. Forced to `'fp32'` unconditionally for
    a fine-tuned checkpoint (a fine-tuned trunk here is always trained fp32,
    full stop -- see `Heads.HeadConfig`'s docstring for why). A frozen
    checkpoint is built at the exact dtype its own `args` recorded, for
    exactness, though the measured effect of getting this one wrong is small
    (a frozen encoder's own inference exits always cast their OUTPUT back to
    fp32 regardless of internal compute dtype; only the internal autocast
    path would differ).
    '''
    from dataclasses import replace                                 # noqa: PLC0415
    from TileEncoderFunc import encoder_config                      # noqa: PLC0415
    ckpt = torch.load(path, map_location='cpu')
    base_cfg = encoder_config(ckpt['encoder'])
    dtype = 'fp32' if not ckpt['frozen'] else ckpt['args'].get('dtype',
                                                               base_cfg.model.dtype)
    cfg = replace(base_cfg, model=replace(base_cfg.model, dtype=dtype))
    encoder = cfg.build(device)
    if not ckpt['frozen']:
        encoder.model.load_state_dict(ckpt['trunk_state'])
    encoder.eval()

    head_cfg = HeadConfig(**ckpt['head_cfg'])
    head = Head(head_cfg, ckpt['reduction'], classifier_class(ckpt['classifier']))
    head.load_state_dict(ckpt['head_state'])
    head = head.to(device).eval()
    return head, encoder, ckpt


# ══════════════════════════════════════════════════════════════════════════
#  Prototype-routing checkpoints (training/PrototypicalRoutingHead) -- NEW
#  functions, the two above UNCHANGED. save_checkpoint/build_from_checkpoint
#  are shaped around ONE encoder + ONE Head (single input, single classify
#  pass) and have real callers today (MppRoutingHead, ClassifierEstMpp) this
#  must not disturb. A prototype-routing model has FIVE trained modules
#  (pooling, support_context, query_context, collapse, head) instead of
#  one Head -- Stage 2's three sub-stages and Stage 3 (the routing head)
#  are genuinely separate, independently-swappable pieces here
#  (PrototypicalRoutingHead/spec.md's own "which arm is the main line"
#  axis), not one (reduction, classifier) pair -- so it needs its own
#  save/load shape, not a forced fit into the existing one. See that
#  spec.md's "Open design decisions", 1.
#
#  `collapse` (renamed from `generator` 2026-09-22, same day
#  `PrototypeChoices.COLLAPSE_CHOICES` was). `support_context`/`query_
#  context` (G/F, `aiNNModel/models/ContextEncoders.py`) got their OWN
#  top-level state/cfg fields the same day they were added TO this
#  function's signature, not the `extra` dict they briefly went through
#  first: they train real weights (`bilstm`/`attnlstm`) exactly like
#  `collapse` already can, so treating them as second-class (a caller
#  reading `extra['support_context_state']` by string key, no default
#  guaranteed) while `pooling`/`collapse`/`head` get real parameters was
#  an inconsistency, not a deliberate design -- `extra` stays for facts
#  that are genuinely outside this format's own opinion (`cross_domain_
#  dataset`), not for a FOURTH and FIFTH trained module.
# ══════════════════════════════════════════════════════════════════════════

def save_prototype_checkpoint(path, *, pooling, support_context=None,
                              query_context=None, collapse=None, head,
                              encoder, encoder_name: str, frozen: bool,
                              pooling_cfg, support_context_cfg=None,
                              query_context_cfg=None, collapse_cfg=None,
                              support_context_name: Optional[str] = None,
                              query_context_name: Optional[str] = None,
                              collapse_name: Optional[str] = None,
                              routing_head_name: str,
                              epoch: int, val: Dict, run_args: Dict,
                              extra: Optional[Dict[str, Any]] = None,
                              optimizer_state: Optional[Dict[str, Any]] = None
                              ) -> Path:
    '''`pooling`/`support_context`/`query_context`/`collapse`/`head`: the
    trained `nn.Module`s Stage 1-2-3 actually built (Pooling, e.g.
    BiLstmSupportContext, e.g. AttnLstmQueryContext, e.g.
    SetTransformerPrototype, e.g. CosineTauHead). `pooling_cfg`/`support_
    context_cfg`/`query_context_cfg`/`collapse_cfg`: their own dataclass
    configs (`asdict`'d the same way `save_checkpoint` already does for
    `head_cfg`) -- `head` (`CosineTauHead` or a later Stage 3 arm) carries
    no separate cfg object of its own so far; if one ever does, add a
    `head_cfg` key the same way rather than overloading these two.

    `support_context`/`query_context`/`collapse` and their `_cfg` siblings
    are all OPTIONAL (2026-09-21 for `collapse`, 2026-09-22 for the other
    two) -- Stage 4's own BASELINE training arm (`cli/train_baseline.py`,
    deleted 2026-09-22, its own inference-time positioning never having
    settled) has no episode, so there was no support set for any Stage-2
    sub-stage to act on at all in principle; it still built real, zero-
    parameter `identity`/`identity`/`mean` instances for these three
    rather than leaving them `None`, specifically so its OWN checkpoints
    came out the same SHAPE as `cli/train.py`'s. Left OPTIONAL here for
    whatever replaces it.

    `support_context_name`/`query_context_name`/`collapse_name`/
    `routing_head_name` (2026-09-22, promoted OUT of `extra`): which
    registry entry (`PrototypeChoices.SUPPORT_CONTEXT_CHOICES`/
    `QUERY_CONTEXT_CHOICES`/`COLLAPSE_CHOICES`/`ROUTING_HEAD_CHOICES` key)
    built each module -- REQUIRED to even USE the matching `_state`: a
    state dict alone cannot say which class it came from, and for the
    zero-parameter entries (`identity`/`mean`/`off`) it provably CANNOT be
    inferred from the state dict's own content either (all three produce
    an identical empty `state_dict()`). This is exactly the same
    necessity `save_checkpoint`'s own `head_name`/`reduction`/`classifier`
    fields already answer for its single-Head world -- both live at the
    TOP LEVEL there, not in `extra`, and this format was inconsistent with
    its own sibling for not doing the same. `extra` now holds only facts
    genuinely outside what is needed to rebuild the model (`cross_domain_
    dataset`, `tile_size`/`rungs`) -- the same escape hatch `save_
    checkpoint`'s own `extra` already is, and the same reason the trained
    WEIGHTS themselves are top-level fields rather than a caller-supplied
    `extra` entry: neither the weights nor the name of the class they
    belong to is a "task-specific fact", both are what this function
    exists to persist.
    '''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        pooling_state=pooling.state_dict(),
        support_context_state=(support_context.state_dict()
                               if support_context is not None else None),
        query_context_state=(query_context.state_dict()
                             if query_context is not None else None),
        collapse_state=collapse.state_dict() if collapse is not None else None,
        head_state=head.state_dict(),
        trunk_state=None if frozen else encoder.model.state_dict(),
        encoder=encoder_name,
        frozen=frozen,
        support_context_name=support_context_name,
        query_context_name=query_context_name,
        collapse_name=collapse_name,
        routing_head_name=routing_head_name,
        pooling_cfg=asdict(pooling_cfg),
        support_context_cfg=(asdict(support_context_cfg)
                             if support_context_cfg is not None else None),
        query_context_cfg=(asdict(query_context_cfg)
                           if query_context_cfg is not None else None),
        collapse_cfg=asdict(collapse_cfg) if collapse_cfg is not None else None,
        epoch=epoch,
        val=val,
        args=run_args,
        extra=extra or {},
        optimizer_state=optimizer_state,
    ), path)
    return path


def build_prototype_from_checkpoint(path, device):
    '''`(pooling, support_context, query_context, collapse, head, encoder,
    ckpt)` -- the prototype-routing analogue of `build_from_checkpoint`.
    Same dtype-safety reasoning: a fine-tuned trunk is forced to fp32
    unconditionally, a frozen one is built at the exact dtype its own
    `args` recorded.

    REGISTRY-AWARE (2026-09-22, replacing a version that unconditionally
    rebuilt `SetTransformerPrototype`/`CosineTauHead` regardless of what
    the checkpoint actually held): the WEIGHTS (`support_context_state`/
    `query_context_state`/`collapse_state`/...) are real top-level fields
    (`save_prototype_checkpoint`'s own docstring), and so is WHICH
    REGISTRY NAME built each one (`support_context_name`/`query_context_
    name`/`collapse_name`/`routing_head_name`) -- the same class of fix
    `save_checkpoint`'s own `head_name`/`reduction`/`classifier` already
    got by never living anywhere else: a state dict cannot say which
    class it came from (least of all for the zero-parameter registry
    entries -- `identity`/`mean`/`off` all produce an IDENTICAL empty
    `state_dict()`), so the name is not a "task-specific fact" `extra`
    exists for, it is part of what this format exists to persist, same as
    the weights themselves. Looked up in `PrototypeChoices`, the same
    four registries `cli/train.py` itself builds from -- lives in
    `aiNNModel/models/` alongside this file (NOT in a training package)
    specifically so this import is safe: the generic checkpoint layer
    importing FROM the generic model layer is fine, importing from
    `training/PrototypicalRoutingHead/Runtime.py` would not be (see
    `PrototypeChoices`'s own module docstring for why it moved there).

    NO backward-compatibility fallback to the old `extra['collapse']`-
    style checkpoints (2026-09-22): every checkpoint on disk when the four
    names moved to the top level was retrained from scratch rather than
    kept around, so there was never a real old-format file this function
    needed to still read -- reading the old `extra` key first as a guess
    would have been dead code with no caller to exercise it.
    '''
    from dataclasses import replace                                 # noqa: PLC0415
    from TileEncoderFunc import encoder_config                      # noqa: PLC0415
    from Pooling import Pooling, PoolingConfig                       # noqa: PLC0415
    from PrototypeChoices import (SUPPORT_CONTEXT_CHOICES,            # noqa: PLC0415
                                  QUERY_CONTEXT_CHOICES, COLLAPSE_CHOICES,
                                  ROUTING_HEAD_CHOICES)

    ckpt = torch.load(path, map_location='cpu')
    base_cfg = encoder_config(ckpt['encoder'])
    dtype = 'fp32' if not ckpt['frozen'] else ckpt['args'].get('dtype',
                                                               base_cfg.model.dtype)
    cfg = replace(base_cfg, model=replace(base_cfg.model, dtype=dtype))
    encoder = cfg.build(device)
    if not ckpt['frozen']:
        encoder.model.load_state_dict(ckpt['trunk_state'])
    encoder.eval()
    in_dim = int(encoder.model_spec.dim)

    pooling = Pooling(PoolingConfig(**ckpt['pooling_cfg'])).to(device)
    pooling.load_state_dict(ckpt['pooling_state'])
    pooling.eval()

    support_context, _ = SUPPORT_CONTEXT_CHOICES[
        ckpt['support_context_name']](in_dim, device)
    if ckpt.get('support_context_state') is not None:
        support_context.load_state_dict(ckpt['support_context_state'])
    support_context.eval()

    query_context, _ = QUERY_CONTEXT_CHOICES[
        ckpt['query_context_name']](in_dim, device)
    if ckpt.get('query_context_state') is not None:
        query_context.load_state_dict(ckpt['query_context_state'])
    query_context.eval()

    collapse, _ = COLLAPSE_CHOICES[ckpt['collapse_name']](in_dim, device)
    if ckpt['collapse_state'] is not None:
        collapse.load_state_dict(ckpt['collapse_state'])
    collapse.eval()

    head = ROUTING_HEAD_CHOICES[ckpt['routing_head_name']](in_dim, device)
    head.load_state_dict(ckpt['head_state'])
    head.eval()

    return pooling, support_context, query_context, collapse, head, encoder, ckpt
