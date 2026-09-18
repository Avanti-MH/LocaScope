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
                    tag: str) -> str:
    '''`<encoder>_<frozen|finetuned>_<head>_<last|best|best_unweighted>.pt`.

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

    `best_unweighted`: `best` is picked on the POOLED val accuracy across
    both eval datasets (`torch.cat` before scoring, so a dataset with more
    val positions gets proportionally more say -- 2026-09-18, BRACS had 1094
    val positions against Ki67's 847). `best_unweighted` is picked on the
    UNWEIGHTED MEAN of each dataset's own accuracy instead, so both datasets
    get equal say regardless of how many positions each happened to
    contribute. Two separate files, not one flag, because they can diverge
    (the epoch that wins pooled is not always the epoch that wins balanced),
    and a caller wanting either one names it directly instead of re-deriving
    which epoch would have won under the other's weighting.
    '''
    if tag not in ('last', 'best', 'best_unweighted'):
        raise ValueError(
            f"tag must be 'last', 'best' or 'best_unweighted', got {tag!r}")
    kind = 'frozen' if frozen else 'finetuned'
    return f'{encoder_name}_{kind}_{head_name}_{tag}.pt'


def save_checkpoint(path, *, head: Head, encoder, encoder_name: str,
                    frozen: bool, head_name: str, head_cfg: HeadConfig,
                    epoch: int, val: Dict, run_args: Dict,
                    extra: Optional[Dict[str, Any]] = None,
                    optimizer_state: Optional[Dict[str, Any]] = None) -> Path:
    '''Everything needed to rebuild this model AND resume training it.
    `build_from_checkpoint` rebuilds the model; a caller that also wants to
    resume training reads `optimizer_state`/`epoch` off the same dict itself
    (`training/MppRoutingHead/cli/train.py`'s `_maybe_resume` is the one that
    does, since building an optimizer is a training-loop concern this
    generic format has no opinion about, same reasoning as `extra` below).

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
    (or chooses not to) -- `_maybe_resume` treats that as "no optimizer state
    to resume", not an error, and just starts Adam fresh on top of the loaded
    weights.
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
