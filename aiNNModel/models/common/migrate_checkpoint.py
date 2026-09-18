#!/usr/bin/env python3
"""Rename a checkpoint's `head_state` keys from an old module-naming scheme
to the current one -- a ONE-TIME, EXACT key rename per classifier, not a
compatibility shim in production code (`Checkpoints.py`/`ClassifierEstMpp.py`
stay strict; a checkpoint that needs this gets migrated once, here, instead).

Generic over WHICH classifier changed names -- not MlpHead-specific.
`_RENAME_RECIPES` is a registry, one entry per classifier that has ever
renamed its internal modules; add an entry the next time some OTHER
architecture does the same, and nothing else about this tool needs to
change. The one entry that exists today:

    MlpHead used to be a single `nn.Sequential` (`net.0`/`net.3`); it is now
    `in_proj`/`hidden`/`out_proj` (2026-09-18, added to make depth/width/
    residual configurable). The two are IDENTICAL at `mlp_depth=1` -- the
    only architecture any pre-2026-09-18 MlpHead checkpoint was ever trained
    with -- so this is a pure rename of 4 tensors, not a guess:

        classify.net.0.weight   ->   classify.in_proj.weight
        classify.net.0.bias     ->   classify.in_proj.bias
        classify.net.3.weight   ->   classify.out_proj.weight
        classify.net.3.bias     ->   classify.out_proj.bias

A checkpoint whose `classifier` has no recipe here (LinearHead, ArcFaceHead,
AttentionPoolHead never renamed anything) is reported unchanged and left
alone.

SAFE BY DEFAULT:
  - dry run unless --write is passed: prints what WOULD change, writes
    nothing.
  - --write makes a `.bak` copy (original bytes, untouched) before
    overwriting, so the rename is reversible even after the fact.
  - Before writing, the remapped state dict is verified by actually building
    the real classifier class (`Heads.classifier_class(ckpt['classifier'])`)
    from `ckpt['head_cfg']` as stored, and calling
    `load_state_dict(strict=True)` on it -- if that does not load clean, the
    file is not touched. A cheap check in front of an irreversible-feeling
    action, not a hope that the rename was right.

Usage:
    python aiNNModel/models/common/migrate_checkpoint.py \\
        result/MppRoutingHead/weights/*mlp*.pt              # dry run
    python aiNNModel/models/common/migrate_checkpoint.py \\
        result/MppRoutingHead/weights/*mlp*.pt --write       # actually rename
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _dir in (_HERE, _HERE.parent):
    _path = str(_dir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch                                                        # noqa: E402

from Head import Head                                                # noqa: E402
from Heads import HeadConfig, classifier_class                      # noqa: E402

#: classifier name -> ((old key, new key), ...), keys as they ACTUALLY sit
#: in `ckpt['head_state']` -- that is `head.state_dict()` of the whole
#: `Head` wrapper (`Checkpoints.save_checkpoint`), not the bare classifier
#: alone, so every key carries the `classify.` prefix `Head.__init__`'s
#: `self.classify = classifier(cfg)` puts there. One entry per classifier
#: that has ever renamed its internal modules. LinearHead/ArcFaceHead/
#: AttentionPoolHead are absent on purpose -- they never have.
_RENAME_RECIPES = {
    'mlp': (
        ('classify.net.0.weight', 'classify.in_proj.weight'),
        ('classify.net.0.bias',   'classify.in_proj.bias'),
        ('classify.net.3.weight', 'classify.out_proj.weight'),
        ('classify.net.3.bias',   'classify.out_proj.bias'),
    ),
}


def _remap(classifier_name: str, head_state: dict):
    """The renamed `head_state`, or `None` if this classifier has no recipe
    or this checkpoint is already new-format. Raises if the recipe's old
    keys are only PARTIALLY present -- that means this checkpoint is not the
    plain old shape the recipe describes, and guessing the rest would be
    exactly the kind of silent-wrong-answer this tool exists to avoid."""
    recipe = _RENAME_RECIPES.get(classifier_name)
    if not recipe:
        return None
    old_keys = {k for k, _ in recipe}
    present = old_keys & head_state.keys()
    if not present:
        return None
    if present != old_keys:
        raise ValueError(
            f'expected all of {sorted(old_keys)}, found only '
            f'{sorted(present)} -- this checkpoint is not the plain old '
            f'{classifier_name!r} shape this recipe knows how to rename')
    out = dict(head_state)
    for old, new in recipe:
        out[new] = out.pop(old)
    return out


def _verify(classifier_name: str, reduction: str, new_head_state: dict,
           head_cfg: dict) -> None:
    """Builds the REAL `Head` wrapper -- not the bare classifier alone,
    since `new_head_state`'s keys carry the `classify.` (and, for
    `reduction='attn'`, `pool.`) prefix `Head.__init__` puts there, the same
    shape `ckpt['head_state']` has always been saved in -- from `head_cfg`
    exactly as the checkpoint stored it (no override -- a config dict that
    never had e.g. `mlp_depth` already defaults to the shape the rename
    assumes), and loads the remapped state dict into it with strict=True.
    Raises on any mismatch; the caller does not write the file if this
    raises."""
    cfg = HeadConfig(**head_cfg)
    Head(cfg, reduction, classifier_class(classifier_name)).load_state_dict(
        new_head_state, strict=True)


def migrate_one(path: str, write: bool) -> str:
    ckpt = torch.load(path, map_location='cpu')
    classifier_name = ckpt.get('classifier')
    if classifier_name not in _RENAME_RECIPES:
        return f'{path}: classifier={classifier_name!r} has no rename recipe -- unchanged'
    if 'head_state' not in ckpt:
        return f'{path}: no head_state at all (a different stale-format issue) -- skipped'

    try:
        new_head_state = _remap(classifier_name, ckpt['head_state'])
    except ValueError as exc:
        return f'{path}: [SKIP] {exc}'
    if new_head_state is None:
        return f'{path}: already new-format -- unchanged'

    _verify(classifier_name, ckpt['reduction'], new_head_state, ckpt['head_cfg'])
    n_renamed = len(_RENAME_RECIPES[classifier_name])

    if not write:
        return (f'{path}: [DRY RUN] would rename {n_renamed} keys '
               f'(verified loadable) -- pass --write to apply')

    backup = path + '.bak'
    shutil.copy2(path, backup)
    ckpt['head_state'] = new_head_state
    torch.save(ckpt, path)
    return f'{path}: renamed and verified. Original backed up at {backup}'


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+', help='checkpoint file(s)')
    ap.add_argument('--write', action='store_true',
                    help='actually rewrite the file (with a .bak backup) -- '
                         'default is dry run, nothing written')
    args = ap.parse_args()

    status = 0
    for path in args.paths:
        try:
            print(migrate_one(path, args.write))
        except Exception as exc:                                   # noqa: BLE001
            print(f'{path}: [FAIL] {type(exc).__name__}: {exc}')
            status = 1
    return status


if __name__ == '__main__':
    sys.exit(main())
