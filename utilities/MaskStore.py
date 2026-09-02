"""A whole-slide tissue mask on disk, and what makes one mask not another.

    slide_mask = MaskStore.build_one(wsi, segmenter)
    MaskStore.save(root, slide_mask, MaskMeta.of(slide_mask, wsi, segmenter))

    slide_mask, meta = MaskStore.load(path, require={'method': 'uni2-pca-seg'})
    trm = TissuesRegionsMask.from_mask(wsi, slide_mask.mask,
                                       slide_mask.origin, slide_mask.span)

WHY A STORE AND NOT JUST A CALL
--------------------------------
The mask costs 3.5 to 6 minutes of GPU per slide (measured, `Uni2PcaSegFunc.
LEVEL`), and THREE later steps read it: the sampling probe of spec.md 12 step
3b, the pre-tile extraction of 3c, and any bench that wants the same regions the
training data came from. Recomputing it three times is not the problem -- the
problem is three recomputations that could silently differ, because a segmenter
is a config and a config can move.

So the mask is written once with the identity of the thing that made it, and
every reader can say which one it got.

WHAT MAKES TWO MASKS DIFFERENT
-------------------------------
`_IDENTITY_FIELDS` is three names: the slide, the method, and the segmenter's own
`identity_id()`. That third one is doing the work -- it already folds in the
config, the encoder's weights and the encoder's whole configuration, so nothing
here has to enumerate `fit_tiles` or `background_threshold` and then fail to be
updated when a fourth field appears.

`mask_ds`, `origin` and `span` are NOT identity. They are determined by the
segmenter and the slide, so a change in them without a change in `segmenter_id`
would mean one of the two is lying, and hashing both would hide it rather than
catch it.

SEGMENTER-AGNOSTIC ON PURPOSE
------------------------------
`SlideMask` lives here rather than in `aiNNModel/Uni2PcaSegFunc.py`, where it was
written, for two reasons. The layering one: this module is in `utilities/` and
`utilities/` must not import `aiNNModel/` -- that arrow points the wrong way.
The other one is that a mask is a mask: an hsv mask and a UNI2-PCA mask are the
same array with the same origin and span, and only `method` and `segmenter_id`
say them apart.

`build_one` is narrower than the store, and says so rather than pretending:
it handles the segmenters that read a slide, and points at
`TissuesRegionsMask.from_wsi` for the ones that take an image.

NOT A COPY OF FeatureStore's CODEC
-----------------------------------
`FeatureStore._codec` exists because its metadata carries `Tuple[str, ...]` and
`Optional[Tuple[int, int]]`, and encoding those reversibly is a real function.
Every field here is a str, an int or a float, so the encoding is `str()` and the
decoding is the annotation's own constructor -- six lines, and reaching into
another module's private helper to save them would cost more than it saved.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from safetensors.numpy import save_file
from safetensors import safe_open

#: Bumped when a field changes MEANING. Adding a field with a default does not
#: need it; renaming one or redefining what it holds does, because old and new
#: would otherwise hash the same and hold different things. Same rule as
#: ConfigIdentity's first: a name is never repurposed.
SCHEMA_VERSION = '1'

#: What makes two masks different. See the module docstring on why this is three
#: names and not thirteen.
_IDENTITY_FIELDS = ('wsi_stem', 'method', 'segmenter_id')


class MaskMismatch(RuntimeError):
    """`load(require=...)` was handed a store that is not the one asked for."""


# ── what a mask is ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SlideMask:
    """A whole-slide tissue mask, with the two numbers that let it be placed.

    `mask` alone is not enough to build a `TissuesRegionsMask`:

        origin   where the mask's top-left sits in LEVEL-0 coordinates. Zero on
                 an SVS; on a MIRAX it is `openslide.bounds-*`, because the
                 canvas around the scanned rectangle holds no image data at all.
        span     the LEVEL-0 extent the mask COVERS, which is not the extent that
                 was asked for. A tiler that drops partial tiles at the right and
                 bottom edge covers less: 276 tiles of 224 is 61,824 level-0 px
                 against a 61,879 px scanned rectangle, and dividing the larger
                 by the mask width gives ds 14.012 rather than 14 -- 0.09 percent,
                 and 55 level-0 px of drift by the far edge.

    `mask_ds` is carried rather than derived so a reader does not have to do that
    division to find out how blocky the mask is. `report` is whatever the
    segmenter wanted to say about the fit that produced it.
    """
    mask:    np.ndarray               # [rows, cols] uint8, 1 = tissue
    origin:  Tuple[int, int]          # (x, y) level-0
    span:    Tuple[int, int]          # (w, h) level-0 actually covered
    mask_ds: float                    # level-0 px per mask px
    report:  Optional[dict] = None

    #: [rows, cols, k] float16 -- what the mask is a THRESHOLD of, kept beside
    #: the bit it produced. `test_EoMT.slide_pca_mask:520-522` argued for this
    #: before the mask store existed: "One bit per cell throws away everything
    #: the second and third components hold, and re-deriving them costs the
    #: whole encode again -- hours, against a few hundred MB."
    #:
    #: The immediate use is `background_threshold`, which is undecided and
    #: cannot be decided from a binary mask -- the mask IS the answer being
    #: questioned. With this stored a threshold sweep is seconds; without it,
    #: every candidate costs another 3.5 to 6 minutes of GPU per slide.
    #:
    #: 581 to 814 MB per slide, measured on the two EoMTest runs. Optional
    #: because a reader that only wants the mask should not pay for it --
    #: `load(with_components=False)` reads the header and one tensor.
    components: Optional[np.ndarray] = None

    @property
    def fraction(self) -> float:
        return float(np.asarray(self.mask).astype(bool).mean())

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(np.asarray(self.mask).shape[:2])


# ── what makes one mask not another ───────────────────────────────────────────

@dataclass(frozen=True)
class MaskMeta:
    wsi_stem:       str
    method:         str          # 'uni2-pca-seg' | 'hsv' | 'hest' | ...
    segmenter_id:   str          # the built segmenter's identity_id()

    #: Provenance, not identity. A checkpoint or a slide that moved between
    #: mounts must not invalidate a cache -- same rule as ModelConfig.weights.
    wsi_path:       str = ''

    #: Facts about the array, so a reader can check geometry without loading it.
    mask_ds:        float = 0.0
    origin_x:       int = 0
    origin_y:       int = 0
    span_w:         int = 0
    span_h:         int = 0
    rows:           int = 0
    cols:           int = 0
    fraction:       float = 0.0
    n_components:   int = 0      # 0 when the store holds only the mask

    report_json:    str = ''
    created_at:     str = ''
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def of(cls, slide_mask: SlideMask, wsi, segmenter) -> 'MaskMeta':
        """Read everything off the three things that determine it.

        A classmethod rather than arguments to `save`, so the identity is
        assembled in one place and a caller cannot assemble a different one.
        """
        rows, cols = slide_mask.shape
        return cls(
            wsi_stem=wsi_stem_of(wsi),
            method=getattr(getattr(segmenter, 'cfg', None), 'method', 'unknown'),
            segmenter_id=segmenter.identity_id(),
            wsi_path=str(getattr(wsi, '_filename', '') or ''),
            mask_ds=float(slide_mask.mask_ds),
            origin_x=int(slide_mask.origin[0]), origin_y=int(slide_mask.origin[1]),
            span_w=int(slide_mask.span[0]), span_h=int(slide_mask.span[1]),
            rows=int(rows), cols=int(cols),
            fraction=slide_mask.fraction,
            n_components=(0 if slide_mask.components is None
                          else int(np.asarray(slide_mask.components).shape[-1])),
            report_json=json.dumps(slide_mask.report or {}, sort_keys=True,
                                   default=str),
        )

    def cfg_hash(self) -> str:
        parts = [f'{n}={getattr(self, n)}' for n in sorted(_IDENTITY_FIELDS)]
        return hashlib.sha256('|'.join(parts).encode()).hexdigest()[:8]

    def filename(self) -> str:
        return f'{self.wsi_stem}__{self.method}__{self.cfg_hash()}.safetensors'

    def to_strings(self) -> Dict[str, str]:
        return {f.name: str(getattr(self, f.name))
                for f in dataclasses.fields(self)}

    @classmethod
    def from_strings(cls, d: Dict[str, str]) -> 'MaskMeta':
        # `f.type` is the STRING 'int', not the type -- this module has
        # `from __future__ import annotations`, so every annotation is lazy.
        # Comparing it against the int object silently matched nothing and
        # handed every field back as a str, which then blew up at the first
        # caller that formatted `mask_ds` with `:.0f`.
        _cast = {'int': int, 'float': float, 'str': str}
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name not in d:
                continue          # a field this build no longer has, or gained
            cast = _cast.get(str(f.type).replace('Optional[', '').rstrip(']'), str)
            kwargs[f.name] = cast(d[f.name])
        return cls(**kwargs)

    @property
    def report(self) -> dict:
        return json.loads(self.report_json) if self.report_json else {}


def wsi_stem_of(wsi_or_path) -> str:
    """The slide's name without directory or extension.

    One definition, because it appears in a filename and in a `require=` query,
    and two spellings of a key is a query that silently matches nothing.
    """
    path = (wsi_or_path if isinstance(wsi_or_path, (str, Path))
            else getattr(wsi_or_path, '_filename', '') or '')
    return Path(str(path)).stem


# ── build ─────────────────────────────────────────────────────────────────────

def build_one(wsi, segmenter) -> SlideMask:
    """Run a slide-reading segmenter over one slide. No print, no paths.

    The building logic is here rather than in the CLI so that a bench, a test or
    a later pipeline stage can produce a mask without shelling out -- and so the
    CLI is argparse plus a loop, which is all `cli/build_reference_store.py` is
    next to `FeatureStore`.

    Only the segmenters whose unit of work is a SLIDE. `Uni2PcaSegFunc` is one:
    it fits a PCA across the slide before it can threshold any part of it, so
    there is no image you could hand it that would let it do its job. hsv, otsu
    and hest are image segmenters and go through
    `TissuesRegionsMask.from_wsi(method=...)`, which reads the plane for them.
    """
    mask_wsi = getattr(segmenter, 'mask_wsi', None)
    if mask_wsi is None:
        raise TypeError(
            f'{type(segmenter).__name__} has no mask_wsi, so it is an IMAGE '
            f'segmenter: it turns one array into one mask and cannot read a '
            f'slide. Build its mask with TissuesRegionsMask.from_wsi(wsi, '
            f'method=segmenter) instead -- that reads the plane and hands it '
            f'over, which is what such a segmenter needs and this function '
            f'cannot supply')
    return mask_wsi(wsi)


# ── write ─────────────────────────────────────────────────────────────────────

def save(root, slide_mask: SlideMask, meta: MaskMeta) -> Path:
    """Validate, then write atomically to root/<meta.filename()>.

    Atomic for the reason FeatureStore is: jobs here get killed by walltime, and
    a truncated mask that still loads is worse than no mask -- it would quietly
    drop a strip of tissue and every tile count downstream would come out low
    with nothing to say why.

    Two tensors when the mask carries components, one when it does not. They go
    in the same file rather than a sidecar because they are the same grid at the
    same instant -- a sidecar can be deleted, moved or rebuilt on its own, and
    then the threshold sweep would be reading one slide's components against
    another's mask.
    """
    mask = np.ascontiguousarray(np.asarray(slide_mask.mask))
    if mask.ndim != 2:
        raise ValueError(f'mask must be 2-D, got shape {mask.shape}')
    if (meta.rows, meta.cols) != mask.shape:
        raise ValueError(
            f'meta says {meta.rows}x{meta.cols}, array is '
            f'{mask.shape[0]}x{mask.shape[1]}. Build the meta with '
            f'MaskMeta.of(slide_mask, ...) so the two cannot disagree')
    if not meta.segmenter_id:
        raise ValueError(
            'segmenter_id is empty, so this mask cannot say what made it and '
            'two different segmenters would write the same filename')

    meta = dataclasses.replace(
        meta, created_at=meta.created_at or time.strftime('%Y-%m-%dT%H:%M:%S'))

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / meta.filename()

    tensors = {'mask': mask.astype(np.uint8)}
    if slide_mask.components is not None:
        components = np.ascontiguousarray(
            np.asarray(slide_mask.components, dtype=np.float16))
        if components.shape[:2] != mask.shape:
            raise ValueError(
                f'components are {components.shape[:2]} cells, mask is '
                f'{mask.shape}. They are two views of the same grid, so a '
                f'disagreement means one of them was cropped or transposed')
        if components.shape[-1] != meta.n_components:
            raise ValueError(
                f'meta says {meta.n_components} components, array has '
                f'{components.shape[-1]}. Build the meta with MaskMeta.of()')
        tensors['components'] = components

    tmp = path.with_suffix('.tmp')
    save_file(tensors, str(tmp), metadata=meta.to_strings())
    os.replace(tmp, path)
    return path


# ── read ──────────────────────────────────────────────────────────────────────

def load_meta(path) -> MaskMeta:
    """Metadata only -- reads the header, not the array."""
    with safe_open(str(path), framework='numpy') as handle:
        md = handle.metadata()
    if md is None:
        raise MaskMismatch(f'{path} has no metadata -- not a MaskStore file')
    return MaskMeta.from_strings(md)


def load(path, *, require: Optional[Dict[str, object]] = None,
         with_components: bool = False):
    """Return (SlideMask, MaskMeta), refusing outright when `require` misses.

    Never falls back. A mask that is not the one asked for is an error, because
    the alternative is sampling tiles from the wrong regions and finding out
    from a training curve that looks merely disappointing.

    `with_components` is off by default because the components are 30 to 40x the
    mask on disk and almost every reader wants the bit. It is on for exactly one
    caller -- the threshold sweep, which needs the value the bit came from.
    Asking for them when the store has none is an error rather than a None:
    a sweep that silently found nothing to sweep would report a flat curve.
    """
    meta = load_meta(path)

    if require:
        bad = {k: (v, getattr(meta, k, '<no such field>'))
               for k, v in require.items() if getattr(meta, k, None) != v}
        if bad:
            lines = '\n'.join(f'  {k}: wanted {w!r}, store has {g!r}'
                              for k, (w, g) in sorted(bad.items()))
            raise MaskMismatch(f'{path}\n{lines}')

    if with_components and not meta.n_components:
        raise MaskMismatch(
            f'{path} holds no components -- it was built before they were '
            f'stored, or by a segmenter that has none. Rebuild it with '
            f'utilities/cli/build_mask_store.py --overwrite')

    # safe_open rather than load_file: load_file reads every tensor, which would
    # pull 500-800 MB of components off disk for a reader that asked for the
    # mask -- the exact cost `with_components=False` exists to avoid.
    with safe_open(str(path), framework='numpy') as handle:
        mask = handle.get_tensor('mask')
        components = handle.get_tensor('components') if with_components else None

    return SlideMask(mask=mask,
                     origin=(meta.origin_x, meta.origin_y),
                     span=(meta.span_w, meta.span_h),
                     mask_ds=meta.mask_ds,
                     report=meta.report,
                     components=components), meta


def find(root, *, wsi_stem: Optional[str] = None,
         method: Optional[str] = None, **eq) -> list:
    """Masks under `root` matching the given fields, by metadata not filename.

    Reading each header beats parsing names: the filename carries a config hash
    a caller would otherwise have to recompute, and an identity rule recomputed
    in a second place is an identity rule that drifts.
    """
    want = {k: v for k, v in (('wsi_stem', wsi_stem), ('method', method))
            if v is not None}
    want.update(eq)

    hits = []
    for candidate in sorted(Path(root).glob('*.safetensors')):
        try:
            meta = load_meta(candidate)
        except Exception:                                        # noqa: BLE001
            continue                      # not ours; leave other files alone
        if all(getattr(meta, k, None) == v for k, v in want.items()):
            hits.append(candidate)
    return hits


def find_one(root, **eq) -> Path:
    """The single mask matching `eq`, or an error naming the ones that did.

    `find()[0]` is the tempting shape and the wrong one: a root can legitimately
    hold masks from two segmenters of the same slide -- they differ in
    `segmenter_id` and so coexist -- and picking whichever sorted first would
    sample tiles from a mask nobody chose.
    """
    hits = find(root, **eq)
    if len(hits) == 1:
        return hits[0]
    query = ', '.join(f'{k}={v!r}' for k, v in sorted(eq.items())) or '(no filter)'
    if not hits:
        raise MaskMismatch(
            f'no mask under {root} matching {query}. '
            f'Present: {[p.name for p in sorted(Path(root).glob("*.safetensors"))]}')
    raise MaskMismatch(
        f'{len(hits)} masks under {root} match {query}, and this call needs '
        f'one. Narrow it with segmenter_id=... :\n' +
        '\n'.join(f'  {p.name}   {load_meta(p).segmenter_id}' for p in hits))
