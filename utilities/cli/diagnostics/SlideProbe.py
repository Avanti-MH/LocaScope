"""Shared low-level WSI-probing primitives for this directory's diagnostics.

Lives IN `cli/diagnostics/`, not up in `utilities/`, because every caller so
far is a diagnostic in this directory -- the production pipeline
(`SafeSlide`, `TissueMask`, ...) has its own, DELIBERATELY DIFFERENT
way of reading a slide's scale (`SafeSlide.base_mpp` averages mpp-x/mpp-y
with an aperio fallback and is built to survive a bad read). `scan_wsi_holes`
and `diag_mask_validity` read `openslide.mpp-x` RAW, on purpose -- they hunt
the failures `SafeSlide` exists to paper over, so going through it here would
undo the reason those two tools open a slide directly. This module does not
try to unify that difference; `get_mpp`'s richer vendor-fallback chain is
here for whichever tool wants it, not forced on the two that don't.
"""
from __future__ import annotations

import openslide


class SlideProbe:
    """An OpenSlide handle that survives a failed read by replacing itself.

    openslide checks its error state on EVERY call, so one read that lands on
    a tile the scanner never wrote kills the handle permanently -- not just
    for reads but for metadata too: `level_count` and `level_downsamples`
    raise the same error afterwards. A probe loop sharing one handle
    therefore reports every point after the first hole as unreadable, and
    the resulting map measures nothing but where the first failure was.

    Replacing the handle after each failure is what makes the map real.
    Reopening re-parses the slide's own index (small on MIRAX's Index.dat),
    so the cost scales with the number of holes, not the number of probes.
    """

    def __init__(self, path: str):
        self.path = path
        self.osl = openslide.OpenSlide(path)
        self.reopens = 0

    def close(self):
        try:
            self.osl.close()
        except Exception:
            pass

    def readable(self, x: int, y: int, level: int, size: tuple) -> bool:
        try:
            self.osl.read_region((x, y), level, size)
            return True
        except Exception:
            self.close()
            self.osl = openslide.OpenSlide(self.path)
            self.reopens += 1
            return False


def bounds_rect(wsi, whole_canvas: bool = False) -> tuple:
    """`(bx, by, bw, bh, scope)` -- the rectangle worth scanning.

    `scope='bounds'` (default) is `openslide.bounds-*`: the outer envelope
    of the cells a MIRAX scanner actually photographed. `scope='canvas'` is
    the full slide, used when `whole_canvas=True` or the slide carries no
    `bounds-*` at all (an SVS doesn't -- it has no pre-scan step to leave a
    ragged envelope behind).
    """
    W0, H0 = wsi.dimensions
    p = wsi.properties
    if whole_canvas or p.get('openslide.bounds-width') is None:
        return 0, 0, W0, H0, 'canvas'
    return (int(p['openslide.bounds-x']), int(p['openslide.bounds-y']),
           int(p['openslide.bounds-width']), int(p['openslide.bounds-height']),
           'bounds')


def get_mpp(props) -> tuple:
    """`(mpp_x, mpp_y)` in micrometres/pixel, trying vendor-specific keys as
    fallback -- `openslide.mpp-x/y` covers most formats, but not every slide
    this project has seen sets it. `(0.0, 0.0)` when nothing matches, not a
    raise: this is a diagnostic reading whatever the slide happens to carry,
    not a step that needs the value to proceed.
    """
    mpp_x = props.get('openslide.mpp-x')
    mpp_y = props.get('openslide.mpp-y')
    if mpp_x and mpp_y:
        return float(mpp_x), float(mpp_y)

    if 'aperio.MPP' in props:
        v = float(props['aperio.MPP'])
        return v, v

    mx = props.get('mirax.LAYER_0_LEVEL_0_SECTION.MICROMETER_PER_PIXEL_X')
    my = props.get('mirax.LAYER_0_LEVEL_0_SECTION.MICROMETER_PER_PIXEL_Y')
    if mx and my:
        return float(mx), float(my)

    # Hamamatsu (.ndpi) usually sets the standard key too, hence the second
    # `.get` here rather than a dedicated hamamatsu.* branch.
    mx = props.get('openslide.mpp-x')
    if mx:
        return float(mx), float(props.get('openslide.mpp-y', mx))

    xres = props.get('tiff.XResolution')
    yres = props.get('tiff.YResolution')
    unit = props.get('tiff.ResolutionUnit')   # '2'=inch, '3'=cm
    if xres and yres and unit:
        factor = 25400.0 if unit == '2' else 10000.0   # micrometres per inch / cm
        return factor / float(xres), factor / float(yres)

    return 0.0, 0.0
