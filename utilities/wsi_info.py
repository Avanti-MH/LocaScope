#!/usr/bin/env python3
"""
Usage:
    python wsi_info.py <wsi_path>
"""

import argparse
import openslide


def get_mpp(props):
    """Extract MPP (µm/px) from slide properties, trying vendor-specific keys as fallback."""

    # Standard key — OpenSlide normalizes this for most formats
    mpp_x = props.get('openslide.mpp-x')
    mpp_y = props.get('openslide.mpp-y')
    if mpp_x and mpp_y:
        return float(mpp_x), float(mpp_y)

    # Aperio (.svs) — sometimes only has aperio.MPP
    if 'aperio.MPP' in props:
        v = float(props['aperio.MPP'])
        return v, v

    # MIRAX (.mrxs)
    mx = props.get('mirax.LAYER_0_LEVEL_0_SECTION.MICROMETER_PER_PIXEL_X')
    my = props.get('mirax.LAYER_0_LEVEL_0_SECTION.MICROMETER_PER_PIXEL_Y')
    if mx and my:
        return float(mx), float(my)

    # Hamamatsu (.ndpi)
    mx = props.get('hamamatsu.XOffsetFromSlideCentre')   # not reliable, skip
    mx = props.get('openslide.mpp-x')                    # ndpi usually sets standard key
    if mx:
        return float(mx), float(props.get('openslide.mpp-y', mx))

    # Generic tiled TIFF — compute from resolution tags
    xres = props.get('tiff.XResolution')
    yres = props.get('tiff.YResolution')
    unit = props.get('tiff.ResolutionUnit')   # '2'=inch, '3'=cm
    if xres and yres and unit:
        factor = 25400.0 if unit == '2' else 10000.0   # µm per inch / cm
        return factor / float(xres), factor / float(yres)

    return 0.0, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wsi_path')
    args = ap.parse_args()

    wsi = openslide.OpenSlide(args.wsi_path)
    p   = wsi.properties

    mpp_x, mpp_y = get_mpp(p)
    W0, H0 = wsi.level_dimensions[0]

    print(f'\n{"─"*52}')
    print(f'  File     : {args.wsi_path}')
    print(f'  Format   : {p.get("openslide.vendor", "unknown")}')
    print(f'  Levels   : {wsi.level_count}')
    print(f'  L0 size  : {W0} x {H0} px')
    if mpp_x:
        print(f'  MPP (x,y): {mpp_x:.4f}, {mpp_y:.4f} µm/px')
        print(f'  L0 FoV   : {W0*mpp_x/1000:.2f} x {H0*mpp_y/1000:.2f} mm')
    else:
        print(f'  MPP      : not found in metadata')

    # ── Per-level table ───────────────────────────────────────────────────────
    print(f'\n  {"Level":>5}  {"Width":>10}  {"Height":>10}  {"Downsample":>10}  {"MPP-x":>8}')
    print(f'  {"─"*5}  {"─"*10}  {"─"*10}  {"─"*10}  {"─"*8}')
    for lv in range(wsi.level_count):
        W, H = wsi.level_dimensions[lv]
        ds   = wsi.level_downsamples[lv]
        mpp  = f'{mpp_x * ds:.4f}' if mpp_x else 'N/A'
        print(f'  {lv:>5}  {W:>10}  {H:>10}  {ds:>10.2f}  {mpp:>8}')

    # ── All properties ────────────────────────────────────────────────────────
    print(f'\n  {"─"*52}')
    print('  All properties:')
    for k, v in sorted(p.items()):
        print(f'    {k} = {v}')
    print()


if __name__ == '__main__':
    main()
