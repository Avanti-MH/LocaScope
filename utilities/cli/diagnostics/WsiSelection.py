"""ONE way to say "which WSIs" across every diagnostic in this directory --
extracted 2026-09-18 alongside `SlideProbe.py`, but deliberately NOT put
there: `SlideProbe.py` is about how to read a slide handle safely, this is
about which slides to hand it in the first place, a different question one
layer up.

Before this, four different answers to "which WSI(s)" existed side by side:
`wsi_info.py` and (the file now named) `diag_mask_validity.py` took exactly
one raw path; `scan_wsi_holes.py` took `nargs='+'` raw paths;
`probe_tile_yield.py` took `--wsi nargs='*'` raw paths defaulting to
whatever a mask store already held; `diag_wsi_scale.py` took `--dataset`
resolved through `AccessDatasets`. `resolve_wsi_paths` is the one function
every tool in this directory now calls instead of repeating its own slice
of that.
"""
from __future__ import annotations

from pathlib import Path

from AccessDatasets import list_names, locate


def resolve_wsi_paths(dataset=None, wsi=None, val_only: bool = False) -> list:
    """`[{'dataset': id_or_None, 'wsi_name': name, 'path': str}, ...]`.

    Exactly one of `dataset`/`wsi` is the normal case:

        dataset=['ki67_pure', 'bracs/test']   every WSI AccessDatasets knows
                                              in those datasets (val_only=True
                                              narrows to the recorded val
                                              split -- needs cli/train.py to
                                              have already run for that
                                              dataset; see `val_split_names`)
        wsi=['/path/to/one.svs', ...]         explicit files, the escape
                                              hatch for a slide that is not
                                              (or not yet) in the registry --
                                              'dataset' and 'wsi_name' come
                                              back None/the file stem, since
                                              AccessDatasets was never asked

    Both given at once is both applied: the datasets' own WSIs plus the
    explicit paths, concatenated -- a caller wanting only one kind passes
    only that one kwarg. Neither given is a caller error, not silently
    "nothing": raises rather than returning an empty list a batch tool could
    mistake for "ran clean, found nothing wrong".
    """
    if not dataset and not wsi:
        raise ValueError('resolve_wsi_paths needs dataset= and/or wsi= -- '
                         'neither was given')

    entries = []
    for dataset_id in (dataset or []):
        names = (val_split_names(dataset_id) if val_only
                else list_names(dataset=dataset_id))
        for name in names:
            entry = locate(name, dataset=dataset_id)
            entries.append(dict(dataset=dataset_id, wsi_name=name,
                                path=entry.path))

    for path in (wsi or []):
        entries.append(dict(dataset=None,
                            wsi_name=Path(path).stem, path=path))
    return entries


def val_split_names(dataset_id: str) -> list:
    """The recorded val half of `dataset_id`'s split -- read, never
    re-derived: a split recomputed here could disagree with the one a
    checkpoint was actually selected against. Needs `cli/train.py` to have
    already run for this dataset (it is what writes `wsi_split.csv`).
    Moved here from `diag_wsi_scale.py` -- every `--val-only` caller in this
    directory wants the identical read, not a per-tool copy.
    """
    import csv
    import _paths
    path = (Path(_paths.RESULT_DIR) / 'cache' / 'mpp_routing_head'
            / dataset_id.replace('/', '_') / 'wsi_split.csv')
    if not path.exists():
        raise FileNotFoundError(
            f'{path} does not exist -- run cli/train.py first for this '
            f'dataset, or drop --val-only to scan every WSI instead')
    with open(path, newline='') as fh:
        return [r['wsi_name'] for r in csv.DictReader(fh) if r['split'] == 'val']
