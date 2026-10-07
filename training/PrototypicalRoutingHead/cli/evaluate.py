#!/usr/bin/env python3
'''Scores saved `cli/train.py` checkpoints on the TEST half of `wsi_split.csv`,
two ways, into two pairs of CSVs:

    original   test_scores_per_combo.csv / _per_rung.csv -- random full 6-way
               episodes, accuracy pooled over their query tiles. Unchanged, so
               its numbers stay comparable with every earlier run and, tile by
               tile, with MppRoutingHead's.
    K x K      test_scores_kxk_per_combo.csv / _per_rung.csv -- the same
               definition training's val uses (`train.kxk_report`): every
               held-out combination, K support x K query batches, rung ->
               combo -> dataset -> total means.

    python training/PrototypicalRoutingHead/cli/evaluate.py

Separate from `train.py` on purpose, same reasoning `MppRoutingHead/cli/
evaluate.py`'s own module docstring gives: the test split is every WSI
`train.py` did NOT hold out for val, and it is meant to be looked at once a
question is settled, not once per training run.

NOT A SECOND SCORING IMPLEMENTATION: this calls `train.py`'s OWN
`episode_forward`/`val_episode_detail`/`combo_report` unchanged, on a fresh
deterministic draw of TEST episodes -- the exact per-combo, per-rung,
native/resampled discipline `combo_report`'s own docstring explains
("blending combos together answers no single question cleanly") applies
here exactly as it does to held-out val; writing a second version of that
logic against the test split would be the same mistake in a new file.

DEEPER, FEWER SLIDES than val: `--n-wsi 5 --n-per-rung
50` (val: 10 slides, `--val-n-per-rung 20`) -- the same "shallow-and-wide
for cheap per-epoch selection, deep-and-narrow for the one-shot final number"
split `MppRoutingHead/cli/evaluate.py`'s own defaults use.

ONLY the full 6-way combo (`tuple(RUNGS)`, `Episodes.HELD_OUT_COMBOS`'s own
third entry) -- not the 3-way/4-way probes. Those are compositional-
generalisation PROBES already tracked every epoch in `val_scores_per_combo.
csv`; test scoring exists to answer ONE question (does the DEPLOYMENT task
generalise), and mixing the probes back in here would just re-measure, at
greater depth, a question val already answers.

`--tag` (`nargs='+'`, default ALL THREE: `6rung`/`native`/`best`) -- unlike
`MppRoutingHead/cli/evaluate.py`'s `best`/`last`/`best_unweighted`, this
project's three checkpoint-selection criteria (`save_tagged`'s own
docstring) do not always agree on which epoch wins, so scoring only one by
default would silently answer "which epoch under ONE of three criteria" --
sweeping all three, with `tag` as its own output column, is what makes that
visible.

A checkpoint's own filename already carries its full arm identity
(`_prototype_weight_filename`'s own doc), so the default (no `--weights`)
scores EVERY `*_<tag>.pt` under the run directory in one call -- a 9-arm
sweep x 3 tags = 27 files, all landing in ONE `test_scores_per_combo.csv`,
comparable the same way `val_scores_per_combo.csv` already lets several arms
sit side by side. Re-running against a newly-added checkpoint MERGES by
`(weights, val_dataset)` -- same reasoning `train.py`'s own `_merge_val_
scores` gives for `--merge`, applied unconditionally here since this file
has no reason to ever want the older rows discarded.
'''
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..', 'utilities'))
from _paths import setup_import_paths, job_result_dir              # noqa: E402
setup_import_paths()

import torch                                                        # noqa: E402

from training.MppRoutingHead.Datasets import (                      # noqa: E402
    add_cache_args, build_manifest, data_record, open_caches, RUNGS,
    RenderConfig,
    CameraBank)
from AccessDatasets import list_names                                # noqa: E402
from training.PrototypicalRoutingHead.Episodes import (              # noqa: E402
    HELD_OUT_COMBOS, group_by_wsi_name, max_feasible_k, pool_by_rung,
    render_episode, sample_val_episode)
from Checkpoints import build_prototype_from_checkpoint              # noqa: E402
# Reused, not reimplemented -- see this module's own docstring.
from train import (episode_forward, val_episode_detail, combo_report,  # noqa: E402
                   kxk_report, score_kxk, _auto_int)


def test_manifests(args, caches, out_dir: Path):
    '''`({dataset_id: manifest_by_wsi}, {dataset_id: mixed pool by rung})` --
    the first for the original table, the second for K x K. The test-half analogue of `cli/
    train.py`'s own `val_manifests`, drawn through the same per-slide sampler
    cache.

    THE FIRST `args.n_wsi` of the recorded test order, not a fresh draw --
    `wsi_split.csv` already shuffled once, so a prefix is a sample, and the
    SAME sample every time without a second seed to keep in step with the
    first (`MppRoutingHead/cli/evaluate.py`'s own `test_rows` reasoning).
    The split is READ (`<dataset>#test`; it refuses when the split is missing): one derived
    here could disagree with the one `train.py` selected checkpoints on.
    '''
    out, pools, sizes = {}, {}, {}
    for dataset_id in args.eval_datasets:
        names = list_names(dataset=f'{dataset_id}#test',
                           split_job=caches.split_job)[:args.n_wsi]
        part = build_manifest(
            dataset_id, masks=caches.masks, sampler_root=caches.sampler_root,
            report_dir=(out_dir / 'sampler_reports'
                        / f'{dataset_id.replace("/", "_")}_test'),
            tile_size=args.tile, n_per_rung=args.n_per_rung, seed=args.seed,
            wsi_names=names)
        out[dataset_id] = group_by_wsi_name(part)
        pools[dataset_id] = pool_by_rung(part)
        sizes[dataset_id] = (len(names), len(part), names)
    return out, pools, sizes


def find_weights(args, out_dir: Path) -> List[Path]:
    if args.weights:
        return [Path(w) for w in args.weights]
    found = []
    for tag in args.tag:
        found += sorted((out_dir / 'weights').glob(f'*_{tag}.pt'))
    if not found:
        raise FileNotFoundError(
            f'no *_{{{"|".join(args.tag)}}}.pt under {out_dir / "weights"} -- '
            f'name one with --weights, or point --out at the run that wrote them')
    return found


def _merge_rows(path: Path, rows: List[Dict], key_fields: Sequence[str]) -> List[Dict]:
    '''`rows`' own `key_fields` key REPLACES matching rows already in
    `path`, every other row already on disk is kept -- `train.py`'s own
    `_merge_val_scores`, keyed on `weights`+`val_dataset` here instead of
    `_IDENTITY_FIELDS`: two different `--tag` checkpoints of the SAME arm
    are two different files, so `weights` (the checkpoint filename) is
    the key that actually distinguishes them; `_IDENTITY_FIELDS` alone
    would not.
    '''
    if not rows or not path.exists():
        return rows
    new_keys = {tuple(str(r[k]) for k in key_fields) for r in rows}
    with open(path, newline='') as fh:
        kept = [r for r in csv.DictReader(fh)
               if tuple(r[k] for k in key_fields) not in new_keys]
    return kept + rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', nargs='+', default=None,
                    help='checkpoint paths; default is every *_<tag>.pt '
                         'under the run directory, for EVERY --tag')
    ap.add_argument('--tag', nargs='+', choices=('6rung', 'native', 'best'),
                    default=['6rung', 'native', 'best'],
                    help="which of save_tagged's three selection criteria "
                         "to score, when --weights is not given (default: "
                         "all three -- see this module's own docstring)")
    ap.add_argument('--eval-datasets', nargs='+',
                    default=['bracs/test', 'ki67_with_photo'])
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--n-wsi', type=int, default=5,
                    help="slides taken from each dataset's test half")
    ap.add_argument('--n-per-rung', type=int, default=50,
                    help='TileSampler budget per WSI -- the pool test '
                         'episodes draw from, not the episode count')
    ap.add_argument('--n-support', type=int, default=5)
    ap.add_argument('--n-query', type=int, default=10)
    ap.add_argument('--p-same-wsi', type=float, default=0.5,
                    help='original table only: its per-episode WSI coin')
    ap.add_argument('--n-episodes', type=int, default=100,
                    help='original table: full 6-way test episodes drawn PER '
                         'dataset PER checkpoint')
    ap.add_argument('--kxk-k', type=_auto_int, default='auto',
                    help='K for the K x K table, or auto = the largest every '
                         'test dataset supplies for every held-out combination')
    ap.add_argument('--max-overlap', type=float, default=0.0,
                    help="K x K table: train.py's overlap rule")
    ap.add_argument('--feasibility-tries', type=int, default=5)
    ap.add_argument('--support-native', choices=('checkpoint', 'on', 'off'),
                    default='checkpoint',
                    help="how the SUPPORT side is rendered. 'checkpoint' "
                         "(default): as each checkpoint trained (its own "
                         "--support-native). 'on': routing-support-native for "
                         "every checkpoint -- rotation only, closer to the raw "
                         "WSI crops PrototypeEstMpp feeds at deployment (which "
                         "do not rotate). 'off': routing-query for every one. Off "
                         "the default, the CSVs are named test_scores[_kxk]_"
                         "support-<on|off>_* so they do not replace the default "
                         "run's")
    ap.add_argument('--encode-batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    # --seg: every checkpoint scored must have been trained under the same one
    add_cache_args(ap)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    # one process, no workers: torch gets every cpu of the job
    from CpuBudget import CpuBudget                                 # noqa: PLC0415
    print(f'  {CpuBudget.for_job(workers=0).apply().line()}', flush=True)

    device = torch.device(args.device)
    caches = open_caches(args, 'PrototypicalRoutingHead', device)
    out_dir = Path(args.out or job_result_dir('PrototypicalRoutingHead'))
    weights = find_weights(args, out_dir)
    test_by_dataset, test_pools, test_sizes = test_manifests(args, caches, out_dir)
    caches.masks.close()

    kxk_max = min(max_feasible_k(
        lambda combo, _p=pool: [(_p, _p)], HELD_OUT_COMBOS, [lambda k: (k, k)],
        n_support=args.n_support, n_query=args.n_query,
        max_overlap=args.max_overlap, tries=args.feasibility_tries,
        seed=args.seed) for pool in test_pools.values())
    kxk_k = kxk_max if args.kxk_k == 'auto' else args.kxk_k
    print(f'\n======== evaluate.py -- TEST split ========', flush=True)
    where = ('given by --weights' if args.weights else
             f'{", ".join("*_" + t + ".pt" for t in args.tag)} in {out_dir / "weights"}')
    print(f'checkpoints   {len(weights)}  ({where})', flush=True)
    for i, (dataset_id, (n_wsi, n_pos, names)) in enumerate(test_sizes.items()):
        print(f'{"test slides" if i == 0 else "":<14s}{dataset_id:<17s} {n_wsi} WSIs  '
              f'{n_pos} positions  ({", ".join(names)})', flush=True)
    print(f'per checkpoint, two tables:', flush=True)
    print(f'  [A] random 6-way   {args.n_episodes} episodes of all six rungs per '
          f'dataset; accuracy pooled over their query tiles, per rung, and split by '
          f'how a tile was read (native level / resampled from a finer one) '
          f'-> test_scores_per_*.csv', flush=True)
    print(f'  [B] K x K          K = {kxk_k} ({args.kxk_k}; the largest the test '
          f'split can draw is {kxk_max}); every held-out combination, KxK '
          f'support x query pairs, rung -> combo -> dataset -> total means '
          f'-> test_scores_kxk_per_*.csv', flush=True)
    print(f'support       ' + (
        'as each checkpoint trained (--support-native recorded in it)'
        if args.support_native == 'checkpoint' else
        f'{args.support_native} for every checkpoint ('
        f'{"routing-support-native: rotation only" if args.support_native == "on" else "routing-query"})'),
        flush=True)
    print(f'retries       --feasibility-tries {args.feasibility_tries}: a draw '
          f'that cannot be drawn or rendered is drawn again, this many times',
          flush=True)
    if kxk_k < 1 or kxk_k > kxk_max:
        raise SystemExit(f'K x K table: K={kxk_k} cannot be drawn (max {kxk_max}). '
                         f'Lower --n-support/--n-query or raise --n-per-rung')
    render_cfg = RenderConfig(tile_size=args.tile)
    bank = CameraBank(render_cfg)
    six_way = (tuple(RUNGS),)   # ONLY the deployment combo -- see module docstring

    out_combo_rows: List[Dict] = []
    out_rung_rows: List[Dict] = []
    kxk_combo_rows: List[Dict] = []
    kxk_rung_rows: List[Dict] = []
    for i_ckpt, path in enumerate(weights, 1):
        (pooling, support_context, query_context, collapse, head,
        encoder, ckpt) = build_prototype_from_checkpoint(path, device)

        run_args = ckpt['args']
        tile = int(run_args.get('tile', args.tile))
        if tile != args.tile:
            raise ValueError(
                f'{path.name} was trained at tile_size={tile} and --tile is '
                f'{args.tile}: scoring it at a different tile size measures '
                f'a different question, so this is a refusal rather than a '
                f'warning')
        # Same refusal for the mask recipe: the test positions are drawn once,
        # under --seg, and a checkpoint trained under another recipe would be
        # scored on positions from a mask it never learned on.
        if run_args['seg'] != args.seg:
            raise ValueError(
                f'{path.name} was trained with seg={run_args["seg"]!r} '
                f'and --seg is {args.seg!r}')

        num_prefix = int(encoder.model_spec.num_prefix)
        extra = ckpt.get('extra', {})
        # Everything else the episodes are made of (Datasets.data_record).
        from ConfigIdentity import record_diff                    # noqa: PLC0415
        stale = record_diff(extra.get('data'), data_record(args.tile, args.seg))
        if stale:
            raise ValueError(
                f'{path.name} was trained on other data than the test episodes '
                f'would be made of: ' + '; '.join(stale) + '. Retrain it, or '
                f'score it with the code and environment it was trained under')
        # support_native: by default read back off THIS
        # checkpoint's own recorded args, same as tile above -- render_episode
        # reproduces whatever this specific arm trained under (routing-query or
        # routing-support-native on the support side); --weights all scores
        # several arms in one call, and each may have trained differently.
        # --support-native on/off overrides that for every
        # checkpoint, to score them all against one kind of support -- the
        # `support_native` column below still says how each one TRAINED,
        # `eval_support` how this run rendered.
        trained_native = bool(run_args.get('support_native', False))
        support_native = (trained_native if args.support_native == 'checkpoint'
                          else args.support_native == 'on')
        # support_context/query_context/collapse/routing_head are top-
        # level fields (Checkpoints.save_prototype_checkpoint's own
        # docstring) -- read directly, no extra fallback: every checkpoint
        # here was retrained under the top-level scheme.
        identity = dict(
            training_framework='metric_based', encoder=ckpt['encoder'],
            pooling=run_args.get('pooling', ''),
            support_context=ckpt['support_context_name'],
            query_context=ckpt['query_context_name'],
            collapse=ckpt['collapse_name'],
            routing_head=ckpt['routing_head_name'],
            cross_domain_dataset=extra.get('cross_domain_dataset', ''),
            # train_dataset/loss: the same fields as train.py's own
            # _IDENTITY_FIELDS -- read off ckpt['args'] since
            # neither one is in `extra` (they are plain CLI values, not a
            # registry name needing a lookup).
            train_dataset=run_args.get('train_dataset', ''),
            loss=run_args.get('loss', ''),
            support_native=trained_native,
            eval_support='native' if support_native else 'full',
            seg=run_args['seg'],
            episode_reuse=extra.get('episode_reuse', ''),
            reuse_k=extra.get('reuse_k', ''))
        tag = path.stem.rsplit('_', 1)[-1]
        cross = identity['cross_domain_dataset'] or 'off'
        print(f'\n-------- [{i_ckpt}/{len(weights)}] reuse '
              f'{identity["episode_reuse"] or "-"} | {identity["collapse"]} | '
              f'{identity["routing_head"]} | ctx {identity["support_context"]}/'
              f'{identity["query_context"]} | cross {cross} | tag {tag} --------',
              flush=True)
        print(f'{path.name}', flush=True)

        rng = random.Random(args.seed)
        detail: Dict[str, List[Dict]] = {d: [] for d in args.eval_datasets}
        n_redraws = 0
        for dataset_id in args.eval_datasets:
            manifest_by_wsi = test_by_dataset[dataset_id]
            n_done = 0
            while n_done < args.n_episodes:
                episode = sample_val_episode(
                    manifest_by_wsi, rng, p_same_wsi=args.p_same_wsi,
                    n_support=args.n_support, n_query=args.n_query,
                    combos=six_way)
                if episode is None:
                    n_redraws += 1
                    continue
                rendered = render_episode(episode, bank, render_cfg,
                                          deterministic=True,
                                          support_native=support_native)
                if rendered is None:
                    n_redraws += 1
                    continue
                with torch.no_grad():
                    logits, target, native = episode_forward(
                        rendered, encoder=encoder, num_prefix=num_prefix,
                        pooling=pooling, support_context=support_context,
                        query_context=query_context, collapse=collapse,
                        head=head, batch_size=args.encode_batch, device=device)
                detail[dataset_id] += val_episode_detail(
                    rendered, logits, target, native, dataset_id)
                n_done += 1
        print(f'[A] random 6-way   ({n_redraws} redraws: episodes that could '
              f'not be drawn or rendered, drawn again)', flush=True)

        c_rows, r_rows = combo_report(detail, int(ckpt['epoch']),
                                      weights=path.name, tag=tag, **identity)
        out_combo_rows += c_rows
        out_rung_rows += r_rows

        print(f'[B] K x K, K={kxk_k}', flush=True)
        for m in (pooling, support_context, query_context, collapse, head):
            m.eval()
        results = score_kxk(
            test_pools, HELD_OUT_COMBOS, k=kxk_k, n_support=args.n_support,
            n_query=args.n_query, max_overlap=args.max_overlap, seed=args.seed,
            tries=args.feasibility_tries, bank=bank, render_cfg=render_cfg,
            support_native=support_native, encoder=encoder, num_prefix=num_prefix,
            pooling=pooling, support_context=support_context,
            query_context=query_context, collapse=collapse, head=head,
            batch_size=args.encode_batch, device=device)
        c_rows, r_rows, _summary = kxk_report(
            results, int(ckpt['epoch']), k=kxk_k, scope='test',
            weights=path.name, tag=tag, **identity)
        kxk_combo_rows += c_rows
        kxk_rung_rows += r_rows

    seg = '' if args.support_native == 'checkpoint' else f'_support-{args.support_native}'
    rc = write_csvs(out_dir, out_combo_rows, out_rung_rows, prefix=f'test_scores{seg}')
    rc_kxk = write_csvs(out_dir, kxk_combo_rows, kxk_rung_rows,
                        prefix=f'test_scores_kxk{seg}')
    return rc or rc_kxk


def write_csvs(out_dir: Path, combo_rows: List[Dict], rung_rows: List[Dict],
               prefix: str = 'test_scores') -> int:
    if not combo_rows:
        print(f'\nno rows to write for {prefix} (every checkpoint failed before scoring)')
        return 1

    combo_path = out_dir / f'{prefix}_per_combo.csv'
    write_combo = _merge_rows(combo_path, combo_rows, ('weights', 'val_dataset'))
    with open(combo_path, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(
            k for r in write_combo for k in r)))
        wr.writeheader()
        wr.writerows(write_combo)
    print(f'\n{combo_path}  ({len(write_combo)} rows)', flush=True)

    rung_path = out_dir / f'{prefix}_per_rung.csv'
    write_rung = _merge_rows(rung_path, rung_rows, ('weights', 'val_dataset', 'rung'))
    with open(rung_path, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(
            k for r in write_rung for k in r)))
        wr.writeheader()
        wr.writerows(write_rung)
    print(f'{rung_path}  ({len(write_rung)} rows)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
