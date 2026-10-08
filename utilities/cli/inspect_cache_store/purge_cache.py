#!/usr/bin/env python3
"""Remove one variant of a cache entry, or a whole subtree, by its address.

    python utilities/cli/inspect_cache_store/purge_cache.py --job BenchLocaScope \\
        --level slide=BRACS_1228 seg=hest-caadecfe1e1afe67 --entry mask --id <id>
    python utilities/cli/inspect_cache_store/purge_cache.py --job MppRoutingHead \\
        --level slide=BRACS_1228 seg=hsv-0123456789abcdef
    ... --yes        actually delete; without it, only list

The address is the caller's, level by level (`Cache.Address` checks it is one
chain of `Cache.TREE`): nothing is searched for. With `--entry` and `--id` one
variant goes -- its record first, then the files its record lists
(`Entry.drop`), so a reader racing the removal sees a miss. Without them the
whole directory at the address goes: every entry and every level below it,
which is how a recipe is dropped (`--level slide=<s> seg=<seg_id>`) -- dropping
a mask's subtree drops every draw, render and stage result made from it.

Before deleting it prints what it would remove and how big it is.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # utilities/
import _paths                                                       # noqa: E402
_paths.setup_import_paths()

from Cache import Address                                           # noqa: E402


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def _gb(n: int) -> str:
    return f'{n / 1e9:.2f} GB'


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--job', required=True, help='whose cache tree')
    ap.add_argument('--level', nargs='*', default=[], metavar='KIND=VALUE',
                    help='the address, one level per argument')
    ap.add_argument('--entry', default=None, help='the entry kind at that address')
    ap.add_argument('--id', default=None, help='the variant of that entry')
    ap.add_argument('--whole-job', action='store_true',
                    help='with no --level: remove the job\'s whole tree')
    ap.add_argument('--yes', action='store_true', help='delete; default only list')
    args = ap.parse_args()

    levels = {}
    for item in args.level:
        kind, sep, value = item.partition('=')
        if not sep:
            ap.error(f'--level {item!r} is not KIND=VALUE')
        levels[kind] = value
    addr = Address(args.job, **levels)
    if (args.entry is None) != (args.id is None):
        ap.error('--entry and --id go together')
    if not levels and args.entry is None and not args.whole_job:
        ap.error('no --level: that is the whole job; say --whole-job')

    if args.entry is not None:
        entry = addr.entry(args.entry)
        stored = entry.stored(args.id)
        if stored is None:
            print(f'no variant {args.id} in {entry.dir}')
            return 1
        targets = [entry.record_path(args.id)] + entry.members(args.id)
        total = sum(_size(p) for p in targets if p.exists())
        for p in targets:
            print(f'  {p}')
        print(f'{len(targets)} item(s), {_gb(total)}')
        if args.yes:
            entry.drop(args.id)
            print('removed')
        return 0

    target = addr.dir if levels else addr.root
    if not target.exists():
        print(f'nothing at {target}')
        return 1
    entries = [p for p in target.rglob('record_*.json')]
    print(f'  {target}')
    print(f'{len(entries)} variant(s) below it, {_gb(_size(target))}')
    if args.yes:
        shutil.rmtree(target)
        print('removed')
    else:
        print('(list only: --yes deletes)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
