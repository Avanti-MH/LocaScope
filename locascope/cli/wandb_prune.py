#!/usr/bin/env python3
'''Keep the latest run of every arm in some wandb projects, and list -- or, with `--delete`, delete -- the rest.

    python -m locascope.cli.wandb_prune --projects prototypical-routing-head classifying-routing-head
    python -m locascope.cli.wandb_prune --projects prototypical-routing-head --delete

An ARM is the run's NAME inside its project (`gigapath-tilegt-corrnet-winembed`, `uni2-ki67_pure-...`): the trainers
name a run after what it trains, so every run of one name is a run of one arm. Of each arm the newest `--keep` runs
(by creation time) stay; a run that is still running always stays. The default is a DRY RUN: it prints every run
with KEEP or DELETE and the totals, and changes nothing. `--delete` deletes the DELETE ones on the wandb server (no
undo; their artifacts are kept unless `--artifacts`) after asking you to type the number of runs.

It talks to wandb with the credentials of ~/.netrc, over the network, and reads and deletes only what it prints. The
local `wandb/` directory of the checkout is not touched.
'''
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class RunRow:
    """One wandb run as the plan sees it."""
    project: str
    id: str
    name: str
    state: str
    created: str          # ISO time: the strings sort as times do
    url: str = ''
    keep: bool = True
    smoke: bool = False   # a smoke / test run (is_smoke)
    why: str = ''         # why it goes, for the report
    steps: int = -1       # how far the run got (wandb's last logged step); -1: not known
    arm: str = ''         # what the run is an arm of (set by plan)


def is_smoke(config: Dict[str, Any], name: str = '') -> bool:
    """A smoke or test run: its name has `Smoke` in it (or starts with `Test`), or the Slurm job name in its config
    starts with `Smoke` or `Test` (the classifying and prototypical trainers record it), or it was stopped after a
    few visits (`max_visits` > 0, the rerank trainers), or it was a dry run."""
    job: str = str(config.get('slurm_job_name') or '')
    try:
        visits: int = int(config.get('max_visits') or 0)
    except (TypeError, ValueError):
        visits = 0
    return ('smoke' in name.lower() or name.startswith('Test') or job.startswith(('Smoke', 'Test')) or visits > 0
            or bool(config.get('dry_run')))


def arm_of(name: str, run_id: str, strip: bool) -> str:
    """The arm a run belongs to: its name, with `strip` the name without its first `-` part. The prototypical runs
    were named `<job id>_<spec>-uni2-mean-cosine_tau-none`, later `PrototypicalRoutingHead-uni2-mean-cosine_tau-none`:
    the first part is the generation of the naming and the rest is the arm. A run with no name is an arm of its own."""
    if not name:
        return run_id
    return name.split('-', 1)[1] if strip and '-' in name else name


def plan(rows: Sequence[RunRow], keep: int = 1, drop: Sequence[str] = (), formal_only: bool = False,
         prefer: str = 'newest', strip: Sequence[str] = (), finished_only: bool = False, since: str = '',
         since_projects: Sequence[str] = ()) -> List[RunRow]:
    """`rows` with `keep` set. Every run of a project in `drop` goes; with `formal_only` a smoke run goes too, and
    with `finished_only` one that did not finish (crashed, failed, killed); of each `(project, arm)` among the rest
    `keep` runs stay -- the newest, or with `prefer='longest'` the ones that got farthest (a run resumed or trained
    on can be the short one: its earlier part is another run). A run still running always stays. The arm is the
    name (`arm_of`, for the projects in `strip` without its first part)."""
    groups: Dict[Tuple[str, str], List[RunRow]] = {}
    for r in rows:
        r.why = ''
        r.arm = arm_of(r.name, r.id, r.project in strip)
        if r.project in drop:
            r.keep, r.why = r.state == 'running', 'project dropped'
        elif since and r.project in since_projects and r.created < since:
            r.keep, r.why = r.state == 'running', f'created before {since}'
        elif formal_only and r.smoke:
            r.keep, r.why = r.state == 'running', 'smoke / test'
        elif finished_only and r.state not in ('finished', 'running'):
            r.keep, r.why = False, f'not finished ({r.state})'
        else:
            groups.setdefault((r.project, r.arm), []).append(r)
    for runs in groups.values():
        runs.sort(key=(lambda r: (r.steps, r.created)) if prefer == 'longest' else (lambda r: r.created), reverse=True)
        for i, r in enumerate(runs):
            r.keep = i < keep or r.state == 'running'
            r.why = '' if r.keep else 'older run of its arm'
    return list(rows)


def fetch(entity: str, project: str, api: Any) -> List[RunRow]:
    return [RunRow(project=project, id=r.id, name=r.name or '', state=r.state, created=str(r.created_at),
                   url=r.url, smoke=is_smoke(dict(r.config or {}), r.name or ''), steps=last_step(r))
            for r in api.runs(f'{entity}/{project}', per_page=200)]


def last_step(run: Any) -> int:
    """The last step a run logged (-1 when wandb does not say)."""
    try:
        return int(run.summary.get('_step', -1))
    except (TypeError, ValueError, AttributeError):
        return -1


def report(rows: Sequence[RunRow]) -> None:
    for project in dict.fromkeys(r.project for r in rows):
        mine: List[RunRow] = [r for r in rows if r.project == project]
        print(f'\n== {project}: {len(mine)} runs, {sum(r.keep for r in mine)} kept, '
              f'{sum(not r.keep for r in mine)} to delete')
        for key in dict.fromkeys(r.arm or r.name or r.id for r in mine):
            arm: List[RunRow] = sorted((x for x in mine if (x.arm or x.name or x.id) == key), key=lambda x: x.created,
                                       reverse=True)
            for r in arm:
                print(f'  {"KEEP  " if r.keep else "DELETE"}  {r.created[:19]}  {r.id:10s}  {r.state:9s}  step {r.steps:6d}  '
                      f'{r.name or r.id}'
                      + (f'   [{r.why}]' if r.why else '') + ('   (smoke)' if r.smoke and r.keep else ''))
            kept: List[RunRow] = [r for r in arm if r.keep]
            gone: List[RunRow] = [r for r in arm if not r.keep and not r.smoke and r.why == 'older run of its arm']
            if kept and gone and max(r.steps for r in gone) > max(r.steps for r in kept):
                print(f'  !! the run that stays got to step {max(r.steps for r in kept)}, a run that goes got to '
                      f'{max(r.steps for r in gone)}: a resumed or continued run is the short one -- look, or '
                      f'--prefer longest')
            if len(kept) > 1:
                print(f'  !! {len(kept)} runs stay for this arm (running ones, or --keep {len(kept)})')


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--projects', nargs='*', default=[], help='the wandb projects to prune (none: list the projects '
                                                              'of the entity and stop)')
    ap.add_argument('--drop-projects', nargs='*', default=[],
                    help='projects whose runs ALL go (listed in the plan like the others)')
    ap.add_argument('--formal-only', action='store_true',
                    help='a smoke or test run goes too (is_smoke), whatever its arm; the newest of each arm is then '
                         'of the formal runs')
    ap.add_argument('--entity', default='', help='the wandb entity (default: yours)')
    ap.add_argument('--strip-prefix', nargs='*', default=[],
                    help='projects whose arm is the run name WITHOUT its first `-` part (the naming generations of the '
                         'prototypical runs: <job id>_<spec>-uni2-... and PrototypicalRoutingHead-uni2-...)')
    ap.add_argument('--since', default='',
                    help='for --since-projects: a run created before this time (UTC, "2026-10-20" or '
                         '"2026-10-20T12:00:00") goes, whatever its arm: only the latest round of training stays')
    ap.add_argument('--since-projects', nargs='*', default=[], help='the projects --since applies to')
    ap.add_argument('--finished-only', action='store_true',
                    help='a run that did not finish (crashed, failed, killed) goes too')
    ap.add_argument('--keep', type=int, default=1, help='runs of each arm that stay (1: the arm is unique)')
    ap.add_argument('--prefer', choices=('newest', 'longest'), default='newest',
                    help='which run of an arm stays: the newest, or the one that logged the most steps')
    ap.add_argument('--delete', action='store_true', help='delete the DELETE runs (default: only print the plan)')
    ap.add_argument('--artifacts', action='store_true', help='with --delete, delete the runs\' artifacts too')
    args = ap.parse_args(argv)
    import wandb                                                                      # noqa: PLC0415
    api = wandb.Api()
    entity: str = args.entity or api.default_entity
    if not args.projects and not args.drop_projects:
        print(f'projects of {entity}:')
        for p in api.projects(entity):
            print(f'  {p.name}')
        return 0
    rows: List[RunRow] = []
    for project in [*args.projects, *args.drop_projects]:
        found = fetch(entity, project, api)
        if not found:
            print(f'{project}: no runs (is the name right? `--projects` with no names lists them)')
        rows += found
    plan(rows, args.keep, args.drop_projects, args.formal_only, args.prefer, args.strip_prefix, args.finished_only,
         args.since, args.since_projects)
    report(rows)
    doomed: List[RunRow] = [r for r in rows if not r.keep]
    print(f'\n{len(doomed)} of {len(rows)} runs to delete, {len(rows) - len(doomed)} kept')
    if not args.delete or not doomed:
        print('dry run: nothing deleted' if doomed else 'nothing to delete')
        return 0
    if input(f'type {len(doomed)} to delete these {len(doomed)} runs from wandb (no undo): ').strip() != str(len(doomed)):
        print('not confirmed: nothing deleted')
        return 1
    for r in doomed:
        api.run(f'{entity}/{r.project}/{r.id}').delete(delete_artifacts=args.artifacts)
        print(f'deleted {r.project}/{r.id}  {r.name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
