"""make_list: the real-photo list `gigapath-winembed_found50_missed50` that `train.py --real-val <list>` scores every epoch (observe only).
Run once, by `RerankHead.sh` (REAL_VAL=<list> and no `real_val/<list>.tsv` yet): `python -m locascope.training.RerankHead.real_val.make_list <list>`.

Spec: .claude/changespec/coae/spec.md, section "winembed 訓練加真實照片觀察驗證".

Range: the photos of SLIDES. "Found" = the Demo ncc file's `ncc` >= FOUND.
  group `found`: the winembed gigapath arm found it; its stage 3 H is the photo's place.
  group `missed`: the winembed gigapath arm did not find it and at least one MISSED_ARMS arm did; the H is the one of the arm with the highest
           ncc among those that found it (ties in the order of MISSED_ARMS); when that arm's H is not sound the next one that found it is taken.
A photo is a candidate when its H is sound (stage 3 found a place, the row has an H, det > 0, the H maps the photo's centre onto the row's own
centre within `RealPhotos.CENTRE_TOL_PX`: the `RealPhoto` of the training, `centre_error()`, the one definition). PER_GROUP photos per group are
drawn from the candidates by `quota` (even over the slides, a slide with fewer takes all, the rest is spread in name order) and `drawn` (crc32
order, no random generator).
Reads: Demo ncc files, RealTest cache stage 3 tables, the photos' file headers. Writes `real_val/<list>.tsv` and `real_val/<list>.counts.txt`."""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from locascope.core.slides.AccessDatasets import locate
from locascope.training.RerankHead.RealPhotos import CENTRE_TOL_PX, DATASET, LISTS, RealPhoto, list_path

SEED: int = 0
PER_GROUP: int = 50
FOUND: float = 0.6
PHOTO_SUFFIXES: Tuple[str, ...] = ('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')
NCC_DIR: str = '/work/u26130998/DemoLocaScope/cache/realtest/RealTest'
REALTEST_ROOT: str = '/work/u26130998/result/cache/RealTest'
STAGE1: str = 'prototype-uni2-attn-set-transformer-attn-score-cross-on'
SLIDES: Tuple[str, ...] = ('S1104043,G7E,110207', 'S1128162,G7E,110727', 'S1140701,G7E,111018', 'S1151088,G7E,111220')
FOUND_ARM: Tuple[str, str] = ('winembed-gigapath-cls-avg', 'e34718e8c865')          # (stage 2 directory name, ncc file id)
MISSED_ARMS: Tuple[Tuple[str, str], ...] = (('slidewin-gigapath-cls-mean', '7d9f2ef6d2e4'), ('slidewin-uni2-cls-avg-mean', 'b4c5f1c40a77'),
                                            ('slidewin-GigaCoAE-RN3-fm-head-mean', '3c2eeb9ad90a'),
                                            ('slidewin-GigaCoAE-RP3-fm-head-mean', '695c6ff1c16c'),
                                            ('phasetiles-raw-localphase', '3efb812a3853'))    # the tie-break order
GROUPS: Tuple[str, ...] = ('found', 'missed')
H_KEYS: Tuple[str, ...] = tuple(f'h{i}{j}' for i in range(3) for j in range(3))
GEOM_KEYS: Tuple[str, ...] = ('crop_x0', 'crop_y0', 'center_x0', 'center_y0')
COLUMNS: Tuple[str, ...] = (('slide', 'index', 'group', 'arm', 'ncc', 'level', 'ds') + H_KEYS + GEOM_KEYS
                            + ('height', 'width', 'path'))
NO_STAGE3_ROW: str = 'no stage 3 row'
NO_PHOTO_FILE: str = 'no photo file'
RETRIEVAL_ONLY: str = 'retrieval only (no stage 3 place)'
NO_H: str = 'no homography (h00 empty)'
BAD_DET: str = 'determinant <= 0'
OFF_CENTRE: str = 'homography off its centre'
Row = Dict[str, Any]                            # one merged answer / output row of a photo


@dataclass(frozen=True)
class Pick:
    """One photo with a sound H and where the H came from."""
    photo: RealPhoto
    arm: str                                    # the stage 2 arm whose stage 3 gave the H
    ncc: float                                  # that arm's ncc


def h_of(row: Row) -> Tuple[float, ...]:
    return tuple(float(row[k]) for k in H_KEYS)


def ncc_of(arm_id: str) -> Dict[Tuple[str, int], float]:
    """`{(slide, index): ncc}` of an arm's ncc file, the photos with a value only (the slide column is quoted: it holds commas)."""
    df: pd.DataFrame = pd.read_csv(os.path.join(NCC_DIR, f'ncc_{arm_id}.csv'))
    df = df[df['ncc'].notna()]
    return {(str(s), int(i)): float(v) for s, i, v in zip(df['slide'], df['index'], df['ncc'])}


def stage3_rows(slide: str, arm: str) -> Dict[int, Row]:
    """`{index: row}`: the arm's stage 3 answer and output tables of a slide merged on (index, rank). Exactly one stage 3 directory and one
    answer and one output file, else raise."""
    pattern: str = f'{REALTEST_ROOT}/slide={slide}/seg=*/region=*/photos=*/stage1={STAGE1}-*/stage2={arm}-*/stage3'
    dirs: List[str] = [d for d in sorted(glob.glob(pattern)) if re.fullmatch(re.escape(arm) + r'-[0-9a-f]+', os.path.basename(os.path.dirname(d))[len('stage2='):])]
    if len(dirs) != 1:
        raise RuntimeError(f'{slide} {arm}: {len(dirs)} stage 3 directories for {pattern}')
    found: Dict[str, List[str]] = {kind: sorted(glob.glob(os.path.join(dirs[0], f'{kind}_*.csv'))) for kind in ('answer', 'output')}
    for kind, files in found.items():
        if len(files) != 1:
            raise RuntimeError(f'{dirs[0]}: {len(files)} {kind}_*.csv files')
    answer: pd.DataFrame = pd.read_csv(found['answer'][0])
    output: pd.DataFrame = pd.read_csv(found['output'][0])
    merged: pd.DataFrame = answer.merge(output.drop(columns=[c for c in ('x0', 'y0', 'confidence') if c in output.columns]), on=['index', 'rank'], how='inner')
    rows: Dict[int, Row] = {int(r['index']): r for r in merged.to_dict('records')}
    if len(rows) != len(merged):
        raise RuntimeError(f'{dirs[0]}: an index twice after the merge')
    return rows


def photo_files(slide: str) -> Dict[int, str]:
    """`{index: file}` of a slide's photos: the index is the trailing digits of the file stem (the rule of the Demo's `photo_index`)."""
    folder: str = locate(slide, dataset=DATASET).related['photos']
    found: Dict[int, str] = {}
    for name in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(name)
        m: Optional['re.Match[str]'] = re.search(r'(\d+)$', stem)
        if m and ext.lower() in PHOTO_SUFFIXES:
            found[int(m.group(1))] = os.path.join(folder, name)
    return found


def pick_of(slide: str, index: int, group: str, arm: str, ncc: float, rows: Dict[int, Row],
            files: Dict[int, str]) -> Tuple[Optional[Pick], str]:
    """`(pick, '')`, or `(None, reason)` when the photo's row of this arm is not a sound H or the photo has no file. Sound = stage 3 found a
    place, the row has an H, the determinant of its 2 x 2 block is > 0 (as `read_list` asks) and `RealPhoto.centre_error()` <= CENTRE_TOL_PX."""
    if index not in rows:
        return None, NO_STAGE3_ROW
    if index not in files:
        return None, NO_PHOTO_FILE
    row: Row = rows[index]
    if int(row['retrieval_only']) != 0:
        return None, RETRIEVAL_ONLY
    if pd.isna(row['h00']):
        return None, NO_H
    width, height = Image.open(files[index]).size
    photo: RealPhoto = RealPhoto(slide=slide, index=index, group=group, path=files[index], level=int(row['level']), ds=float(row['ds']),
                                 h=h_of(row), crop=(float(row['crop_x0']), float(row['crop_y0'])),
                                 centre=(float(row['center_x0']), float(row['center_y0'])), size=(int(height), int(width)))
    if float(np.linalg.det(photo.matrix()[:2, :2])) <= 0.0:
        return None, BAD_DET
    if photo.centre_error() > CENTRE_TOL_PX:
        return None, OFF_CENTRE
    return Pick(photo=photo, arm=arm, ncc=ncc), ''


@dataclass
class SlideCount:
    """What the candidate search of one (slide, group) saw."""
    found: int = 0                              # photos the group's condition holds for (before the H is looked at)
    fallback: int = 0                           # missed: photos whose first arm's H was not sound but a later arm's was
    unsound: Dict[str, int] = field(default_factory=dict)       # reason (of the first arm tried) -> photos that are no candidate for it


def candidates(slide: str, ncc_found: Dict[Tuple[str, int], float],
               ncc_missed: Dict[str, Dict[Tuple[str, int], float]]) -> Tuple[Dict[str, List[Pick]], Dict[str, SlideCount]]:
    """The candidates of a slide by group, and the counts. found: ncc of FOUND_ARM >= FOUND -> its H. missed: that ncc < FOUND or missing, and some
    MISSED_ARMS arm >= FOUND -> the arms that found it by (-ncc, order in MISSED_ARMS); the first whose `pick_of` is a candidate."""
    files: Dict[int, str] = photo_files(slide)
    rows_found: Dict[int, Row] = stage3_rows(slide, FOUND_ARM[0])
    rows_missed: Dict[str, Dict[int, Row]] = {arm: stage3_rows(slide, arm) for arm, _ in MISSED_ARMS}
    out: Dict[str, List[Pick]] = {g: [] for g in GROUPS}
    counts: Dict[str, SlideCount] = {g: SlideCount() for g in GROUPS}
    indices: List[int] = sorted(set(files) | {i for (s, i) in ncc_found if s == slide} | {i for d in ncc_missed.values() for (s, i) in d if s == slide})
    for index in indices:
        f_ncc: float = ncc_found.get((slide, index), float('nan'))
        if f_ncc >= FOUND:
            counts['found'].found += 1
            pick, reason = pick_of(slide, index, 'found', FOUND_ARM[0], f_ncc, rows_found, files)
            if pick is not None:
                out['found'].append(pick)
            else:
                counts['found'].unsound[reason] = counts['found'].unsound.get(reason, 0) + 1
            continue
        found_by: List[Tuple[float, int, str]] = [(-d[(slide, index)], order, arm) for order, (arm, _) in enumerate(MISSED_ARMS)
                                                  for d in (ncc_missed[arm],) if d.get((slide, index), float('nan')) >= FOUND]
        if not found_by:
            continue
        counts['missed'].found += 1
        first_reason: str = ''
        taken: Optional[Pick] = None
        for k, (neg, _, arm) in enumerate(sorted(found_by)):
            pick, reason = pick_of(slide, index, 'missed', arm, -neg, rows_missed[arm], files)
            if k == 0:
                first_reason = reason
            if pick is not None:
                taken = pick
                counts['missed'].fallback += int(k > 0)
                break
        if taken is not None:
            out['missed'].append(taken)
        else:
            counts['missed'].unsound[first_reason] = counts['missed'].unsound.get(first_reason, 0) + 1
    return out, counts


def quota(n_candidates: Dict[str, int], total: int) -> Dict[str, int]:
    """Split `total` over the slides that have candidates: even (the remainder one each in name order); a slide with no more candidates than
    its share takes all of them and the rest is split again over the others, until no slide is capped. Fewer candidates than `total` in all:
    every candidate."""
    take: Dict[str, int] = {s: 0 for s in n_candidates}
    pending: List[str] = sorted(s for s, n in n_candidates.items() if n > 0)
    remaining: int = total
    while pending:
        base, extra = divmod(remaining, len(pending))
        share: Dict[str, int] = {s: base + int(k < extra) for k, s in enumerate(pending)}
        capped: List[str] = [s for s in pending if n_candidates[s] <= share[s]]
        if not capped:
            take.update(share)
            break
        for s in capped:
            take[s] = n_candidates[s]
            remaining -= n_candidates[s]
        pending = [s for s in pending if s not in capped]
    return take


def crc_key(p: Pick) -> Tuple[int, int]:
    return zlib.crc32(f'{SEED}|{p.photo.slide}|{p.photo.index}'.encode()), p.photo.index


def drawn(cands: List[Pick], k: int) -> List[Pick]:
    """The first k by crc32 of `SEED|slide|index` (then index): the same on any machine."""
    return sorted(cands, key=crc_key)[:k]


def tsv_line(p: Pick) -> str:
    q: RealPhoto = p.photo
    cells: List[str] = [q.slide, str(q.index), q.group, p.arm, repr(p.ncc), str(q.level), repr(q.ds)]
    cells += [repr(v) for v in q.h] + [repr(v) for v in q.crop] + [repr(v) for v in q.centre] + [str(q.size[0]), str(q.size[1]), q.path]
    return '\t'.join(cells)


def main() -> None:
    ap: argparse.ArgumentParser = argparse.ArgumentParser()
    ap.add_argument('name', help='the list to make: `real_val/<name>.tsv` and `real_val/<name>.counts.txt`')
    args: argparse.Namespace = ap.parse_args()
    os.makedirs(LISTS, exist_ok=True)
    ncc_found: Dict[Tuple[str, int], float] = ncc_of(FOUND_ARM[1])
    ncc_missed: Dict[str, Dict[Tuple[str, int], float]] = {arm: ncc_of(i) for arm, i in MISSED_ARMS}
    cands: Dict[str, Dict[str, List[Pick]]] = {}
    counts: Dict[str, Dict[str, SlideCount]] = {}
    for slide in SLIDES:
        cands[slide], counts[slide] = candidates(slide, ncc_found, ncc_missed)
        print(f'{slide}: ' + ', '.join(f'{g} {len(cands[slide][g])}' for g in GROUPS), flush=True)
    picked: List[Pick] = []
    lines: List[str] = [f'make_list: list {args.name}, SEED {SEED}, PER_GROUP {PER_GROUP}, FOUND {FOUND}, CENTRE_TOL_PX {CENTRE_TOL_PX}',
                        f'slides: {", ".join(SLIDES)}', f'found arm: {FOUND_ARM[0]}',
                        f'missed arms (tie-break order): {", ".join(a for a, _ in MISSED_ARMS)}', '']
    lines.append('| group | slide | ncc found | candidates | quota | drawn | missed: later arm used | not a candidate (reason: photos) |')
    lines.append('|---|---|---|---|---|---|---|---|')
    totals: Dict[str, int] = {g: 0 for g in GROUPS}
    for g in GROUPS:
        take: Dict[str, int] = quota({s: len(cands[s][g]) for s in SLIDES}, PER_GROUP)
        for slide in SLIDES:
            chosen: List[Pick] = drawn(cands[slide][g], take[slide])
            picked += chosen
            totals[g] += len(chosen)
            c: SlideCount = counts[slide][g]
            why: str = '; '.join(f'{r}: {n}' for r, n in sorted(c.unsound.items())) or '-'
            lines.append(f'| {g} | {slide} | {c.found} | {len(cands[slide][g])} | {take[slide]} | {len(chosen)} | {c.fallback} | {why} |')
    lines.append('')
    lines.append(f'drawn: found {totals["found"]}, missed {totals["missed"]}, total {len(picked)}')
    lines.append(f'{OFF_CENTRE}: ' + str(sum(c.unsound.get(OFF_CENTRE, 0) for s in SLIDES for c in counts[s].values()))
                 + '  (not 0 = report, do not widen CENTRE_TOL_PX)')
    picked.sort(key=lambda p: (p.photo.slide, p.photo.group, p.photo.index))
    with open(os.path.join(LISTS, f'{args.name}.counts.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    tmp: str = list_path(args.name) + '.tmp'                    # the list appears whole or not at all: its presence means "made"
    with open(tmp, 'w') as f:
        f.write('\t'.join(COLUMNS) + '\n')
        for p in picked:
            f.write(tsv_line(p) + '\n')
    os.replace(tmp, list_path(args.name))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    code: int = 0
    try:
        main()
    except Exception:                                              # noqa: BLE001
        import traceback
        traceback.print_exc()
        code = 1
    print('ALL DONE' if code == 0 else f'FAILED (exit {code})', flush=True)
    sys.exit(code)
