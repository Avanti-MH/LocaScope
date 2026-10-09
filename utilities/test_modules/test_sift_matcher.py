#!/usr/bin/env python3
"""The GPU ratio matcher of stage 3 (`knn_ratio_matches`) against OpenCV's
`BFMatcher.knnMatch` + Lowe ratio test: the same matches, the same distances.
Made-up descriptors shaped like SIFT's (128 floats, 0..255); the matcher runs on
the CPU here, the same code the GPU runs. On a CUDA machine it runs there too.

    python utilities/test_modules/test_sift_matcher.py
"""

from __future__ import annotations

import os
import sys
from typing import Callable, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
import _paths                                                    # noqa: E402
_paths.setup_import_paths()

import cv2                                                       # noqa: E402
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402

from stage3_localization.SIFT_RANSAC import knn_ratio_matches    # noqa: E402

RATIO: float = 0.75
_RESULTS: List[Tuple[str, bool]] = []


def check(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        _RESULTS.append((name, True))
        print(f'  ok    {name}')
    except Exception as exc:                                     # noqa: BLE001
        _RESULTS.append((name, False))
        print(f'  FAIL  {name}: {exc}')


def made_up(seed: int, n_query: int = 600, n_crop: int = 4000
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Crop descriptors, and query descriptors that are a third noisy copies of
    crop ones (matches the ratio test keeps), a third near-copies of two crop
    descriptors at once (the ratio test's borderline) and a third unrelated."""
    rng: np.random.Generator = np.random.default_rng(seed)
    crop: np.ndarray = rng.uniform(0, 255, (n_crop, 128)).astype(np.float32)
    own: np.ndarray = crop[rng.integers(0, n_crop, n_query // 3)]
    near: np.ndarray = (own + rng.normal(0, 6, own.shape)).astype(np.float32)
    a: np.ndarray = crop[rng.integers(0, n_crop, n_query // 3)]
    b: np.ndarray = crop[rng.integers(0, n_crop, n_query // 3)]
    mix: np.ndarray = (rng.uniform(0.4, 0.6, (len(a), 1)) * a
                       + (1 - rng.uniform(0.4, 0.6, (len(a), 1))) * b).astype(np.float32)
    lone: np.ndarray = rng.uniform(0, 255, (n_query - len(near) - len(mix), 128)
                                   ).astype(np.float32)
    return np.concatenate([near, mix, lone]), crop


def opencv(query: np.ndarray, crop: np.ndarray) -> List[Tuple[int, int, float]]:
    found = cv2.BFMatcher(cv2.NORM_L2).knnMatch(query, crop, k=2)
    return [(m.queryIdx, m.trainIdx, m.distance) for m, n in found
            if m.distance < RATIO * n.distance]


def same_as_opencv(seed: int, chunk: int, device: str) -> None:
    query, crop = made_up(seed)
    want: List[Tuple[int, int, float]] = opencv(query, crop)
    got: List[Tuple[int, int, float]] = [
        (m.queryIdx, m.trainIdx, m.distance)
        for m in knn_ratio_matches(query, crop, RATIO, torch.device(device), chunk)]
    assert len(want) > 50, f'only {len(want)} matches: the data tests nothing'
    assert [w[:2] for w in want] == [g[:2] for g in got], (
        f'{len(want)} matches by OpenCV, {len(got)} here; '
        f'{len({w[:2] for w in want} ^ {g[:2] for g in got})} differ')
    worst: float = max(abs(w[2] - g[2]) for w, g in zip(want, got))
    assert worst < 1e-3, f'a distance differs by {worst}'


def main() -> int:
    print('stage 3 ratio matcher')
    for seed in (0, 1, 2):
        check(f'cpu, seed {seed}: the matches and distances are OpenCV\'s',
              lambda seed=seed: same_as_opencv(seed, 2048, 'cpu'))
    check('cpu, a chunk smaller than the query: the same',
          lambda: same_as_opencv(3, 64, 'cpu'))
    if torch.cuda.is_available():
        for seed in (0, 1):
            check(f'cuda, seed {seed}: the same',
                  lambda seed=seed: same_as_opencv(seed, 2048, 'cuda'))
    else:
        print('  (no cuda here: the GPU run is not tested)')

    def keeps_nothing_when_nothing_is_close() -> None:
        rng = np.random.default_rng(9)
        q = rng.uniform(0, 255, (50, 128)).astype(np.float32)
        c = rng.uniform(0, 255, (200, 128)).astype(np.float32)
        assert len(knn_ratio_matches(q, c, 0.01, torch.device('cpu'))) == 0
    check('a ratio nothing passes gives no matches', keeps_nothing_when_nothing_is_close)

    failed: List[str] = [n for n, ok in _RESULTS if not ok]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
