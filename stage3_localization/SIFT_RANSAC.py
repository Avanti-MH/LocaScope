import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# utilities/ by hand, then _paths for the rest -- the idiom stage1_estimation
# uses. The project root it adds is what resolves `stage2_retrieval.X`.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'utilities'))
from _paths import setup_import_paths                                   # noqa: E402
setup_import_paths()

from PatchingLib import QueryPatchContainer                     # noqa: E402
from ReadGeometry import ReadSpec                              # noqa: E402
from SlideReader import SlideReader                            # noqa: E402
from stage2_retrieval.StageInterface import Candidate, CandidateSet  # noqa: E402


# ── homography predicate ──────────────────────────────────────────────────────

def is_invertible(H) -> bool:
    """Can this H be inverted at all? The first line against a degenerate one.

    Ask the question the consumers actually ask, rather than thresholding the
    determinant: a homography is defined only up to scale, so any threshold on
    det(H) is a guess about that scale, and the value being guessed at has no
    fixed meaning between two shots. `cv2.invert` with DECOMP_LU returns 0 when
    the factorisation hits a zero pivot, which is the same condition that makes
    `np.linalg.inv` raise LinAlgError.

    A singular H is not a homography. It collapses the plane onto a line, so
    infinitely many source points share one destination and no inverse mapping
    exists. `cv2.warpPerspective` does NOT say so: it inverts M internally and
    ignores invert()'s return code, so it paints black instead of failing.

    This catches only the exact half of the problem. log/TODO.log:476 records
    the other half as failure mode B: an H degenerate enough to map every match
    to one place, scoring 233-278 inliers with a ratio near 1, because every
    point passes a geometric check made with the broken H itself. Those are
    near-singular rather than singular, so LU still inverts them and this
    returns True.
    """
    ok, _ = cv2.invert(np.asarray(H, dtype=np.float64), flags=cv2.DECOMP_LU)
    return bool(ok)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SiftRansacResult:
    '''Where the query is, LEVEL-0 and fractional: the crop's read origin
    plus the homography's offset times ds, with nothing truncated on the way
    (see stage2_retrieval/StageInterface.py for why that matters).

    `x0, y0` is where the query's own (0, 0) landed; `center_x0, center_y0`
    its centre through H. The centre is the rotation-invariant anchor -- the
    query is rotated about it -- so prefer it when the orientation is unknown.
    On a failed fit both fall back to the candidate window's own position.'''
    x0: float
    y0: float
    center_x0: float
    center_y0: float
    H: Optional[np.ndarray]  # 3x3, query px -> wsi_crop px; None if failed
    inlier_count: int
    match_count: int
    success: bool
    rank: int                # which candidate of the CandidateSet
    candidate: Candidate
    ds: float
    level: int


# ── Localizer class ───────────────────────────────────────────────────────────

class SiftRansacLocalizer:
    '''
    Stage 3: SIFT + RANSAC inside one candidate window.

        loc = SiftRansacLocalizer(min_inliers=10, padding=2).build(wsi)
        result = loc.localize(query, candidate_set, rank=0)

    Retrieval places the query to a tile; this places it to a pixel:

      read_wsi_crop       the candidate's window +- `padding` tiles, clipped to
                          its region, read on demand at the candidate set's
                          level; `crop_origin_l0` is the level-0 integer the
                          read started at
      detect_and_match    SIFT on query and crop, BFMatcher, Lowe ratio 0.75
      estimate_homography RANSAC H: query px -> crop px; a crop pixel (u, v)
                          is level-0 crop_origin_l0 + (u, v) * ds, so the
                          query's (0, 0) and centre go through H and land at
                          level-0, fractional

    A fit with fewer than `min_inliers` inliers, or a degenerate H, falls back
    to the candidate window's own position. Intermediate state stays on self
    for the figures; each step builds the ones before it if not called yet.
    '''

    def __init__(self, min_inliers: int = 10, padding: int = 2):
        self.min_inliers = min_inliers
        self.padding = padding
        self.reader: Optional[SlideReader] = None
        self._reset(None, None, 0)

    def build(self, wsi) -> 'SiftRansacLocalizer':
        '''Bind a slide. The localizer reads its own crops: a SlideReader of
        its own, not one borrowed from stage 2.'''
        self.reader = wsi if isinstance(wsi, SlideReader) else SlideReader(wsi)
        return self

    def _reset(self, query, cs, rank) -> None:
        self.query = query
        self.cs = cs
        self.rank = rank
        self.candidate: Optional[Candidate] = cs[rank] if cs is not None else None
        self.wsi_crop: Optional[np.ndarray] = None
        #: level-0 integer the crop's read started at: wsi_crop[0, 0]
        self.crop_origin_l0: Optional[Tuple[int, int]] = None
        self.query_kps = None
        self.query_descs: Optional[np.ndarray] = None
        self.crop_kps = None
        self.crop_descs: Optional[np.ndarray] = None
        self.good_matches: Optional[list] = None
        self.result: Optional[SiftRansacResult] = None

    def localize(self, query, cs: CandidateSet, rank: int = 0) -> SiftRansacResult:
        '''Stage 3 on stage 2's output: candidate `rank` of `cs`.'''
        self.prepare(query, cs, rank)
        return self.estimate_homography()

    def prepare(self, query, cs: CandidateSet, rank: int = 0) -> 'SiftRansacLocalizer':
        '''Set the query and the candidate without running anything -- for a
        caller that steps through the stages (the figures).'''
        if self.reader is None:
            raise RuntimeError('call build(wsi) first')
        if not isinstance(query, QueryPatchContainer):
            qc = QueryPatchContainer(np.asarray(query))
            qc.extract_all(cs.grids[cs[rank].region_index].tile_size, overlap=True)
            query = qc
        self._reset(query, cs, rank)
        return self

    # ── Stage 1 ──────────────────────────────────────────────────────────────

    def read_wsi_crop(self, padding: Optional[int] = None) -> np.ndarray:
        '''Read the candidate's window +- padding tiles, clipped to its region.'''
        pad = padding if padding is not None else self.padding
        cs, c = self.cs, self.candidate
        grid = cs.grids[c.region_index]
        ts = grid.tile_size

        # The window's top-left, level px from the region's own origin
        local_x, local_y = cs.window_local(c)
        # Window covers the query rounded up to whole tiles
        win_w = int(np.ceil(self.query.width  / ts)) * ts
        win_h = int(np.ceil(self.query.height / ts)) * ts

        x0 = max(0, local_x - pad * ts)
        y0 = max(0, local_y - pad * ts)
        x1 = min(grid.width, local_x + win_w + pad * ts)
        y1 = min(grid.height, local_y + win_h + pad * ts)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f'empty WSI crop for candidate {self.rank} '
                f'(region {c.region_index} {c.lattice} r{c.row} c{c.col}): '
                f'window at local ({local_x}, {local_y}) in a {grid.width}x'
                f'{grid.height} region; {win_w}x{win_h} +{pad} tiles gives '
                f'x[{x0}:{x1}] y[{y0}:{y1}]')

        # Anchored at the region's level-0 origin, the region's own phase, at
        # the set's ds: a native read, no resampling. The integer it starts at
        # is what is booked -- not a level-n position, which would truncate.
        self.crop_origin_l0 = grid.local_to_l0(x0, y0)
        crop = self.reader.read(self.crop_origin_l0[0], self.crop_origin_l0[1],
                                ReadSpec(x1 - x0, y1 - y0), cs.ds, level=cs.level)
        if crop is None:
            raise ValueError(f'crop at level-0 {self.crop_origin_l0} size '
                             f'{x1 - x0}x{y1 - y0} falls off the slide')
        self.wsi_crop = crop
        return self.wsi_crop

    def crop_to_l0(self, u, v) -> Tuple[float, float]:
        '''Level-0 position of crop pixel (u, v) -- fractional, through
        openslide's own rule: the read origin plus (u, v) level px times ds.'''
        ds = self.cs.ds
        return self.crop_origin_l0[0] + u * ds, self.crop_origin_l0[1] + v * ds

    # ── Stage 2 ──────────────────────────────────────────────────────────────

    def detect_and_match(self) -> list:
        '''SIFT detect on query + wsi_crop, then BFMatcher with Lowe ratio test.'''
        if self.wsi_crop is None:
            self.read_wsi_crop()

        sift = cv2.SIFT_create()
        q_gray = cv2.cvtColor(self.query.img, cv2.COLOR_RGB2GRAY)
        c_gray = cv2.cvtColor(self.wsi_crop,  cv2.COLOR_RGB2GRAY)

        self.query_kps, self.query_descs = sift.detectAndCompute(q_gray, None)
        self.crop_kps,  self.crop_descs  = sift.detectAndCompute(c_gray, None)

        # knnMatch(k=2) needs two crop descriptors to give every query its
        # second neighbour for the ratio test; a blank crop has fewer.
        if (self.query_descs is None or self.crop_descs is None
                or len(self.crop_descs) < 2):
            self.good_matches = []
            return self.good_matches

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(self.query_descs, self.crop_descs, k=2)
        self.good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        return self.good_matches

    # ── Stage 3 ──────────────────────────────────────────────────────────────

    def estimate_homography(self) -> SiftRansacResult:
        '''RANSAC homography -> the query's top-left and centre at level-0.'''
        if self.good_matches is None:
            self.detect_and_match()

        cs, c = self.cs, self.candidate
        n_matches = len(self.good_matches)
        H = None
        inliers = 0
        success = False

        # Fallback: the candidate window's own place. The query footprint is
        # the ROTATED query's, so width/height swap at 90/270.
        h_q, w_q = self.query.img.shape[:2]
        w_eff, h_eff = (h_q, w_q) if c.rotation in (90, 270) else (w_q, h_q)
        x0, y0 = (float(v) for v in cs.origin_l0(c))
        cx0, cy0 = x0 + w_eff / 2.0 * cs.ds, y0 + h_eff / 2.0 * cs.ds

        if n_matches >= 4:
            src_pts = np.float32(
                [self.query_kps[m.queryIdx].pt for m in self.good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [self.crop_kps[m.trainIdx].pt for m in self.good_matches]
            ).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                # Count first, reject second: the inlier count is what tells the
                # two failure modes apart in the log, and it stays true of the
                # match even when the model fitted to it is thrown away.
                inliers = int(mask.sum())
                if is_invertible(H):
                    success = inliers >= self.min_inliers
                else:
                    # findHomography returns a degenerate H when RANSAC's support
                    # is the bare 4-point minimal sample: 4 pairs give exactly the
                    # 8 equations H needs, so any quadruple is its own inlier set,
                    # and a collinear or coincident one has no inverse. Dropped so
                    # consumers' `H is not None` means what it says.
                    H = None
                if success:
                    pts = np.array([[[0.0, 0.0]], [[w_q / 2.0, h_q / 2.0]]],
                                   dtype=np.float32)
                    mapped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
                    x0, y0 = self.crop_to_l0(*mapped[0])
                    cx0, cy0 = self.crop_to_l0(*mapped[1])

        self.result = SiftRansacResult(
            x0=float(x0), y0=float(y0), center_x0=float(cx0), center_y0=float(cy0),
            H=H, inlier_count=inliers, match_count=n_matches, success=success,
            rank=self.rank, candidate=c, ds=cs.ds, level=cs.level)
        return self.result
