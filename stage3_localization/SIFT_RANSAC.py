from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


from ConfigIdentity import IdentifiedConfig, register             # noqa: E402
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


# ── config ────────────────────────────────────────────────────────────────────

@register('sift-ransac')
@dataclass(frozen=True)
class SiftRansacConfig(IdentifiedConfig):
    '''What stage 3 is: how many inliers make a fit, how far past the window
    the crop reaches, the Lowe ratio, RANSAC's reprojection threshold, and how
    many of stage 2's candidates are verified. `workers` is how many are
    verified at once -- it changes no result.'''
    min_inliers: int = 10
    padding:     int = 2
    ratio:       float = 0.75
    ransac_px:   float = 5.0
    n_verify:    int = 10
    workers:     int = 8

    BASELINE = {'min_inliers': 10, 'padding': 2, 'ratio': 0.75,
                'ransac_px': 5.0, 'n_verify': 10}
    NOT_IDENTITY = ('workers',)
    #: The crop, the match and the fit (ConfigIdentity rule 3).
    VERSION = 0


#: Named localizers, every field written out (test_config_identity's recipe
#: lint); `--stage3 sift:<name>`.
SIFT_RECIPES: Dict[str, SiftRansacConfig] = {
    'default': SiftRansacConfig(min_inliers=10, padding=2, ratio=0.75,
                                ransac_px=5.0, n_verify=10, workers=8),
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SiftRansacResult:
    '''Where the query is, LEVEL-0 and fractional: the crop's read origin
    plus the homography's offset times ds, with nothing truncated on the way
    (see stage2_retrieval/StageInterface.py for why that matters).

    `x0, y0` is where the query's own (0, 0) landed; `center_x0, center_y0`
    its centre through H. The centre is the rotation-invariant anchor -- the
    query is rotated about it -- so prefer it when the orientation is unknown.
    On a failed fit both fall back to the candidate window's own position.

    `crop_l0` is the crop's level-0 box (x, y, w, h). `pairs` holds every
    ratio-test match as rows (qx, qy, x0, y0, distance, inlier): the query
    pixel, where its crop match sits at level 0, the descriptor distance, and
    whether RANSAC kept it.'''
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
    crop_l0: Optional[Tuple[float, float, float, float]] = None
    pairs: Optional[np.ndarray] = field(default=None, repr=False)

    def row(self) -> dict:
        '''This result as one table row: the candidate it verified, where the
        query landed, the fit and the crop. `pairs` is `pair_rows`'.'''
        c = self.candidate
        H = (np.asarray(self.H, dtype=np.float64).reshape(-1) if self.H is not None
             else [None] * 9)
        crop = self.crop_l0 or (None,) * 4
        return dict(
            rank=self.rank + 1, region=c.region_index, lattice=c.lattice,
            row=c.row, col=c.col, rotation=c.rotation,
            success=self.success, n_good=self.match_count,
            n_inliers=self.inlier_count,
            x0=self.x0, y0=self.y0,
            center_x0=self.center_x0, center_y0=self.center_y0,
            **{f'h{i // 3}{i % 3}': (None if v is None else float(v))
               for i, v in enumerate(H)},
            crop_x0=crop[0], crop_y0=crop[1], crop_w0=crop[2], crop_h0=crop[3])

    def pair_rows(self) -> List[dict]:
        if self.pairs is None:
            return []
        return [dict(rank=self.rank + 1, qx=float(p[0]), qy=float(p[1]),
                     x0=float(p[2]), y0=float(p[3]), dist=float(p[4]),
                     inlier=bool(p[5])) for p in self.pairs]


# ── Localizer class ───────────────────────────────────────────────────────────

class SiftRansacLocalizer:
    '''
    Stage 3: SIFT + RANSAC inside the candidate windows of stage 2.

        loc = SiftRansacLocalizer(SIFT_RECIPES['default']).build(wsi)
        results = loc.localize_top(query, candidate_set)   # one per verified rank
        result = loc.localize(query, candidate_set, rank=0)

    Retrieval places the query to a tile; this places it to a pixel:

      read_wsi_crop       the candidate's window +- `padding` tiles, clipped to
                          its region, read on demand at the candidate set's
                          level; `crop_origin_l0` is the level-0 integer the
                          read started at
      detect_and_match    SIFT on query and crop, BFMatcher, Lowe ratio
      estimate_homography RANSAC H: query px -> crop px; a crop pixel (u, v)
                          is level-0 crop_origin_l0 + (u, v) * ds, so the
                          query's (0, 0) and centre go through H and land at
                          level-0, fractional

    `localize_top` verifies the first `n_verify` candidates: the query's SIFT
    once, each candidate's crop, match and fit on its own thread. A fit with
    fewer than `min_inliers` inliers, or a degenerate H, falls back to the
    candidate window's own position. Intermediate state stays on self for the
    figures; each step builds the ones before it if not called yet.
    '''

    def __init__(self, cfg: SiftRansacConfig):
        if not isinstance(cfg, SiftRansacConfig):
            raise TypeError(f'SiftRansacLocalizer takes a SiftRansacConfig (a '
                            f'SIFT_RECIPES entry or a replace of one), got '
                            f'{type(cfg).__name__}')
        self.cfg = cfg
        self.reader: Optional[SlideReader] = None
        self._reset(None, None, 0)

    @property
    def min_inliers(self) -> int:
        return self.cfg.min_inliers

    @property
    def padding(self) -> int:
        return self.cfg.padding

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

    def localize_top(self, query, cs: CandidateSet,
                     n: Optional[int] = None) -> List[SiftRansacResult]:
        '''Every one of the first `n` (default `n_verify`) candidates,
        verified, in rank order. The query's keypoints are computed once and
        shared; each rank runs on a localizer of its own, `workers` at a time.'''
        n = min(self.cfg.n_verify if n is None else int(n), len(cs))
        if n <= 0:
            return []
        self.prepare(query, cs, 0)
        kps, descs = self._query_features(self.query)
        query = self.query

        def one(rank: int) -> SiftRansacResult:
            loc = SiftRansacLocalizer(self.cfg).build(self.reader)
            loc._reset(query, cs, rank)
            loc.query_kps, loc.query_descs = kps, descs
            return loc.estimate_homography()

        if self.cfg.workers <= 1 or n == 1:
            return [one(r) for r in range(n)]
        with ThreadPoolExecutor(max_workers=min(self.cfg.workers, n)) as pool:
            return list(pool.map(one, range(n)))

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

    @staticmethod
    def _query_features(query: QueryPatchContainer):
        sift = cv2.SIFT_create()
        gray = cv2.cvtColor(query.img, cv2.COLOR_RGB2GRAY)
        return sift.detectAndCompute(gray, None)

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
        '''SIFT detect on query + wsi_crop, then BFMatcher with the Lowe ratio
        test. The query's keypoints are reused when already set (`localize_top`).'''
        if self.wsi_crop is None:
            self.read_wsi_crop()

        sift = cv2.SIFT_create()
        if self.query_kps is None:
            self.query_kps, self.query_descs = self._query_features(self.query)
        c_gray = cv2.cvtColor(self.wsi_crop, cv2.COLOR_RGB2GRAY)
        self.crop_kps, self.crop_descs = sift.detectAndCompute(c_gray, None)

        # knnMatch(k=2) needs two crop descriptors to give every query its
        # second neighbour for the ratio test; a blank crop has fewer.
        if (self.query_descs is None or self.crop_descs is None
                or len(self.crop_descs) < 2):
            self.good_matches = []
            return self.good_matches

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(self.query_descs, self.crop_descs, k=2)
        self.good_matches = [m for m, n in matches
                             if m.distance < self.cfg.ratio * n.distance]
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
        keep = np.zeros(n_matches, dtype=bool)

        # Fallback: the candidate window's own place. The query footprint is
        # the ROTATED query's, so width/height swap at 90/270.
        h_q, w_q = self.query.img.shape[:2]
        w_eff, h_eff = (h_q, w_q) if c.rotation in (90, 270) else (w_q, h_q)
        x0, y0 = (float(v) for v in cs.origin_l0(c))
        cx0, cy0 = x0 + w_eff / 2.0 * cs.ds, y0 + h_eff / 2.0 * cs.ds

        src = np.float32([self.query_kps[m.queryIdx].pt
                          for m in self.good_matches]).reshape(-1, 2)
        dst = np.float32([self.crop_kps[m.trainIdx].pt
                          for m in self.good_matches]).reshape(-1, 2)
        if n_matches >= 4:
            H, mask = cv2.findHomography(src.reshape(-1, 1, 2),
                                         dst.reshape(-1, 1, 2),
                                         cv2.RANSAC, self.cfg.ransac_px)
            if H is not None:
                # Count first, reject second: the inlier count is what tells the
                # two failure modes apart in the log, and it stays true of the
                # match even when the model fitted to it is thrown away.
                keep = mask.reshape(-1).astype(bool)
                inliers = int(keep.sum())
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

        ds = cs.ds
        ox, oy = self.crop_origin_l0
        pairs = np.zeros((n_matches, 6), dtype=np.float64)
        if n_matches:
            pairs[:, 0:2] = src
            pairs[:, 2] = ox + dst[:, 0] * ds
            pairs[:, 3] = oy + dst[:, 1] * ds
            pairs[:, 4] = [m.distance for m in self.good_matches]
            pairs[:, 5] = keep
        crop_h, crop_w = self.wsi_crop.shape[:2]
        self.result = SiftRansacResult(
            x0=float(x0), y0=float(y0), center_x0=float(cx0), center_y0=float(cy0),
            H=H, inlier_count=inliers, match_count=n_matches, success=success,
            rank=self.rank, candidate=c, ds=ds, level=cs.level,
            crop_l0=(float(ox), float(oy), crop_w * ds, crop_h * ds),
            pairs=pairs)
        return self.result
