"""
估計 query 顯微照片的 MPP —— 用「倍率指紋」比對 WSI 各層(跟位置無關)。

原理:組織在某倍率下的「結構大小(像素)」全片一致,所以
  - 頻率重心(spectral centroid):倍率越高 → 結構越大 → 重心頻率越低
  - 自相關長度(autocorr length):倍率越高 → 特徵長度越大
這兩個量「只看倍率、不看是哪一塊組織」,所以稀疏 sample 幾塊就能估,
不必命中 query 的真實位置。

流程:每層 sample 幾塊 tile → 算指紋取中位數 → 找跟 query 指紋最近的層
      → 那層的 MPP 就是 query 的(再內插出連續值)。

依賴:pip install openslide-python opencv-python numpy
用法:python estimate_query_mpp.py slide.svs query.jpg
"""
import argparse
import numpy as np
import cv2
import openslide


def _windowed(gray):
    g = gray.astype(np.float32)
    g = g - g.mean()
    win = np.hanning(g.shape[0])[:, None] * np.hanning(g.shape[1])[None, :]
    return g * win


def spectral_centroid(gray):
    """能量重心頻率;倍率越高越小。"""
    g = _windowed(gray)
    power = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = g.shape
    y, x = np.indices((h, w))
    r = np.hypot(x - w // 2, y - h // 2).astype(int)
    prof = np.bincount(r.ravel(), power.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    prof[0] = 0.0                       # 去 DC
    f = np.arange(len(prof))
    return float((f * prof).sum() / (prof.sum() + 1e-9))


def autocorr_length(gray):
    """自相關掉到 0.5 的距離(像素);倍率越高越大。"""
    g = gray.astype(np.float32) - float(gray.mean())
    F = np.fft.fft2(g)
    ac = np.fft.fftshift(np.real(np.fft.ifft2(F * np.conj(F))))
    if ac.max() <= 0:
        return 0.0
    ac /= ac.max()
    line = ac[g.shape[0] // 2, g.shape[1] // 2:]
    below = np.where(line < 0.5)[0]
    return float(below[0]) if len(below) else float(len(line))


def level_fingerprint(wsi, level, stat, tile, n=40, min_std=6):
    """該層的倍率指紋:sample n 塊非空白 tile,取統計量中位數。"""
    W, H = wsi.level_dimensions[level]
    if W <= tile or H <= tile:
        return np.nan
    ds = wsi.level_downsamples[level]
    vals, tries = [], 0
    while len(vals) < n and tries < n * 6:
        tries += 1
        x = np.random.randint(0, W - tile)
        y = np.random.randint(0, H - tile)
        t = np.array(wsi.read_region((int(x * ds), int(y * ds)), level,
                                     (tile, tile)).convert("L"))
        if t.std() < min_std:           # 跳過背景/空白
            continue
        vals.append(stat(t))
    return float(np.median(vals)) if vals else np.nan


def interp_mpp(pairs, q_val):
    """pairs: [(mpp, stat)],統計量單調對應 mpp,於 log-mpp 線性內插。"""
    pairs = sorted([p for p in pairs if not np.isnan(p[1])], key=lambda p: p[1])
    stats = [p[1] for p in pairs]
    mpps = [p[0] for p in pairs]
    if q_val <= stats[0]:
        return mpps[0]
    if q_val >= stats[-1]:
        return mpps[-1]
    for i in range(len(stats) - 1):
        if stats[i] <= q_val <= stats[i + 1]:
            t = (q_val - stats[i]) / (stats[i + 1] - stats[i] + 1e-9)
            return float(np.exp(np.log(mpps[i]) * (1 - t) + np.log(mpps[i + 1]) * t))
    return mpps[-1]


def estimate(wsi_path, query_path, tile=None, samples=40):
    wsi = openslide.OpenSlide(wsi_path)
    base_mpp = wsi.properties.get("openslide.mpp-x")
    if not base_mpp:
        raise ValueError("WSI 沒有 openslide.mpp-x metadata,無法建對照曲線")
    base_mpp = float(base_mpp)

    q = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2GRAY)
    if q is None:
        raise FileNotFoundError(query_path)
    if tile is None:
        tile = min(q.shape[0], q.shape[1], 256)
    cy, cx, h = q.shape[0] // 2, q.shape[1] // 2, tile // 2
    q_crop = q[cy - h:cy - h + tile, cx - h:cx - h + tile]   # 中央裁切,不 resize

    for name, stat in [("頻率重心 spectral_centroid", spectral_centroid),
                       ("自相關長度 autocorr_length", autocorr_length)]:
        ref = [(base_mpp * wsi.level_downsamples[lv],
                level_fingerprint(wsi, lv, stat, tile, samples), lv)
               for lv in range(wsi.level_count)]
        valid = [(m, s, lv) for m, s, lv in ref if not np.isnan(s)]
        q_val = stat(q_crop)
        nearest = min(valid, key=lambda r: abs(r[1] - q_val))
        mpp_i = interp_mpp([(m, s) for m, s, _ in valid], q_val)

        print(f"=== {name} ===")
        print(f"  query 指紋 = {q_val:.3f}")
        print(f"  最近層 = level {nearest[2]} (MPP {nearest[0]:.3f})")
        print(f"  內插估計 MPP ≈ {mpp_i:.3f}")
        spread = ", ".join(f"L{lv}={s:.2f}(MPP{m:.2f})" for m, s, lv in valid)
        print(f"  各層指紋: {spread}\n")


def estimate_knn(wsi_path, query_path, tile=None, samples=40, k=3):
    """
    Fuse spectral_centroid + autocorr_length into a 2-D feature vector,
    then use distance-weighted KNN in log-MPP space to estimate MPP.
    Returns the estimated MPP (float).
    """
    wsi = openslide.OpenSlide(wsi_path)
    base_mpp = wsi.properties.get("openslide.mpp-x")
    if not base_mpp:
        raise ValueError("WSI 沒有 openslide.mpp-x metadata")
    base_mpp = float(base_mpp)

    q = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2GRAY)
    if q is None:
        raise FileNotFoundError(query_path)
    if tile is None:
        tile = min(q.shape[0], q.shape[1], 256)
    cy, cx, h = q.shape[0] // 2, q.shape[1] // 2, tile // 2
    q_crop = q[cy - h: cy - h + tile, cx - h: cx - h + tile]

    # ── Build 2-D reference table ────────────────────────────────────────────
    ref_mpps, ref_feats = [], []
    for lv in range(wsi.level_count):
        mpp_lv = base_mpp * wsi.level_downsamples[lv]
        sc = level_fingerprint(wsi, lv, spectral_centroid, tile, samples)
        ac = level_fingerprint(wsi, lv, autocorr_length,   tile, samples)
        if not (np.isnan(sc) or np.isnan(ac)):
            ref_mpps.append(mpp_lv)
            ref_feats.append([sc, ac])

    ref_feats = np.array(ref_feats)   # (n_levels, 2)
    ref_mpps  = np.array(ref_mpps)

    # ── Query feature vector ─────────────────────────────────────────────────
    q_feat = np.array([spectral_centroid(q_crop), autocorr_length(q_crop)])

    # ── Z-score normalise (each dim independently) ───────────────────────────
    mean = ref_feats.mean(axis=0)
    std  = ref_feats.std(axis=0) + 1e-9
    ref_norm = (ref_feats - mean) / std
    q_norm   = (q_feat   - mean) / std

    # ── KNN: distance-weighted MPP in log space ──────────────────────────────
    dists = np.linalg.norm(ref_norm - q_norm, axis=1)
    k = min(k, len(dists))
    idx = np.argsort(dists)[:k]

    nearest_d = dists[idx]
    nearest_m = ref_mpps[idx]

    if nearest_d[0] < 1e-9:          # exact hit
        estimated = float(nearest_m[0])
    else:
        w = 1.0 / (nearest_d + 1e-9)
        w /= w.sum()
        estimated = float(np.exp((w * np.log(nearest_m)).sum()))

    print("=== KNN (k={}) ===".format(k))
    print(f"  query 特徵 = sc:{q_feat[0]:.3f}  ac:{q_feat[1]:.3f}")
    for rank, i in enumerate(idx):
        print(f"  rank{rank+1}: level MPP={ref_mpps[i]:.3f}  "
              f"sc={ref_feats[i,0]:.2f}  ac={ref_feats[i,1]:.2f}  dist={dists[i]:.4f}")
    print(f"  估計 MPP ≈ {estimated:.4f}\n")
    return estimated


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wsi", help="WSI 檔(.svs/.ndpi/...)")
    ap.add_argument("query", help="query 顯微照片")
    ap.add_argument("--tile", type=int, default=None, help="比對視窗大小(預設取 query 與 256 的較小值)")
    ap.add_argument("--samples", type=int, default=40, help="每層 sample 幾塊")
    ap.add_argument("--k", type=int, default=3, help="KNN 的 K 值")
    args = ap.parse_args()
    estimate(args.wsi, args.query, args.tile, args.samples)
    estimate_knn(args.wsi, args.query, args.tile, args.samples, args.k)