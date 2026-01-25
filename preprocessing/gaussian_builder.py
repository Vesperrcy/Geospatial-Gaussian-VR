import numpy as np
import open3d as o3d
from pathlib import Path
import argparse

from typing import Optional

# Optional (fast CPU KNN)
try:
    from scipy.spatial import cKDTree  # type: ignore
    _HAS_SCIPY = True
except Exception:
    cKDTree = None
    _HAS_SCIPY = False

# ==== 配置 ====

INPUT_PATH = Path("data/TumTLS_v2.ply")
OUTPUT_PATH = Path("data/TumTLS_v2_gaussians_demo.npz")

# ==============================
# Preset switch
# ==============================
# verify: 小数据验证（清晰优先，允许洞）:
# python preprocessing/gaussian_builder.py --preset verify
# final : 高密度/全量（连续优先，可更“面”）
# python preprocessing/gaussian_builder.py --preset final
# python preprocessing/gaussian_builder.py --preset final --aniso_block 80000
PRESET_DEFAULT = "verify"
PRESET = PRESET_DEFAULT  # overwritten by apply_preset()

# ==============================
# Logging (speed)
# ==============================
VERBOSE = True          # False = 少打印更快
LOG_EVERY = 50000       # 进度打印间隔（点数大时建议 50000~200000）

def log(msg: str):
    if VERBOSE:
        print(msg)

# ==============================
# Sampling strategy (for large TLS clouds)
# ==============================
SAMPLE_MODE = "random"   # "random" | "voxel" | "all" (overwritten by preset)
VOXEL_SIZE = 0.02        # meters; 0.01~0.05 typical

MAX_GAUSSIANS = 400000
MAX_GAUSSIANS_AFTER_SAMPLING = MAX_GAUSSIANS

K_NEIGHBORS_ISO = 8

# ---- Step 6: 法线对齐各向异性 ----
K_NEIGHBORS_NORMAL = 24
K_NEIGHBORS_TANGENT = 64


# ---- Speed: use Open3D built-in normal estimation ----
USE_O3D_NORMALS = True

# ==============================
# Batch anisotropy (million+ points)
# ==============================
USE_BATCH_ANISO = True          # if True and SciPy available, use cKDTree + numpy batching
ANISO_BLOCK_SIZE = 50000        # 20k~200k depending on RAM (higher = faster, more RAM)
DT_USE_PARTITION = True         # use np.partition for per-row percentile (fast)

MIN_NEIGHBORS_NORMAL = 10
MIN_NEIGHBORS_TANGENT = 16
KNN_DISTANCE_CLIP_ALPHA = 2.5
RADIUS_FACTOR_FROM_ISO = 6.0

# --- 法线方向薄片化 ---
SIGMA_N_MODE = "fixed"   # "fixed" | "relative"
SIGMA_N_FIXED = 0.008
SIGMA_N_REL_FACTOR = 0.25
SIGMA_N_MIN = 0.004
SIGMA_N_MAX = 0.015

# --- 切向距离加权（TLS 强烈推荐） ---
USE_TANGENT_WEIGHTING = True
TANGENT_WEIGHT_SIGMA_FACTOR = 1.6
TANGENT_WEIGHT_SIGMA_W_MIN = 0.02
TANGENT_WEIGHT_SIGMA_W_MAX = 0.25

# --- 切向 regularization：收紧范围，避免雾化 ---
TANGENT_REL_MIN = 0.35
TANGENT_REL_MAX = 1.1

# ---- NEW: 贴片绝对大小用“局部点间距”控制 ----
DT_PERCENTILE = 20
DT_TO_SIGMA_FACTOR = 0.75

# PCA 仅用于控制形状比
AR_MIN = 1.0
AR_MAX = 3.0

# clamp
USE_ADAPTIVE_CLAMP = True
S_MIN = 0.02
S_MAX = 0.0875

ADAPTIVE_Q_MIN = 2
ADAPTIVE_Q_MAX = 95
ADAPTIVE_MIN_FACTOR = 0.35
ADAPTIVE_MAX_FACTOR = 0.9

ABS_S_MIN = 0.005
ABS_S_MAX = 0.50

# ---- S_MAX policy ----
S_MAX_POLICY = "max"   # "siso" | "dt" | "max"
USE_DT_BASED_SMAX = True
DT_SMAX_PERCENTILE = 90
DT_SMAX_FACTOR = 0.75
DT_STATS_MAX_POINTS = 50000

# ==== 坐标系转换 ====
APPLY_UNITY_FRAME = True
M_ENU_TO_UNITY = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0],
    [ 0.0,  1.0,  0.0],
], dtype=np.float32)

# ==== Recenter ====
APPLY_RECENTER = True
RECENTER_MODE = "bbox_center"

# ==============================
# Pre-shift large coordinates (TLS / UTM float precision fix)
# ==============================
ENABLE_PRE_SHIFT = True
PRE_SHIFT_MODE = "bbox_center"   # "bbox_center" | "mean" | "first_point"


def apply_preset(preset: str):
    """Apply two stable presets: verify (sharp) and final (continuous)."""
    global PRESET
    global SAMPLE_MODE, VOXEL_SIZE, MAX_GAUSSIANS, MAX_GAUSSIANS_AFTER_SAMPLING
    global TANGENT_WEIGHT_SIGMA_FACTOR, TANGENT_WEIGHT_SIGMA_W_MAX
    global DT_PERCENTILE, DT_TO_SIGMA_FACTOR
    global TANGENT_REL_MAX
    global S_MAX_POLICY, DT_SMAX_PERCENTILE, DT_SMAX_FACTOR
    global SIGMA_N_FIXED

    p = (preset or PRESET_DEFAULT).lower().strip()
    PRESET = p

    if p == "verify":
        SAMPLE_MODE = "voxel"
        VOXEL_SIZE = 0.05
        MAX_GAUSSIANS = 10000000
        MAX_GAUSSIANS_AFTER_SAMPLING = MAX_GAUSSIANS

        TANGENT_WEIGHT_SIGMA_FACTOR = 1.0
        TANGENT_WEIGHT_SIGMA_W_MAX = 0.14

        DT_PERCENTILE = 10
        DT_TO_SIGMA_FACTOR = 0.35

        TANGENT_REL_MAX = 0.80

        S_MAX_POLICY = "siso"
        DT_SMAX_PERCENTILE = 90
        DT_SMAX_FACTOR = 0.70

        SIGMA_N_FIXED = 0.006

    elif p == "final":
        SAMPLE_MODE = "voxel"
        VOXEL_SIZE = 0.02
        MAX_GAUSSIANS = -1
        MAX_GAUSSIANS_AFTER_SAMPLING = -1

        TANGENT_WEIGHT_SIGMA_FACTOR = 1.6
        TANGENT_WEIGHT_SIGMA_W_MAX = 0.25

        DT_PERCENTILE = 20
        DT_TO_SIGMA_FACTOR = 0.75

        TANGENT_REL_MAX = 1.10

        S_MAX_POLICY = "max"
        DT_SMAX_PERCENTILE = 90
        DT_SMAX_FACTOR = 0.75

        SIGMA_N_FIXED = 0.008

    else:
        raise ValueError(f"Unknown preset: {preset}")

    log(f"[INFO] Preset applied: {PRESET}")
    log(f"[INFO] Sampling mode={SAMPLE_MODE}, voxel_size={VOXEL_SIZE}, max_gaussians={MAX_GAUSSIANS}, cap_after={MAX_GAUSSIANS_AFTER_SAMPLING}")
    log(f"[INFO] dt->st: DT_PERCENTILE={DT_PERCENTILE}, DT_TO_SIGMA_FACTOR={DT_TO_SIGMA_FACTOR:.2f}; tangent sigma_factor={TANGENT_WEIGHT_SIGMA_FACTOR:.2f}, w_max={TANGENT_WEIGHT_SIGMA_W_MAX:.2f}")
    log(f"[INFO] S_MAX_POLICY={S_MAX_POLICY}, SIGMA_N_FIXED={SIGMA_N_FIXED:.3f}")


# -------------------- 工具函数 --------------------

def compute_isotropic_scales(points, kdtree, k_neighbors,
                            knn_idx: Optional[np.ndarray] = None,
                            knn_dist2: Optional[np.ndarray] = None):
    num = points.shape[0]
    scales_iso = np.zeros((num,), dtype=np.float32)

    log("[INFO] Estimating isotropic scales from neighbor distances...")

    # Fast path: cached KNN
    if knn_idx is not None and knn_dist2 is not None:
        d = np.sqrt(knn_dist2[:, 1:k_neighbors + 1].astype(np.float64))
        mean_dist = np.sqrt(np.mean((d ** 2), axis=1))
        s = (mean_dist * 0.75).astype(np.float32)
        scales_iso[:] = s

        if VERBOSE:
            dist_arr = s.astype(np.float64)
            log("[INFO] Isotropic scale summary (m): "
                f"min={dist_arr.min():.4f}, p50={np.percentile(dist_arr, 50):.4f}, "
                f"p95={np.percentile(dist_arr, 95):.4f}, max={dist_arr.max():.4f}, mean={dist_arr.mean():.4f}")
        return scales_iso

    # Slow fallback
    dist_list = []
    for i in range(num):
        k, idx, dist2 = kdtree.search_knn_vector_3d(points[i], k_neighbors)
        if k > 1:
            mean_dist = float(np.sqrt(np.mean(dist2[1:])))
        else:
            mean_dist = 0.05

        s = mean_dist * 0.75
        scales_iso[i] = s
        dist_list.append(s)

        if VERBOSE and (i > 0 and i % LOG_EVERY == 0):
            log(f"  [ISO] processed {i}/{num} points...")

    dist_arr = np.asarray(dist_list, dtype=np.float64)
    if VERBOSE:
        log("[INFO] Isotropic scale summary (m): "
            f"min={dist_arr.min():.4f}, p50={np.percentile(dist_arr, 50):.4f}, "
            f"p95={np.percentile(dist_arr, 95):.4f}, max={dist_arr.max():.4f}, mean={dist_arr.mean():.4f}")
    return scales_iso


def compute_adaptive_clamp(scales_iso: np.ndarray) -> tuple[float, float]:
    q_min = float(np.percentile(scales_iso, ADAPTIVE_Q_MIN))
    q_max = float(np.percentile(scales_iso, ADAPTIVE_Q_MAX))

    s_min = q_min * ADAPTIVE_MIN_FACTOR
    s_max = q_max * ADAPTIVE_MAX_FACTOR

    s_min = float(np.clip(s_min, ABS_S_MIN, ABS_S_MAX))
    s_max = float(np.clip(s_max, ABS_S_MIN, ABS_S_MAX))

    if s_max <= s_min:
        s_max = min(ABS_S_MAX, s_min * 2.0)

    return s_min, s_max


def orthonormalize_rotation(R):
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1.0
        R_ortho = U @ Vt
    return R_ortho.astype(np.float32)


def cov6_to_mat(cov6_row: np.ndarray) -> np.ndarray:
    xx, xy, xz, yy, yz, zz = cov6_row
    return np.array([
        [xx, xy, xz],
        [xy, yy, yz],
        [xz, yz, zz],
    ], dtype=np.float32)


def mat_to_cov6(Sigma: np.ndarray) -> np.ndarray:
    return np.array([
        Sigma[0, 0], Sigma[0, 1], Sigma[0, 2],
        Sigma[1, 1], Sigma[1, 2], Sigma[2, 2]
    ], dtype=np.float32)


def apply_frame_transform(points: np.ndarray,
                          rotations: np.ndarray,
                          cov6: np.ndarray,
                          M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts_t = (points @ M.T).astype(np.float32)

    R = rotations.reshape(-1, 3, 3).astype(np.float32)
    R_t = (M.astype(np.float32) @ R).astype(np.float32)
    for i in range(R_t.shape[0]):
        R_t[i] = orthonormalize_rotation(R_t[i])
    rot_t = R_t.reshape(-1, 9).astype(np.float32)

    cov_t = np.zeros_like(cov6, dtype=np.float32)
    MT = M.T.astype(np.float32)
    for i in range(cov6.shape[0]):
        Sigma = cov6_to_mat(cov6[i])
        Sigma_t = (M @ Sigma @ MT).astype(np.float32)
        cov_t[i] = mat_to_cov6(Sigma_t)

    return pts_t, rot_t, cov_t


def compute_recenter_origin(points: np.ndarray, mode: str = "bbox_center") -> np.ndarray:
    if points.size == 0:
        return np.zeros((3,), dtype=np.float32)

    mode = (mode or "bbox_center").lower()
    if mode == "mean":
        origin = points.mean(axis=0)
    elif mode == "first_point":
        origin = points[0]
    else:
        pmin = points.min(axis=0)
        pmax = points.max(axis=0)
        origin = (pmin + pmax) * 0.5

    return origin.astype(np.float32)


def apply_recenter(points: np.ndarray, origin: np.ndarray) -> np.ndarray:
    return (points - origin.reshape(1, 3)).astype(np.float32)


def robust_knn_indices(points, kdtree, i, k_neighbors, scales_iso):
    k, idx, dist2 = kdtree.search_knn_vector_3d(points[i], k_neighbors)
    if k <= 1:
        return np.empty((0,), dtype=np.int64)

    d = np.sqrt(np.asarray(dist2[1:], dtype=np.float64))
    neigh_idx = np.asarray(idx[1:], dtype=np.int64)

    d_med = float(np.median(d))
    clip_thr = KNN_DISTANCE_CLIP_ALPHA * d_med

    s_iso = float(scales_iso[i])
    mean_neighbor_dist = s_iso / 0.75 if s_iso > 1e-8 else 0.05
    radius_thr = RADIUS_FACTOR_FROM_ISO * mean_neighbor_dist

    thr = min(clip_thr, radius_thr)
    keep = d <= thr
    return neigh_idx[keep]


def make_tangent_basis_from_normal(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = n.astype(np.float32)
    if abs(n[2]) < 0.9:
        a = np.array([0, 0, 1], dtype=np.float32)
    else:
        a = np.array([0, 1, 0], dtype=np.float32)

    t0 = np.cross(a, n)
    t0_norm = np.linalg.norm(t0)
    if t0_norm < 1e-8:
        a = np.array([1, 0, 0], dtype=np.float32)
        t0 = np.cross(a, n)
        t0_norm = np.linalg.norm(t0)

    t0 = t0 / max(t0_norm, 1e-8)
    t1 = np.cross(n, t0)
    t1 = t1 / max(np.linalg.norm(t1), 1e-8)
    return t0.astype(np.float32), t1.astype(np.float32)


# Vectorized tangent basis from normals (batch)
def make_tangent_basis_from_normals_batch(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized tangent basis from normals.

    Args:
        n: (B,3) float32 unit normals
    Returns:
        t0, t1: (B,3) float32, orthonormal with n
    """
    n = n.astype(np.float32, copy=False)
    B = n.shape[0]

    # choose anchor a: if |nz|<0.9 use z-axis else y-axis
    a = np.zeros((B, 3), dtype=np.float32)
    use_z = (np.abs(n[:, 2]) < 0.9)
    a[use_z, 2] = 1.0
    a[~use_z, 1] = 1.0

    t0 = np.cross(a, n)
    t0n = np.linalg.norm(t0, axis=1, keepdims=True)

    # fix degenerate rows
    bad = (t0n[:, 0] < 1e-8)
    if np.any(bad):
        a2 = np.zeros((np.count_nonzero(bad), 3), dtype=np.float32)
        a2[:, 0] = 1.0
        t0_bad = np.cross(a2, n[bad])
        t0[bad] = t0_bad
        t0n[bad] = np.linalg.norm(t0_bad, axis=1, keepdims=True)

    t0 = t0 / np.maximum(t0n, 1e-8)
    t1 = np.cross(n, t0)
    t1 = t1 / np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-8)
    return t0.astype(np.float32, copy=False), t1.astype(np.float32, copy=False)


def compute_sigma_n(s_iso: float) -> float:
    if SIGMA_N_MODE.lower() == "relative":
        s = SIGMA_N_REL_FACTOR * float(s_iso)
    else:
        s = float(SIGMA_N_FIXED)
    return float(np.clip(s, SIGMA_N_MIN, SIGMA_N_MAX))


# ---- Sampling helper ----
def sample_points_for_processing(pcd: o3d.geometry.PointCloud,
                                 max_points: int,
                                 mode: str = "random",
                                 voxel_size: float = 0.02) -> o3d.geometry.PointCloud:
    mode = (mode or "random").lower()

    if mode == "all" or max_points is None or max_points <= 0:
        return pcd

    if mode == "voxel":
        vs = float(max(1e-6, voxel_size))
        return pcd.voxel_down_sample(vs)

    n = np.asarray(pcd.points).shape[0]
    if n <= max_points:
        return pcd
    idx = np.random.choice(n, int(max_points), replace=False)
    return pcd.select_by_index(idx)


# ---- Fast batched KNN (Open3D Tensor NNS) ----
def batch_knn_search(points_np: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = int(k)
    if k <= 0:
        raise ValueError("k must be > 0")

    pts = np.asarray(points_np, dtype=np.float32)
    n = pts.shape[0]

    try:
        import open3d as _o3d
        t_pts = _o3d.core.Tensor(pts, dtype=_o3d.core.Dtype.Float32)
        nns = _o3d.core.nns.NearestNeighborSearch(t_pts)
        nns.knn_index()
        idx_t, dist2_t = nns.knn_search(t_pts, k)
        idx = idx_t.numpy().astype(np.int64, copy=False)
        dist2 = dist2_t.numpy().astype(np.float64, copy=False)
        return idx, dist2
    except Exception:
        pass

    log("[WARN] Open3D tensor NNS not available; falling back to KDTreeFlann loop (slow).")
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    kdt = o3d.geometry.KDTreeFlann(pcd_tmp)

    idx = np.empty((n, k), dtype=np.int64)
    dist2 = np.empty((n, k), dtype=np.float64)

    for i in range(n):
        kk, ii, dd = kdt.search_knn_vector_3d(pts[i], k)
        if kk < k:
            ii = list(ii) + [i] * (k - kk)
            dd = list(dd) + [0.0] * (k - kk)
        idx[i, :] = np.asarray(ii[:k], dtype=np.int64)
        dist2[i, :] = np.asarray(dd[:k], dtype=np.float64)
        if VERBOSE and (i > 0 and i % LOG_EVERY == 0):
            log(f"  [KNN-FALLBACK] processed {i}/{n} points...")

    return idx, dist2


def knn_slice(knn_idx: np.ndarray, knn_dist2: np.ndarray, i: int, k_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    k_neighbors = int(k_neighbors)
    idx = knn_idx[i, 1:k_neighbors + 1]
    d = np.sqrt(knn_dist2[i, 1:k_neighbors + 1].astype(np.float64))
    return idx.astype(np.int64, copy=False), d


def robust_knn_indices_cached(knn_idx: np.ndarray,
                              knn_dist2: np.ndarray,
                              points: np.ndarray,
                              i: int,
                              k_neighbors: int,
                              scales_iso: np.ndarray) -> np.ndarray:
    if k_neighbors <= 0:
        return np.empty((0,), dtype=np.int64)

    neigh_idx, d = knn_slice(knn_idx, knn_dist2, i, k_neighbors)
    if neigh_idx.size == 0:
        return np.empty((0,), dtype=np.int64)

    d_med = float(np.median(d))
    clip_thr = KNN_DISTANCE_CLIP_ALPHA * d_med

    s_iso = float(scales_iso[i])
    mean_neighbor_dist = s_iso / 0.75 if s_iso > 1e-8 else 0.05
    radius_thr = RADIUS_FACTOR_FROM_ISO * mean_neighbor_dist

    thr = min(clip_thr, radius_thr)
    keep = d <= thr
    return neigh_idx[keep]


# ---- Fast percentile for sorted arrays ----
def fast_percentile_from_sorted(x_sorted: np.ndarray, q: float) -> float:
    n = int(x_sorted.size)
    if n <= 0:
        return 0.0
    if n == 1:
        return float(x_sorted[0])
    pos = (float(q) / 100.0) * (n - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if hi == lo:
        return float(x_sorted[lo])
    w = pos - lo
    return float((1.0 - w) * x_sorted[lo] + w * x_sorted[hi])


def weighted_cov_2d(UV: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    w = w.astype(np.float64)
    w_sum = float(np.sum(w))
    if w_sum < 1e-12:
        mu = UV.mean(axis=0)
        X = UV - mu
        C2 = (X.T @ X) / max(1, UV.shape[0] - 1)
        return mu, C2, 0.0

    mu = np.sum(UV * w.reshape(-1, 1), axis=0) / w_sum
    X = UV - mu.reshape(1, 2)
    C2 = (X.T * w) @ X / max(w_sum, 1e-12)

    Neff = (w_sum * w_sum) / max(float(np.sum(w * w)), 1e-12)
    return mu, C2, float(Neff)


def eig2x2_sym(C2: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    a = float(C2[0, 0])
    b = float(C2[0, 1])
    c = float(C2[1, 1])

    tr = a + c
    det = a * c - b * b
    disc = tr * tr - 4.0 * det
    disc = max(disc, 0.0)
    s = np.sqrt(disc)
    l1 = 0.5 * (tr + s)
    l2 = 0.5 * (tr - s)

    if abs(b) > 1e-18:
        v1 = np.array([l1 - c, b], dtype=np.float64)
    else:
        v1 = np.array([1.0, 0.0], dtype=np.float64) if a >= c else np.array([0.0, 1.0], dtype=np.float64)
    v1n = float(np.linalg.norm(v1))
    if v1n < 1e-18:
        v1 = np.array([1.0, 0.0], dtype=np.float64)
    else:
        v1 /= v1n

    v2 = np.array([-v1[1], v1[0]], dtype=np.float64)
    return l1, l2, v1.astype(np.float32), v2.astype(np.float32)


# Vectorized analytic eigen-decomposition for symmetric 2x2 matrices
def eig2x2_sym_batch(C2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized analytic eigen-decomposition for symmetric 2x2 matrices.

    Args:
        C2: (B,2,2) float64/float32 symmetric
    Returns:
        l1,l2: (B,) eigenvalues (l1>=l2)
        v1,v2: (B,2) unit eigenvectors corresponding to l1,l2
    """
    a = C2[:, 0, 0].astype(np.float64, copy=False)
    b = C2[:, 0, 1].astype(np.float64, copy=False)
    c = C2[:, 1, 1].astype(np.float64, copy=False)

    tr = a + c
    det = a * c - b * b
    disc = tr * tr - 4.0 * det
    disc = np.maximum(disc, 0.0)
    s = np.sqrt(disc)
    l1 = 0.5 * (tr + s)
    l2 = 0.5 * (tr - s)

    # v1
    v1 = np.zeros((C2.shape[0], 2), dtype=np.float64)
    use_b = (np.abs(b) > 1e-18)
    v1[use_b, 0] = (l1[use_b] - c[use_b])
    v1[use_b, 1] = b[use_b]
    # diagonal-ish
    diag = ~use_b
    v1[diag, 0] = 1.0
    v1n = np.linalg.norm(v1, axis=1, keepdims=True)
    v1 = v1 / np.maximum(v1n, 1e-18)

    v2 = np.stack([-v1[:, 1], v1[:, 0]], axis=1)
    return l1.astype(np.float64), l2.astype(np.float64), v1.astype(np.float32), v2.astype(np.float32)
def compute_normal_aligned_gaussians_batched(points: np.ndarray,
                                             scales_iso: np.ndarray,
                                             normals: np.ndarray,
                                             k_tangent: int = K_NEIGHBORS_TANGENT,
                                             block_size: int = 50000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched anisotropic gaussian estimation.

    Design notes (million+ points):
    - Uses SciPy cKDTree to query KNN for *blocks* of points.
    - Vectorizes tangent-plane projection, weighted 2D covariance, 2x2 eigens, and dt-percentile.
    - Avoids Python per-point loops; the only loop is over blocks.

    Args:
        points:   (N,3) float32
        scales_iso: (N,) float32
        normals:  (N,3) float32 (unit)
        k_tangent: K for tangent neighborhood
        block_size: points per block

    Returns:
        scales_aniso: (N,3) float32
        rotations:   (N,9) float32
        cov6:        (N,6) float32
    """
    if not _HAS_SCIPY:
        raise RuntimeError("SciPy not available (cKDTree missing). Set USE_BATCH_ANISO=False or install scipy.")

    pts = np.asarray(points, dtype=np.float32)
    N = pts.shape[0]
    k = int(k_tangent)
    if k <= 2:
        raise ValueError("k_tangent must be >= 3")

    nrm = np.asarray(normals, dtype=np.float32)
    nn = np.linalg.norm(nrm, axis=1)
    good = nn > 1e-8
    # normalize robustly
    nrm = nrm.copy()
    nrm[good] /= nn[good].reshape(-1, 1)
    nrm[~good] = np.array([0, 0, 1], dtype=np.float32)

    # outputs
    scales_aniso = np.zeros((N, 3), dtype=np.float32)
    rotations = np.zeros((N, 9), dtype=np.float32)
    cov6 = np.zeros((N, 6), dtype=np.float32)

    tree = cKDTree(pts)  # builds once

    log("[INFO] Step 6 (BATCH): cKDTree KNN + numpy broadcasting...")
    log(f"[INFO] N={N}, k_tangent={k}, block_size={block_size}")

    # constants for dt percentile rank among K neighbors
    # We'll use only neighbors excluding self if possible.
    # cKDTree.query returns self at dist=0 when querying points from the same set.
    q = float(DT_PERCENTILE)

    for b0 in range(0, N, int(block_size)):
        b1 = min(N, b0 + int(block_size))
        B = b1 - b0

        p = pts[b0:b1]                 # (B,3)
        n = nrm[b0:b1]                 # (B,3)
        s_iso = scales_iso[b0:b1].astype(np.float32, copy=False)  # (B,)

        # Query KNN for this block
        # d: (B,k), idx: (B,k)
        d, idx = tree.query(p, k=k, workers=-1)
        idx = idx.astype(np.int64, copy=False)

        # drop self if present at column 0 (dist==0)
        # keep at most k-1 neighbors
        idx_n = idx[:, 1:]

        neigh = pts[idx_n]             # (B,k-1,3)
        r = neigh - p[:, None, :]      # (B,k-1,3)

        # project to tangent plane: r_par = r - (r·n) n
        rn = np.einsum('bkc,bc->bk', r, n, optimize=True)          # (B,k-1)
        r_par = r - rn[:, :, None] * n[:, None, :]                # (B,k-1,3)

        # tangent basis (B,3)
        t0, t1 = make_tangent_basis_from_normals_batch(n)

        # u,v coordinates
        u = np.einsum('bkc,bc->bk', r_par, t0, optimize=True).astype(np.float64, copy=False)
        v = np.einsum('bkc,bc->bk', r_par, t1, optimize=True).astype(np.float64, copy=False)

        d2 = u * u + v * v

        # weights
        if USE_TANGENT_WEIGHTING:
            mean_neighbor_dist = (s_iso / 0.75).astype(np.float64)
            sigma_w = (TANGENT_WEIGHT_SIGMA_FACTOR * mean_neighbor_dist)
            sigma_w = np.clip(sigma_w, TANGENT_WEIGHT_SIGMA_W_MIN, TANGENT_WEIGHT_SIGMA_W_MAX)
            w = np.exp(-0.5 * d2 / np.maximum(sigma_w[:, None] * sigma_w[:, None], 1e-12))
            w_sum = np.sum(w, axis=1)
            w_sum = np.maximum(w_sum, 1e-12)

            mu_u = np.sum(w * u, axis=1) / w_sum
            mu_v = np.sum(w * v, axis=1) / w_sum

            du = u - mu_u[:, None]
            dv = v - mu_v[:, None]

            c00 = np.sum(w * du * du, axis=1) / w_sum
            c01 = np.sum(w * du * dv, axis=1) / w_sum
            c11 = np.sum(w * dv * dv, axis=1) / w_sum
        else:
            mu_u = np.mean(u, axis=1)
            mu_v = np.mean(v, axis=1)
            du = u - mu_u[:, None]
            dv = v - mu_v[:, None]
            denom = max(1, u.shape[1] - 1)
            c00 = np.sum(du * du, axis=1) / denom
            c01 = np.sum(du * dv, axis=1) / denom
            c11 = np.sum(dv * dv, axis=1) / denom

        C2 = np.zeros((B, 2, 2), dtype=np.float64)
        C2[:, 0, 0] = c00
        C2[:, 0, 1] = c01
        C2[:, 1, 0] = c01
        C2[:, 1, 1] = c11

        l1, l2, v1_2d, v2_2d = eig2x2_sym_batch(C2)
        l1 = np.maximum(l1, 0.0) + 1e-12
        l2 = np.maximum(l2, 0.0) + 1e-12

        # rho distances in tangent plane
        rho = np.sqrt(d2)

        # per-row percentile for dt
        if DT_USE_PARTITION:
            # rank position (0..K-2)
            m = rho.shape[1]
            pos = int(np.clip(np.round((q / 100.0) * (m - 1)), 0, m - 1))
            # partition along axis=1
            rho_part = np.partition(rho, pos, axis=1)
            d_t = rho_part[:, pos]
        else:
            d_t = np.percentile(rho, q, axis=1)

        # aspect ratio
        ar = np.sqrt(l1 / l2)
        ar = np.clip(ar, AR_MIN, AR_MAX)

        # st_base from dt
        st_base = (DT_TO_SIGMA_FACTOR * d_t).astype(np.float64)

        # light regularization by s_iso
        st_base = np.clip(st_base,
                          (TANGENT_REL_MIN * s_iso).astype(np.float64),
                          (TANGENT_REL_MAX * s_iso).astype(np.float64))

        # clamp global
        st_base = np.clip(st_base, S_MIN, S_MAX)

        s_ar = np.sqrt(ar)
        st1 = np.clip(st_base * s_ar, S_MIN, S_MAX)
        st2 = np.clip(st_base / np.maximum(s_ar, 1e-8), S_MIN, S_MAX)

        # sigma_n
        if SIGMA_N_MODE.lower() == "relative":
            sn = (SIGMA_N_REL_FACTOR * s_iso).astype(np.float64)
        else:
            sn = np.full((B,), float(SIGMA_N_FIXED), dtype=np.float64)
        sn = np.clip(sn, SIGMA_N_MIN, SIGMA_N_MAX)
        # avoid too thick vs tangent
        sn = np.minimum(sn, 0.5 * np.maximum(np.minimum(st1, st2), 1e-6))

        # map 2D eigvecs into 3D directions on tangent basis
        # v1_2d, v2_2d: (B,2)
        t_major = (v1_2d[:, 0:1] * t0 + v1_2d[:, 1:2] * t1).astype(np.float32)
        t_minor = (v2_2d[:, 0:1] * t0 + v2_2d[:, 1:2] * t1).astype(np.float32)

        # normalize + orthogonalize
        t_major = t_major / np.maximum(np.linalg.norm(t_major, axis=1, keepdims=True), 1e-8)
        # Gram-Schmidt for minor
        proj = np.sum(t_minor * t_major, axis=1, keepdims=True)
        t_minor = t_minor - proj * t_major
        t_minor = t_minor / np.maximum(np.linalg.norm(t_minor, axis=1, keepdims=True), 1e-8)

        # rotation matrix columns: [t_major, t_minor, n]
        R = np.zeros((B, 3, 3), dtype=np.float32)
        R[:, :, 0] = t_major
        R[:, :, 1] = t_minor
        R[:, :, 2] = n

        # NOTE: for speed we skip per-matrix SVD orthonormalize here; basis is already near-orthonormal.
        # If you see drift, re-enable a slower orthonormalization per block.

        sigma = np.stack([st1.astype(np.float32), st2.astype(np.float32), sn.astype(np.float32)], axis=1)
        scales_aniso[b0:b1] = sigma
        rotations[b0:b1] = R.reshape(B, 9)

        # covariance: Sigma = R diag(s^2) R^T
        s2 = (sigma.astype(np.float32) ** 2)
        # compute using einsum: (B,3,3) @ (B,3,3) @ (B,3,3)
        D = np.zeros((B, 3, 3), dtype=np.float32)
        D[:, 0, 0] = s2[:, 0]
        D[:, 1, 1] = s2[:, 1]
        D[:, 2, 2] = s2[:, 2]
        Sigma = np.einsum('bij,bjk,bkl->bil', R, D, np.transpose(R, (0, 2, 1)), optimize=True)

        # pack cov6
        cov6_block = np.zeros((B, 6), dtype=np.float32)
        cov6_block[:, 0] = Sigma[:, 0, 0]
        cov6_block[:, 1] = Sigma[:, 0, 1]
        cov6_block[:, 2] = Sigma[:, 0, 2]
        cov6_block[:, 3] = Sigma[:, 1, 1]
        cov6_block[:, 4] = Sigma[:, 1, 2]
        cov6_block[:, 5] = Sigma[:, 2, 2]
        cov6[b0:b1] = cov6_block

        if VERBOSE and (b0 > 0 and (b0 % (LOG_EVERY * 1)) == 0):
            log(f"  [BATCH-ANISO] processed {b1}/{N}")

    if VERBOSE:
        log("[INFO] Step 6 (BATCH) done.")
        log(f"[INFO] aniso sx/st1 p50={np.percentile(scales_aniso[:,0],50):.4f}, p95={np.percentile(scales_aniso[:,0],95):.4f}")
        log(f"[INFO] aniso sy/st2 p50={np.percentile(scales_aniso[:,1],50):.4f}, p95={np.percentile(scales_aniso[:,1],95):.4f}")
        log(f"[INFO] aniso sz/sn  p50={np.percentile(scales_aniso[:,2],50):.4f}, p95={np.percentile(scales_aniso[:,2],95):.4f}")

    return scales_aniso, rotations, cov6


def estimate_dt_distribution(points: np.ndarray,
                             kdtree: o3d.geometry.KDTreeFlann,
                             scales_iso: np.ndarray,
                             max_points: int = 50000,
                             knn_idx: Optional[np.ndarray] = None,
                             knn_dist2: Optional[np.ndarray] = None,
                             normals: Optional[np.ndarray] = None) -> dict:
    n_total = points.shape[0]
    m = int(min(max_points, n_total))
    if m <= 0:
        return {"count": 0}

    sel = np.random.choice(n_total, m, replace=False) if m < n_total else np.arange(n_total, dtype=np.int64)

    dt_vals = []
    fallback = 0

    log(f"[INFO] Estimating d_t distribution on {m}/{n_total} points (subset)...")

    for j, i in enumerate(sel):
        # normal (reuse Open3D normals if provided)
        if normals is not None:
            nrm = normals[int(i)].astype(np.float32)
            nn = float(np.linalg.norm(nrm))
            if nn < 1e-8:
                fallback += 1
                continue
            nrm = nrm / nn
        else:
            # PCA for normal (slower)
            if knn_idx is not None and knn_dist2 is not None:
                neigh_n = robust_knn_indices_cached(knn_idx, knn_dist2, points, int(i), K_NEIGHBORS_NORMAL, scales_iso)
            else:
                neigh_n = robust_knn_indices(points, kdtree, int(i), K_NEIGHBORS_NORMAL, scales_iso)
            if neigh_n.shape[0] < MIN_NEIGHBORS_NORMAL:
                fallback += 1
                continue

            neigh_pts = points[neigh_n, :]
            center = neigh_pts.mean(axis=0, keepdims=True)
            X = neigh_pts - center
            denom = max(1, (X.shape[0] - 1))
            C = (X.T @ X) / denom

            vals, vecs = np.linalg.eigh(C)
            order = np.argsort(vals)
            vecs = vecs[:, order]
            nrm = vecs[:, 0].astype(np.float32)
            nn = float(np.linalg.norm(nrm))
            if nn < 1e-8:
                fallback += 1
                continue
            nrm = nrm / nn

        # tangent neighbors
        if knn_idx is not None and knn_dist2 is not None:
            neigh_t = robust_knn_indices_cached(knn_idx, knn_dist2, points, int(i), K_NEIGHBORS_TANGENT, scales_iso)
        else:
            neigh_t = robust_knn_indices(points, kdtree, int(i), K_NEIGHBORS_TANGENT, scales_iso)
        if neigh_t.shape[0] < MIN_NEIGHBORS_TANGENT:
            fallback += 1
            continue

        neigh = points[neigh_t, :]
        p = points[int(i)].reshape(1, 3)
        r = neigh - p
        rn = (r @ nrm.reshape(3, 1)).reshape(-1, 1)
        r_par = r - rn * nrm.reshape(1, 3)

        t0, t1 = make_tangent_basis_from_normal(nrm)
        u = (r_par @ t0.reshape(3, 1)).reshape(-1)
        v = (r_par @ t1.reshape(3, 1)).reshape(-1)

        rho = np.sqrt(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)
        rho = rho[rho > 1e-9]
        if rho.size < 4:
            fallback += 1
            continue

        dt = float(np.percentile(rho, DT_PERCENTILE))
        dt_vals.append(dt)

        if VERBOSE and ((j + 1) % LOG_EVERY == 0):
            log(f"  [d_t] processed {j+1}/{m}... (fallback={fallback})")

    if len(dt_vals) == 0:
        return {"count": 0, "fallback": fallback}

    dt_arr = np.asarray(dt_vals, dtype=np.float64)
    stats = {
        "count": int(dt_arr.size),
        "fallback": int(fallback),
        "min": float(dt_arr.min()),
        "mean": float(dt_arr.mean()),
        "max": float(dt_arr.max()),
        "p50": float(np.percentile(dt_arr, 50)),
        "p75": float(np.percentile(dt_arr, 75)),
        "p90": float(np.percentile(dt_arr, 90)),
        "p95": float(np.percentile(dt_arr, 95)),
    }

    if VERBOSE:
        log("[INFO] d_t stats (m) on subset: "
            f"count={stats['count']}, fallback={stats['fallback']}, "
            f"p50={stats['p50']:.4f}, p90={stats['p90']:.4f}, p95={stats['p95']:.4f}, max={stats['max']:.4f}")

    return stats


def compute_normal_aligned_gaussians(points, kdtree, scales_iso,
                                     knn_idx: Optional[np.ndarray] = None,
                                     knn_dist2: Optional[np.ndarray] = None,
                                     normals: Optional[np.ndarray] = None):
    num = points.shape[0]
    scales_aniso = np.zeros((num, 3), dtype=np.float32)
    rotations = np.zeros((num, 9), dtype=np.float32)
    cov6 = np.zeros((num, 6), dtype=np.float32)

    sigma_t1_list = []
    sigma_t2_list = []
    sigma_n_list = []

    neff_list = []
    sigma_w_list = []

    dt_list = []
    ar_list = []
    st_base_list = []
    hit_smax = 0
    hit_smin = 0

    fallback_normal = 0
    fallback_tangent = 0

    log("[INFO] Step 6: Estimating normal-aligned anisotropic Gaussians...")
    log(f"[INFO] K_NEIGHBORS_NORMAL={K_NEIGHBORS_NORMAL}, K_NEIGHBORS_TANGENT={K_NEIGHBORS_TANGENT}")
    log(f"[INFO] Tangent weighting={USE_TANGENT_WEIGHTING}, sigma_factor={TANGENT_WEIGHT_SIGMA_FACTOR:.2f}")
    log(f"[INFO] Tangent regularization: [{TANGENT_REL_MIN:.2f}, {TANGENT_REL_MAX:.2f}] * s_iso")
    log(f"[INFO] st_base from dt: DT_PERCENTILE={DT_PERCENTILE}, DT_TO_SIGMA_FACTOR={DT_TO_SIGMA_FACTOR:.2f}, AR=[{AR_MIN:.1f},{AR_MAX:.1f}]")
    log(f"[INFO] Sigma_n mode={SIGMA_N_MODE}, fixed={SIGMA_N_FIXED:.3f} m")
    log(f"[INFO] Using clamp range: S_MIN={S_MIN:.4f} m, S_MAX={S_MAX:.4f} m")

    for i in range(num):
        s_iso = float(scales_iso[i])
        s_iso_clamped = float(np.clip(s_iso, S_MIN, S_MAX))

        # 1) normal
        if normals is not None:
            n = normals[i].astype(np.float32)
            n_norm = float(np.linalg.norm(n))
            if n_norm < 1e-8:
                fallback_normal += 1
                sigma = np.array([s_iso_clamped, s_iso_clamped, s_iso_clamped], dtype=np.float32)
                R = np.eye(3, dtype=np.float32)
                Sigma = np.eye(3, dtype=np.float32) * (s_iso_clamped ** 2)
                scales_aniso[i] = sigma
                rotations[i] = R.reshape(-1)
                cov6[i] = mat_to_cov6(Sigma)
                continue
            n = n / n_norm
        else:
            if knn_idx is not None and knn_dist2 is not None:
                neigh_n = robust_knn_indices_cached(knn_idx, knn_dist2, points, i, K_NEIGHBORS_NORMAL, scales_iso)
            else:
                neigh_n = robust_knn_indices(points, kdtree, i, K_NEIGHBORS_NORMAL, scales_iso)
            if neigh_n.shape[0] < MIN_NEIGHBORS_NORMAL:
                fallback_normal += 1
                sigma = np.array([s_iso_clamped, s_iso_clamped, s_iso_clamped], dtype=np.float32)
                R = np.eye(3, dtype=np.float32)
                Sigma = np.eye(3, dtype=np.float32) * (s_iso_clamped ** 2)
                scales_aniso[i] = sigma
                rotations[i] = R.reshape(-1)
                cov6[i] = mat_to_cov6(Sigma)
                continue

            neigh_pts = points[neigh_n, :]
            center = neigh_pts.mean(axis=0, keepdims=True)
            X = neigh_pts - center
            denom = max(1, (X.shape[0] - 1))
            C = (X.T @ X) / denom

            vals, vecs = np.linalg.eigh(C)
            order = np.argsort(vals)
            vecs = vecs[:, order]

            n = vecs[:, 0].astype(np.float32)
            n_norm = float(np.linalg.norm(n))
            if n_norm < 1e-8:
                fallback_normal += 1
                sigma = np.array([s_iso_clamped, s_iso_clamped, s_iso_clamped], dtype=np.float32)
                R = np.eye(3, dtype=np.float32)
                Sigma = np.eye(3, dtype=np.float32) * (s_iso_clamped ** 2)
                scales_aniso[i] = sigma
                rotations[i] = R.reshape(-1)
                cov6[i] = mat_to_cov6(Sigma)
                continue
            n = n / n_norm

        # 2) tangent neighbors
        if knn_idx is not None and knn_dist2 is not None:
            neigh_t = robust_knn_indices_cached(knn_idx, knn_dist2, points, i, K_NEIGHBORS_TANGENT, scales_iso)
        else:
            neigh_t = robust_knn_indices(points, kdtree, i, K_NEIGHBORS_TANGENT, scales_iso)

        if neigh_t.shape[0] < MIN_NEIGHBORS_TANGENT:
            fallback_tangent += 1
            sn = compute_sigma_n(s_iso)
            st = s_iso_clamped
            t0, t1 = make_tangent_basis_from_normal(n)
            R = np.stack([t0, t1, n], axis=1)
            R = orthonormalize_rotation(R)
            sigma = np.array([st, st, sn], dtype=np.float32)
            Sigma = (R @ np.diag((sigma ** 2).astype(np.float32)) @ R.T).astype(np.float32)
            scales_aniso[i] = sigma
            rotations[i] = R.reshape(-1)
            cov6[i] = mat_to_cov6(Sigma)
            sigma_t1_list.append(st)
            sigma_t2_list.append(st)
            sigma_n_list.append(sn)
            continue

        neigh = points[neigh_t, :]
        p = points[i].reshape(1, 3)
        r = neigh - p

        rn = (r @ n.reshape(3, 1)).reshape(-1, 1)
        r_par = r - rn * n.reshape(1, 3)

        t0, t1 = make_tangent_basis_from_normal(n)
        u = (r_par @ t0.reshape(3, 1)).reshape(-1)
        v = (r_par @ t1.reshape(3, 1)).reshape(-1)

        UV = np.stack([u, v], axis=1).astype(np.float64)

        # weighted cov for orientation/shape ratio
        if USE_TANGENT_WEIGHTING:
            mean_neighbor_dist = s_iso / 0.75 if s_iso > 1e-8 else 0.05
            sigma_w = TANGENT_WEIGHT_SIGMA_FACTOR * mean_neighbor_dist
            sigma_w = float(np.clip(sigma_w, TANGENT_WEIGHT_SIGMA_W_MIN, TANGENT_WEIGHT_SIGMA_W_MAX))
            d2 = (u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)
            w = np.exp(-0.5 * d2 / max(sigma_w * sigma_w, 1e-12))
            _, C2, Neff = weighted_cov_2d(UV, w)
            neff_list.append(Neff)
            sigma_w_list.append(sigma_w)
        else:
            UVc = UV - UV.mean(axis=0, keepdims=True)
            C2 = (UVc.T @ UVc) / max(1, UV.shape[0] - 1)

        l1, l2, v1_2d, v2_2d = eig2x2_sym(C2)
        l1 = max(l1, 0.0)
        l2 = max(l2, 0.0)

        rho = np.sqrt(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)
        rho = rho[rho > 1e-9]
        if rho.size < 4:
            d_t = s_iso / 0.75 if s_iso > 1e-8 else 0.05
        else:
            rho_sorted = np.sort(rho)
            d_t = fast_percentile_from_sorted(rho_sorted, float(DT_PERCENTILE))

        l1 = float(l1 + 1e-12)
        l2 = float(l2 + 1e-12)
        ar = float(np.sqrt(l1 / l2))
        ar = float(np.clip(ar, AR_MIN, AR_MAX))

        st_base = float(DT_TO_SIGMA_FACTOR * d_t)
        st_base = float(np.clip(st_base, TANGENT_REL_MIN * s_iso, TANGENT_REL_MAX * s_iso))

        if st_base >= S_MAX - 1e-12:
            hit_smax += 1
        if st_base <= S_MIN + 1e-12:
            hit_smin += 1
        st_base = float(np.clip(st_base, S_MIN, S_MAX))

        s_ar = float(np.sqrt(ar))
        st1 = float(np.clip(st_base * s_ar, S_MIN, S_MAX))
        st2 = float(np.clip(st_base / max(s_ar, 1e-8), S_MIN, S_MAX))

        sn = compute_sigma_n(s_iso)
        sn = float(min(sn, 0.5 * max(min(st1, st2), 1e-6)))

        v1 = v1_2d
        v2 = v2_2d

        t_major = v1[0] * t0 + v1[1] * t1
        t_minor = v2[0] * t0 + v2[1] * t1

        t_major = t_major / max(np.linalg.norm(t_major), 1e-8)
        t_minor = t_minor - np.dot(t_minor, t_major) * t_major
        t_minor = t_minor / max(np.linalg.norm(t_minor), 1e-8)

        R = np.stack([t_major, t_minor, n], axis=1).astype(np.float32)
        R = orthonormalize_rotation(R)

        sigma = np.array([st1, st2, sn], dtype=np.float32)
        Sigma = (R @ np.diag((sigma ** 2).astype(np.float32)) @ R.T).astype(np.float32)

        scales_aniso[i] = sigma
        rotations[i] = R.reshape(-1)
        cov6[i] = mat_to_cov6(Sigma)

        sigma_t1_list.append(st1)
        sigma_t2_list.append(st2)
        sigma_n_list.append(sn)

        dt_list.append(d_t)
        ar_list.append(ar)
        st_base_list.append(st_base)

        if VERBOSE and (i > 0 and i % LOG_EVERY == 0):
            log(f"  [N-ANISO] processed {i}/{num} "
                f"(fallback_normal={fallback_normal}, fallback_tangent={fallback_tangent}, hitSmax={hit_smax})")

    # ---- Output statistics (compact) ----
    if VERBOSE:
        log("[INFO] Step 6 done.")
        log(f"[INFO] Fallback: normal={fallback_normal}, tangent={fallback_tangent}")
        log(f"[INFO] Clamp-hit: S_MAX={hit_smax}/{num} ({100.0*hit_smax/max(1,num):.2f}%), "
            f"S_MIN={hit_smin}/{num} ({100.0*hit_smin/max(1,num):.2f}%)")

        st1_arr = np.asarray(sigma_t1_list, dtype=np.float64) if len(sigma_t1_list) else None
        st2_arr = np.asarray(sigma_t2_list, dtype=np.float64) if len(sigma_t2_list) else None
        sn_arr  = np.asarray(sigma_n_list, dtype=np.float64)  if len(sigma_n_list)  else None
        if st1_arr is not None:
            log(f"[INFO] st1(m): p50={np.percentile(st1_arr,50):.4f}, p95={np.percentile(st1_arr,95):.4f}, max={st1_arr.max():.4f}")
        if st2_arr is not None:
            log(f"[INFO] st2(m): p50={np.percentile(st2_arr,50):.4f}, p95={np.percentile(st2_arr,95):.4f}, max={st2_arr.max():.4f}")
        if sn_arr is not None:
            log(f"[INFO] sn (m): p50={np.percentile(sn_arr,50):.4f}, p95={np.percentile(sn_arr,95):.4f}, max={sn_arr.max():.4f}")

    return scales_aniso, rotations, cov6


def main():
    log(f"[INFO] Loading point cloud from: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    pcd = o3d.io.read_point_cloud(str(INPUT_PATH))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    log(f"[INFO] Loaded {points.shape[0]} points")

    # ---- Pre-shift ----
    origin_pre_shift = np.zeros((3,), dtype=np.float32)
    if ENABLE_PRE_SHIFT:
        origin_pre_shift = compute_recenter_origin(points.astype(np.float64), PRE_SHIFT_MODE).astype(np.float32)
        pcd.translate((-origin_pre_shift).astype(np.float64), relative=True)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        log("[INFO] Pre-shift enabled (float precision fix):")
        log(f"  PRE_SHIFT_MODE={PRE_SHIFT_MODE}")
        log(f"  origin_pre_shift (input frame) = [{origin_pre_shift[0]:.3f}, {origin_pre_shift[1]:.3f}, {origin_pre_shift[2]:.3f}]")

    # ---- Sampling ----
    num_points_full = points.shape[0]
    pcd_proc = sample_points_for_processing(
        pcd,
        max_points=MAX_GAUSSIANS,
        mode=SAMPLE_MODE,
        voxel_size=VOXEL_SIZE,
    )

    proc_n = np.asarray(pcd_proc.points).shape[0]
    cap_n = int(MAX_GAUSSIANS_AFTER_SAMPLING) if MAX_GAUSSIANS_AFTER_SAMPLING is not None else -1
    if cap_n > 0 and proc_n > cap_n:
        log(f"[INFO] Sampling produced {proc_n} points; capping to {cap_n} by random sampling...")
        pcd_proc = sample_points_for_processing(pcd_proc, cap_n, mode="random", voxel_size=VOXEL_SIZE)
        proc_n = np.asarray(pcd_proc.points).shape[0]

    points = np.asarray(pcd_proc.points)
    colors = np.asarray(pcd_proc.colors)
    log(f"[INFO] Sampling mode={SAMPLE_MODE}, full={num_points_full} -> processed={points.shape[0]}")

    pcd_sample = o3d.geometry.PointCloud()
    pcd_sample.points = o3d.utility.Vector3dVector(points)

    if colors.size > 0:
        if colors.max() > 1.1:
            colors = colors / 255.0
        pcd_sample.colors = o3d.utility.Vector3dVector(colors)

    normals = None
    if USE_O3D_NORMALS:
        log(f"[INFO] Estimating normals with Open3D (C++) KNN={K_NEIGHBORS_NORMAL} ...")
        pcd_sample.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=K_NEIGHBORS_NORMAL))
        pcd_sample.normalize_normals()
        normals = np.asarray(pcd_sample.normals)

    log("[INFO] Building KDTree...")
    kdtree = o3d.geometry.KDTreeFlann(pcd_sample)

    K_MAX_CACHE = int(max(K_NEIGHBORS_ISO, K_NEIGHBORS_NORMAL, K_NEIGHBORS_TANGENT) + 1)
    log(f"[INFO] Building cached batched KNN (K={K_MAX_CACHE})...")
    knn_idx_all, knn_dist2_all = batch_knn_search(points, K_MAX_CACHE)

    scales_iso = compute_isotropic_scales(points, kdtree, K_NEIGHBORS_ISO, knn_idx_all, knn_dist2_all)

    global S_MIN, S_MAX
    if USE_ADAPTIVE_CLAMP:
        S_MIN, S_MAX = compute_adaptive_clamp(scales_iso)
        log("[INFO] Adaptive clamp enabled:")
        log(f"  ADAPTIVE_Q_MIN={ADAPTIVE_Q_MIN} -> S_MIN={S_MIN:.4f} m")
        log(f"  ADAPTIVE_Q_MAX={ADAPTIVE_Q_MAX} -> S_MAX={S_MAX:.4f} m")
    else:
        log("[INFO] Adaptive clamp disabled:")
        log(f"  Using fixed S_MIN={S_MIN:.4f} m, S_MAX={S_MAX:.4f} m")

    # ---- Optional: refine S_MAX using d_t distribution ----
    s_max_siso = float(S_MAX)
    s_max_dt = None

    if USE_DT_BASED_SMAX:
        dt_stats = estimate_dt_distribution(
            points, kdtree, scales_iso,
            max_points=DT_STATS_MAX_POINTS,
            knn_idx=knn_idx_all, knn_dist2=knn_dist2_all,
            normals=normals
        )
        if dt_stats.get("count", 0) > 0:
            key = f"p{int(DT_SMAX_PERCENTILE)}"
            dt_ref = float(dt_stats.get(key, dt_stats.get("p90", dt_stats.get("p75", dt_stats.get("p50", 0.0)))))
            s_max_dt = float(np.clip(DT_SMAX_FACTOR * dt_ref, ABS_S_MIN, ABS_S_MAX))
            log("[INFO] d_t-based S_MAX estimation:")
            log(f"  DT_SMAX_PERCENTILE={DT_SMAX_PERCENTILE} -> d_t_ref={dt_ref:.4f} m")
            log(f"  DT_SMAX_FACTOR={DT_SMAX_FACTOR:.2f} -> S_MAX_dt={s_max_dt:.4f} m")

    policy = (S_MAX_POLICY or "max").lower()
    if policy == "dt" and s_max_dt is not None:
        S_MAX = float(s_max_dt)
        log(f"[INFO] S_MAX_POLICY=dt -> S_MAX={S_MAX:.4f} m")
    elif policy == "siso" or s_max_dt is None:
        S_MAX = float(s_max_siso)
        log(f"[INFO] S_MAX_POLICY=siso -> S_MAX={S_MAX:.4f} m")
    else:
        S_MAX = float(max(s_max_siso, s_max_dt))
        S_MAX = float(np.clip(S_MAX, ABS_S_MIN, ABS_S_MAX))
        log(f"[INFO] S_MAX_POLICY=max -> S_MAX={S_MAX:.4f} m")

    if S_MAX <= S_MIN:
        S_MAX = float(min(ABS_S_MAX, S_MIN * 2.0))
        log(f"[WARN] Adjusted S_MAX to keep it > S_MIN: S_MAX={S_MAX:.4f} m")

    if USE_BATCH_ANISO and _HAS_SCIPY:
        if normals is None:
            raise RuntimeError("USE_BATCH_ANISO requires normals. Enable USE_O3D_NORMALS=True.")
        scales_aniso, rotations, cov6 = compute_normal_aligned_gaussians_batched(
            points=points,
            scales_iso=scales_iso,
            normals=normals,
            k_tangent=K_NEIGHBORS_TANGENT,
            block_size=ANISO_BLOCK_SIZE,
        )
    else:
        scales_aniso, rotations, cov6 = compute_normal_aligned_gaussians(
            points, kdtree, scales_iso,
            knn_idx_all, knn_dist2_all,
            normals=normals
        )

    if APPLY_UNITY_FRAME:
        log("[INFO] Applying ENU -> Unity frame transform (Z=North, Y=Up)...")
        points, rotations, cov6 = apply_frame_transform(points, rotations, cov6, M_ENU_TO_UNITY)

    origin = np.zeros((3,), dtype=np.float32)
    if APPLY_RECENTER:
        origin = compute_recenter_origin(points, RECENTER_MODE)
        points = apply_recenter(points, origin)
        log("[INFO] Recentering enabled:")
        log(f"  RECENTER_MODE={RECENTER_MODE}")
        log(f"  origin (Unity frame) = [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]")

    if VERBOSE:
        log("[INFO] Covariance diag stats (m^2): "
            f"xx[{cov6[:,0].min():.3e},{cov6[:,0].max():.3e}] "
            f"yy[{cov6[:,3].min():.3e},{cov6[:,3].max():.3e}] "
            f"zz[{cov6[:,5].min():.3e},{cov6[:,5].max():.3e}]")

    opacity = np.ones((points.shape[0],), dtype=np.float32)
    if VERBOSE:
        log("[INFO] XYZ range (Unity frame): "
            f"X[{points[:,0].min():.2f}, {points[:,0].max():.2f}], "
            f"Y[{points[:,1].min():.2f}, {points[:,1].max():.2f}], "
            f"Z[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
        if colors.size > 0:
            log("[INFO] Colors range: "
                f"R[{colors[:,0].min():.2f}, {colors[:,0].max():.2f}], "
                f"G[{colors[:,1].min():.2f}, {colors[:,1].max():.2f}], "
                f"B[{colors[:,2].min():.2f}, {colors[:,2].max():.2f}]")

    cov0 = np.zeros((points.shape[0], 4), dtype=np.float32)
    cov1 = np.zeros((points.shape[0], 4), dtype=np.float32)
    cov0[:, 0] = cov6[:, 0]
    cov0[:, 1] = cov6[:, 1]
    cov0[:, 2] = cov6[:, 2]
    cov0[:, 3] = cov6[:, 3]
    cov1[:, 0] = cov6[:, 4]
    cov1[:, 1] = cov6[:, 5]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    out_npz = OUTPUT_PATH
    if PRESET in ("verify", "final"):
        out_npz = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_{PRESET}{OUTPUT_PATH.suffix}")

    np.savez(
        out_npz,
        positions=points.astype(np.float32),
        origin=origin.astype(np.float32),
        origin_pre_shift=origin_pre_shift.astype(np.float32),
        scales=scales_aniso.astype(np.float32),
        scales_iso=scales_iso.astype(np.float32),
        rotations=rotations.astype(np.float32),
        cov6=cov6.astype(np.float32),
        cov0=cov0,
        cov1=cov1,
        colors=colors.astype(np.float32),
        opacity=opacity.astype(np.float32),
    )
    log(f"[INFO] Saved Gaussians to: {out_npz}")

    txt_path = out_npz.with_suffix(".txt")
    log(f"[INFO] Also writing simple text format to: {txt_path}")

    if colors.size == 0:
        colors_txt = np.tile(np.array([[0.7, 0.7, 0.7]], dtype=np.float32), (points.shape[0], 1))
    else:
        colors_txt = colors

    data_mat = np.hstack([
        points.astype(np.float32),
        colors_txt.astype(np.float32),
        cov6.astype(np.float32)
    ])
    np.savetxt(txt_path, data_mat, fmt="%.6f")

    origin_path = txt_path.with_suffix(".origin.txt")
    np.savetxt(origin_path, np.vstack([origin_pre_shift.reshape(1, 3), origin.reshape(1, 3)]), fmt="%.6f")
    log(f"[INFO] Saved origin sidecar to: {origin_path}")

    log("[INFO] Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=PRESET_DEFAULT, choices=["verify", "final"])
    ap.add_argument("--quiet", action="store_true", help="Disable most logs for speed")
    ap.add_argument("--log_every", type=int, default=LOG_EVERY, help="Progress log interval")
    ap.add_argument("--no_batch_aniso", action="store_true", help="Disable SciPy batched anisotropy")
    ap.add_argument("--aniso_block", type=int, default=ANISO_BLOCK_SIZE, help="Batch size for anisotropy")
    args = ap.parse_args()

    if args.quiet:
        VERBOSE = False
    LOG_EVERY = int(max(1, args.log_every))

    if args.no_batch_aniso:
        USE_BATCH_ANISO = False
    ANISO_BLOCK_SIZE = int(max(1000, args.aniso_block))

    apply_preset(args.preset)
    main()