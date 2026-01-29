import numpy as np
import open3d as o3d
from pathlib import Path
import argparse
import json
from typing import Optional

# Optional (fast CPU KNN)
try:
    from scipy.spatial import cKDTree  # type: ignore
    _HAS_SCIPY = True
except Exception:
    cKDTree = None
    _HAS_SCIPY = False

# ==============================
# Paths
# ==============================
INPUT_PATH = Path("data/TumTLS_v2.ply")
OUTPUT_PATH = Path("data/TumTLS_v2_gaussians_demo.npz")

# ==============================
# Preset switch
# ==============================
PRESET_DEFAULT = "verify"
PRESET = PRESET_DEFAULT  # overwritten by apply_preset()

# ==============================
# Logging
# ==============================
VERBOSE = True
LOG_EVERY = 50000

def log(msg: str):
    if VERBOSE:
        print(msg)

# ==============================
# Sampling strategy (ONLY random downsample)
# ==============================
SAMPLE_MODE = "random"   # enforced random
MAX_GAUSSIANS = 4_000_000
MAX_GAUSSIANS_AFTER_SAMPLING = MAX_GAUSSIANS

# KNN
K_NEIGHBORS_ISO = 8
K_NEIGHBORS_NORMAL = 24
K_NEIGHBORS_TANGENT = 64

# ==============================
# Normals strategy (B): prefer file normals
# ==============================
USE_FILE_NORMALS_FIRST = True     # ← 핵심：优先用文件 normals
ALLOW_O3D_ESTIMATE_NORMALS = True # 文件没有 normals 才允许算
# 注意：你现在 PLY 有 normals，因此不会走 estimate_normals()

# ==============================
# Batch anisotropy (million+ points)
# ==============================
USE_BATCH_ANISO = True
ANISO_BLOCK_SIZE = 50000
DT_USE_PARTITION = True

MIN_NEIGHBORS_NORMAL = 10
MIN_NEIGHBORS_TANGENT = 16
KNN_DISTANCE_CLIP_ALPHA = 2.5
RADIUS_FACTOR_FROM_ISO = 6.0

# --- normal thickness ---
SIGMA_N_MODE = "fixed"   # "fixed" | "relative"
SIGMA_N_FIXED = 0.008
SIGMA_N_REL_FACTOR = 0.25
SIGMA_N_MIN = 0.004
SIGMA_N_MAX = 0.015

# --- tangent weighting ---
USE_TANGENT_WEIGHTING = True
TANGENT_WEIGHT_SIGMA_FACTOR = 1.6
TANGENT_WEIGHT_SIGMA_W_MIN = 0.02
TANGENT_WEIGHT_SIGMA_W_MAX = 0.25

# --- tangent regularization ---
TANGENT_REL_MIN = 0.35
TANGENT_REL_MAX = 1.1

# --- dt -> st_base ---
DT_PERCENTILE = 20
DT_TO_SIGMA_FACTOR = 0.75

# aspect ratio clamp
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

# ==============================
# Frame transform (ENU -> Unity)
# ==============================
APPLY_UNITY_FRAME = True
M_ENU_TO_UNITY = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0],
    [ 0.0,  1.0,  0.0],
], dtype=np.float32)

# ==============================
# Recenter
# ==============================
APPLY_RECENTER = True
RECENTER_MODE = "bbox_center"

# ==============================
# Pre-shift large coordinates (UTM float precision fix)
# ==============================
ENABLE_PRE_SHIFT = True
PRE_SHIFT_MODE = "bbox_center"

# ==============================
# Chunk writing (integrated)
# ==============================
WRITE_CHUNKS = False
CHUNK_DIR = Path("data/chunks_TUMv2")
CHUNK_PREFIX = "chunk"
CHUNK_SIZE = np.array([10.0, 10.0, 10.0], dtype=np.float32)
CHUNK_EPS = 1e-5


def apply_preset(preset: str):
    """verify/final 现在只负责 max_points 与一些参数，不再改 sampling 模式（永远 random）"""
    global PRESET
    global MAX_GAUSSIANS, MAX_GAUSSIANS_AFTER_SAMPLING
    global TANGENT_WEIGHT_SIGMA_FACTOR, TANGENT_WEIGHT_SIGMA_W_MAX
    global DT_PERCENTILE, DT_TO_SIGMA_FACTOR
    global TANGENT_REL_MAX
    global S_MAX_POLICY, DT_SMAX_PERCENTILE, DT_SMAX_FACTOR
    global SIGMA_N_FIXED

    p = (preset or PRESET_DEFAULT).lower().strip()
    PRESET = p

    if p == "verify":
        # 小数据验证：更保守的参数
        MAX_GAUSSIANS = 4_433_050  # 你之前看到的 N=4433050 就是它
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
        # 全量：默认不限制点数（-1=all），但你也可以用 --max_points 覆盖
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
    log(f"[INFO] Sampling mode=random, max_points={MAX_GAUSSIANS}, cap_after={MAX_GAUSSIANS_AFTER_SAMPLING}")
    log(f"[INFO] dt->st: DT_PERCENTILE={DT_PERCENTILE}, DT_TO_SIGMA_FACTOR={DT_TO_SIGMA_FACTOR:.2f}; "
        f"tangent sigma_factor={TANGENT_WEIGHT_SIGMA_FACTOR:.2f}, w_max={TANGENT_WEIGHT_SIGMA_W_MAX:.2f}")
    log(f"[INFO] S_MAX_POLICY={S_MAX_POLICY}, SIGMA_N_FIXED={SIGMA_N_FIXED:.3f}")


# -------------------- helpers --------------------
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

def orthonormalize_rotation(R):
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1.0
        R_ortho = U @ Vt
    return R_ortho.astype(np.float32)

def cov6_to_mat(cov6_row: np.ndarray) -> np.ndarray:
    xx, xy, xz, yy, yz, zz = cov6_row
    return np.array([[xx, xy, xz],
                     [xy, yy, yz],
                     [xz, yz, zz]], dtype=np.float32)

def mat_to_cov6(Sigma: np.ndarray) -> np.ndarray:
    return np.array([Sigma[0, 0], Sigma[0, 1], Sigma[0, 2],
                     Sigma[1, 1], Sigma[1, 2], Sigma[2, 2]], dtype=np.float32)

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


# ---- Random sampling that preserves normals/colors by select_by_index ----
def random_downsample_pcd(pcd: o3d.geometry.PointCloud, max_points: int) -> o3d.geometry.PointCloud:
    n = np.asarray(pcd.points).shape[0]
    if max_points is None or max_points <= 0 or n <= max_points:
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


def compute_isotropic_scales(points, kdtree, k_neighbors,
                            knn_idx: Optional[np.ndarray] = None,
                            knn_dist2: Optional[np.ndarray] = None):
    num = points.shape[0]
    scales_iso = np.zeros((num,), dtype=np.float32)

    log("[INFO] Estimating isotropic scales from neighbor distances...")

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

    # slow fallback
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


def make_tangent_basis_from_normals_batch(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = n.astype(np.float32, copy=False)
    B = n.shape[0]

    a = np.zeros((B, 3), dtype=np.float32)
    use_z = (np.abs(n[:, 2]) < 0.9)
    a[use_z, 2] = 1.0
    a[~use_z, 1] = 1.0

    t0 = np.cross(a, n)
    t0n = np.linalg.norm(t0, axis=1, keepdims=True)

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


def eig2x2_sym_batch(C2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = C2[:, 0, 0].astype(np.float64, copy=False)
    b = C2[:, 0, 1].astype(np.float64, copy=False)
    c = C2[:, 1, 1].astype(np.float64, copy=False)

    tr = a + c
    det = a * c - b * b
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    s = np.sqrt(disc)
    l1 = 0.5 * (tr + s)
    l2 = 0.5 * (tr - s)

    v1 = np.zeros((C2.shape[0], 2), dtype=np.float64)
    use_b = (np.abs(b) > 1e-18)
    v1[use_b, 0] = (l1[use_b] - c[use_b])
    v1[use_b, 1] = b[use_b]
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
    if not _HAS_SCIPY:
        raise RuntimeError("SciPy not available (cKDTree missing). Set --no_batch_aniso or install scipy.")

    pts = np.asarray(points, dtype=np.float32)
    N = pts.shape[0]
    k = int(k_tangent)
    if k <= 2:
        raise ValueError("k_tangent must be >= 3")

    nrm = np.asarray(normals, dtype=np.float32)
    nn = np.linalg.norm(nrm, axis=1)
    good = nn > 1e-8
    nrm = nrm.copy()
    nrm[good] /= nn[good].reshape(-1, 1)
    nrm[~good] = np.array([0, 0, 1], dtype=np.float32)

    scales_aniso = np.zeros((N, 3), dtype=np.float32)
    rotations = np.zeros((N, 9), dtype=np.float32)
    cov6 = np.zeros((N, 6), dtype=np.float32)

    tree = cKDTree(pts)

    log("[INFO] Step 6 (BATCH): cKDTree KNN + numpy broadcasting...")
    log(f"[INFO] N={N}, k_tangent={k}, block_size={block_size}")

    q = float(DT_PERCENTILE)

    for b0 in range(0, N, int(block_size)):
        b1 = min(N, b0 + int(block_size))
        B = b1 - b0

        p = pts[b0:b1]
        n = nrm[b0:b1]
        s_iso = scales_iso[b0:b1].astype(np.float32, copy=False)

        d, idx = tree.query(p, k=k, workers=-1)
        idx = idx.astype(np.int64, copy=False)

        idx_n = idx[:, 1:]
        neigh = pts[idx_n]
        r = neigh - p[:, None, :]

        rn = np.einsum('bkc,bc->bk', r, n, optimize=True)
        r_par = r - rn[:, :, None] * n[:, None, :]

        t0, t1 = make_tangent_basis_from_normals_batch(n)

        u = np.einsum('bkc,bc->bk', r_par, t0, optimize=True).astype(np.float64, copy=False)
        v = np.einsum('bkc,bc->bk', r_par, t1, optimize=True).astype(np.float64, copy=False)

        d2 = u * u + v * v

        if USE_TANGENT_WEIGHTING:
            mean_neighbor_dist = (s_iso / 0.75).astype(np.float64)
            sigma_w = (TANGENT_WEIGHT_SIGMA_FACTOR * mean_neighbor_dist)
            sigma_w = np.clip(sigma_w, TANGENT_WEIGHT_SIGMA_W_MIN, TANGENT_WEIGHT_SIGMA_W_MAX)
            w = np.exp(-0.5 * d2 / np.maximum(sigma_w[:, None] * sigma_w[:, None], 1e-12))
            w_sum = np.maximum(np.sum(w, axis=1), 1e-12)

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

        rho = np.sqrt(d2)

        if DT_USE_PARTITION:
            m = rho.shape[1]
            pos = int(np.clip(np.round((q / 100.0) * (m - 1)), 0, m - 1))
            rho_part = np.partition(rho, pos, axis=1)
            d_t = rho_part[:, pos]
        else:
            d_t = np.percentile(rho, q, axis=1)

        ar = np.sqrt(l1 / l2)
        ar = np.clip(ar, AR_MIN, AR_MAX)

        st_base = (DT_TO_SIGMA_FACTOR * d_t).astype(np.float64)
        st_base = np.clip(
            st_base,
            (TANGENT_REL_MIN * s_iso).astype(np.float64),
            (TANGENT_REL_MAX * s_iso).astype(np.float64),
        )
        st_base = np.clip(st_base, S_MIN, S_MAX)

        s_ar = np.sqrt(ar)
        st1 = np.clip(st_base * s_ar, S_MIN, S_MAX)
        st2 = np.clip(st_base / np.maximum(s_ar, 1e-8), S_MIN, S_MAX)

        if SIGMA_N_MODE.lower() == "relative":
            sn = (SIGMA_N_REL_FACTOR * s_iso).astype(np.float64)
        else:
            sn = np.full((B,), float(SIGMA_N_FIXED), dtype=np.float64)
        sn = np.clip(sn, SIGMA_N_MIN, SIGMA_N_MAX)
        sn = np.minimum(sn, 0.5 * np.maximum(np.minimum(st1, st2), 1e-6))

        t_major = (v1_2d[:, 0:1] * t0 + v1_2d[:, 1:2] * t1).astype(np.float32)
        t_minor = (v2_2d[:, 0:1] * t0 + v2_2d[:, 1:2] * t1).astype(np.float32)

        t_major = t_major / np.maximum(np.linalg.norm(t_major, axis=1, keepdims=True), 1e-8)
        proj = np.sum(t_minor * t_major, axis=1, keepdims=True)
        t_minor = t_minor - proj * t_major
        t_minor = t_minor / np.maximum(np.linalg.norm(t_minor, axis=1, keepdims=True), 1e-8)

        R = np.zeros((B, 3, 3), dtype=np.float32)
        R[:, :, 0] = t_major
        R[:, :, 1] = t_minor
        R[:, :, 2] = n

        sigma = np.stack([st1.astype(np.float32), st2.astype(np.float32), sn.astype(np.float32)], axis=1)
        scales_aniso[b0:b1] = sigma
        rotations[b0:b1] = R.reshape(B, 9)

        s2 = (sigma.astype(np.float32) ** 2)
        D = np.zeros((B, 3, 3), dtype=np.float32)
        D[:, 0, 0] = s2[:, 0]
        D[:, 1, 1] = s2[:, 1]
        D[:, 2, 2] = s2[:, 2]
        Sigma = np.einsum('bij,bjk,bkl->bil', R, D, np.transpose(R, (0, 2, 1)), optimize=True)

        cov6_block = np.zeros((B, 6), dtype=np.float32)
        cov6_block[:, 0] = Sigma[:, 0, 0]
        cov6_block[:, 1] = Sigma[:, 0, 1]
        cov6_block[:, 2] = Sigma[:, 0, 2]
        cov6_block[:, 3] = Sigma[:, 1, 1]
        cov6_block[:, 4] = Sigma[:, 1, 2]
        cov6_block[:, 5] = Sigma[:, 2, 2]
        cov6[b0:b1] = cov6_block

        if VERBOSE and (b0 > 0 and (b0 % LOG_EVERY == 0)):
            log(f"  [BATCH-ANISO] processed {b1}/{N}")

    if VERBOSE:
        log("[INFO] Step 6 (BATCH) done.")
        log(f"[INFO] aniso st1 p50={np.percentile(scales_aniso[:,0],50):.4f}, p95={np.percentile(scales_aniso[:,0],95):.4f}")
        log(f"[INFO] aniso st2 p50={np.percentile(scales_aniso[:,1],50):.4f}, p95={np.percentile(scales_aniso[:,1],95):.4f}")
        log(f"[INFO] aniso sn  p50={np.percentile(scales_aniso[:,2],50):.4f}, p95={np.percentile(scales_aniso[:,2],95):.4f}")

    return scales_aniso, rotations, cov6


def write_chunks_from_arrays(P: np.ndarray,
                             C: np.ndarray,
                             cov6: np.ndarray,
                             chunk_dir: Path,
                             chunk_prefix: str,
                             chunk_size: np.ndarray,
                             eps: float = 1e-5):
    chunk_dir.mkdir(parents=True, exist_ok=True)

    N = P.shape[0]
    xyz_min = P.min(axis=0)
    xyz_max = P.max(axis=0)

    origin = xyz_min.copy()
    extent = xyz_max - xyz_min
    grid = np.ceil((extent + eps) / chunk_size).astype(int)
    nx, ny, nz = map(int, grid.tolist())

    rel = P - origin[None, :]
    ijk = np.floor(rel / chunk_size[None, :]).astype(int)
    ijk[:, 0] = np.clip(ijk[:, 0], 0, nx - 1)
    ijk[:, 1] = np.clip(ijk[:, 1], 0, ny - 1)
    ijk[:, 2] = np.clip(ijk[:, 2], 0, nz - 1)

    chunk_dict: dict[tuple[int,int,int], list[int]] = {}
    for i in range(N):
        key = (int(ijk[i, 0]), int(ijk[i, 1]), int(ijk[i, 2]))
        chunk_dict.setdefault(key, []).append(i)

    chunk_meta = []
    for (ix, iy, iz), idx_list in chunk_dict.items():
        idx_arr = np.asarray(idx_list, dtype=np.int64)
        P_chunk = P[idx_arr]
        C_chunk = C[idx_arr] if C.size else np.tile(np.array([[0.7,0.7,0.7]], dtype=np.float32), (P_chunk.shape[0],1))
        cov6_chunk = cov6[idx_arr]

        mat = np.hstack([P_chunk, C_chunk, cov6_chunk])
        fname = f"{chunk_prefix}_{ix}_{iy}_{iz}.txt"
        fpath = chunk_dir / fname
        np.savetxt(fpath, mat, fmt="%.6f")

        bbox_min = P_chunk.min(axis=0)
        bbox_max = P_chunk.max(axis=0)
        center = 0.5 * (bbox_min + bbox_max)

        chunk_meta.append({
            "ijk": [ix, iy, iz],
            "filename": fname,
            "count": int(P_chunk.shape[0]),
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "center": center.tolist(),
        })

    index = {
        "origin": origin.tolist(),
        "chunk_size": chunk_size.tolist(),
        "grid_shape": [nx, ny, nz],
        "num_points": int(N),
        "num_chunks": int(len(chunk_meta)),
        "chunks": chunk_meta,
    }

    index_path = chunk_dir / "chunks_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    log(f"[INFO] Wrote {len(chunk_meta)} chunk files to: {chunk_dir}")
    log(f"[INFO] Wrote chunk index JSON: {index_path}")


def main():
    log(f"[INFO] Loading point cloud from: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    pcd = o3d.io.read_point_cloud(str(INPUT_PATH))
    if pcd.is_empty():
        raise RuntimeError("Open3D loaded empty point cloud. Check file path/format.")

    log(f"[INFO] Loaded {np.asarray(pcd.points).shape[0]} points")

    # ---- Pre-shift ----
    origin_pre_shift = np.zeros((3,), dtype=np.float32)
    if ENABLE_PRE_SHIFT:
        pts0 = np.asarray(pcd.points).astype(np.float64)
        origin_pre_shift = compute_recenter_origin(pts0, PRE_SHIFT_MODE).astype(np.float32)
        pcd.translate((-origin_pre_shift).astype(np.float64), relative=True)
        log("[INFO] Pre-shift enabled (float precision fix):")
        log(f"  PRE_SHIFT_MODE={PRE_SHIFT_MODE}")
        log(f"  origin_pre_shift (input frame) = [{origin_pre_shift[0]:.3f}, {origin_pre_shift[1]:.3f}, {origin_pre_shift[2]:.3f}]")

    # ---- Random downsample (preserve normals/colors by select_by_index) ----
    num_points_full = np.asarray(pcd.points).shape[0]
    pcd_proc = random_downsample_pcd(pcd, MAX_GAUSSIANS)
    proc_n = np.asarray(pcd_proc.points).shape[0]

    cap_n = int(MAX_GAUSSIANS_AFTER_SAMPLING) if MAX_GAUSSIANS_AFTER_SAMPLING is not None else -1
    if cap_n > 0 and proc_n > cap_n:
        log(f"[INFO] Random sampling produced {proc_n} points; capping to {cap_n}...")
        pcd_proc = random_downsample_pcd(pcd_proc, cap_n)
        proc_n = np.asarray(pcd_proc.points).shape[0]

    log(f"[INFO] Sampling mode=random, full={num_points_full} -> processed={proc_n}")

    # ---- Extract arrays ----
    points = np.asarray(pcd_proc.points).astype(np.float32, copy=False)
    colors = np.asarray(pcd_proc.colors).astype(np.float32, copy=False)

    if colors.size > 0 and colors.max() > 1.1:
        colors = colors / 255.0

    # ---- Normals: prefer file normals (B) ----
    normals = None
    if USE_FILE_NORMALS_FIRST and pcd_proc.has_normals():
        normals = np.asarray(pcd_proc.normals).astype(np.float32, copy=False)
        log("[INFO] Using normals from input PLY (no estimation).")
    elif ALLOW_O3D_ESTIMATE_NORMALS:
        log(f"[INFO] Estimating normals with Open3D (C++) KNN={K_NEIGHBORS_NORMAL} ...")
        pcd_proc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=K_NEIGHBORS_NORMAL))
        pcd_proc.normalize_normals()
        normals = np.asarray(pcd_proc.normals).astype(np.float32, copy=False)
    else:
        raise RuntimeError("No normals in file and estimation disabled.")

    # ---- KDTree for fallback pieces + cached KNN ----
    log("[INFO] Building KDTree...")
    kdtree = o3d.geometry.KDTreeFlann(pcd_proc)

    K_MAX_CACHE = int(max(K_NEIGHBORS_ISO, K_NEIGHBORS_NORMAL, K_NEIGHBORS_TANGENT) + 1)
    log(f"[INFO] Building cached batched KNN (K={K_MAX_CACHE})...")
    knn_idx_all, knn_dist2_all = batch_knn_search(points, K_MAX_CACHE)

    scales_iso = compute_isotropic_scales(points, kdtree, K_NEIGHBORS_ISO, knn_idx_all, knn_dist2_all)

    # ---- Adaptive clamp ----
    global S_MIN, S_MAX
    if USE_ADAPTIVE_CLAMP:
        S_MIN, S_MAX = compute_adaptive_clamp(scales_iso)
        log("[INFO] Adaptive clamp enabled:")
        log(f"  ADAPTIVE_Q_MIN={ADAPTIVE_Q_MIN} -> S_MIN={S_MIN:.4f} m")
        log(f"  ADAPTIVE_Q_MAX={ADAPTIVE_Q_MAX} -> S_MAX={S_MAX:.4f} m")
    else:
        log("[INFO] Adaptive clamp disabled:")
        log(f"  Using fixed S_MIN={S_MIN:.4f} m, S_MAX={S_MAX:.4f} m")

    # ---- S_MAX policy (keep your previous logic, simplified) ----
    # 这里保留你的策略结构，但不再重复 estimate_dt_distribution（你可之后再加回去）
    if S_MAX <= S_MIN:
        S_MAX = float(min(ABS_S_MAX, S_MIN * 2.0))
        log(f"[WARN] Adjusted S_MAX to keep it > S_MIN: S_MAX={S_MAX:.4f} m")

    # ---- Step 6 batched aniso ----
    if USE_BATCH_ANISO and _HAS_SCIPY:
        scales_aniso, rotations, cov6 = compute_normal_aligned_gaussians_batched(
            points=points,
            scales_iso=scales_iso,
            normals=normals,
            k_tangent=K_NEIGHBORS_TANGENT,
            block_size=ANISO_BLOCK_SIZE,
        )
    else:
        raise RuntimeError("This script version expects SciPy batched anisotropy. Install scipy or extend fallback path.")

    # ---- Frame transform ----
    if APPLY_UNITY_FRAME:
        log("[INFO] Applying ENU -> Unity frame transform (Z=North, Y=Up)...")
        points, rotations, cov6 = apply_frame_transform(points, rotations, cov6, M_ENU_TO_UNITY)

    # ---- Recenter ----
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

    # ---- Optional: write chunks (integrated) ----
    if WRITE_CHUNKS:
        log("[INFO] Writing chunks...")
        write_chunks_from_arrays(
            P=points.astype(np.float32),
            C=colors.astype(np.float32),
            cov6=cov6.astype(np.float32),
            chunk_dir=CHUNK_DIR,
            chunk_prefix=CHUNK_PREFIX,
            chunk_size=CHUNK_SIZE.astype(np.float32),
            eps=CHUNK_EPS,
        )

    log("[INFO] Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=PRESET_DEFAULT, choices=["verify", "final"])
    ap.add_argument("--quiet", action="store_true", help="Disable most logs for speed")
    ap.add_argument("--log_every", type=int, default=LOG_EVERY, help="Progress log interval")

    ap.add_argument("--no_batch_aniso", action="store_true", help="Disable SciPy batched anisotropy")
    ap.add_argument("--aniso_block", type=int, default=ANISO_BLOCK_SIZE, help="Batch size for anisotropy")

    # NEW: override max points
    ap.add_argument("--max_points", type=int, default=None, help="Override max points for random downsample (-1=all)")

    # NEW: chunk options
    ap.add_argument("--write_chunks", action="store_true", help="Write chunk txt files + chunks_index.json")
    ap.add_argument("--chunk_dir", type=str, default=str(CHUNK_DIR), help="Chunk output directory")
    ap.add_argument("--chunk_prefix", type=str, default=CHUNK_PREFIX, help="Chunk filename prefix")
    ap.add_argument("--chunk_size", type=float, nargs=3, default=CHUNK_SIZE.tolist(), help="Chunk size dx dy dz")

    args = ap.parse_args()

    if args.quiet:
        VERBOSE = False
    LOG_EVERY = int(max(1, args.log_every))

    if args.no_batch_aniso:
        USE_BATCH_ANISO = False
    ANISO_BLOCK_SIZE = int(max(1000, args.aniso_block))

    apply_preset(args.preset)

    if args.max_points is not None:
        MAX_GAUSSIANS = int(args.max_points)
        MAX_GAUSSIANS_AFTER_SAMPLING = MAX_GAUSSIANS
        log(f"[INFO] Overriding max_points -> {MAX_GAUSSIANS}")

    WRITE_CHUNKS = bool(args.write_chunks)
    CHUNK_DIR = Path(args.chunk_dir)
    CHUNK_PREFIX = str(args.chunk_prefix)
    CHUNK_SIZE = np.array(args.chunk_size, dtype=np.float32)

    main()
