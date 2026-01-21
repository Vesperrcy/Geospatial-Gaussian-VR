import numpy as np
import open3d as o3d
from pathlib import Path
import argparse

# ==== 配置 ====

INPUT_PATH = Path("data/TumTLS_v2.ply")
OUTPUT_PATH = Path("data/TumTLS_v2_gaussians_demo.npz")

# ==============================
# Preset switch
# ==============================
# verify: 小数据验证（清晰优先，允许洞）:
#python preprocessing/gaussian_builder.py --preset verify
# final : 高密度/全量（连续优先，可更“面”）
#python preprocessing/gaussian_builder.py --preset final
PRESET_DEFAULT = "verify"

# This is overwritten by apply_preset()
PRESET = PRESET_DEFAULT

# ==============================
# Sampling strategy (for large TLS clouds)
# ==============================
# Random sampling is quick but can distort local density; voxel sampling is more stable.
SAMPLE_MODE = "random"   # "random" | "voxel" | "all" (overwritten by preset)
VOXEL_SIZE = 0.02        # meters; 0.01~0.05 typical for TLS depending on desired density (overwritten by preset)

# Cap final gaussian count after sampling (<=0 means no cap)
# For random mode this also acts as the sample count.
MAX_GAUSSIANS = 400000
MAX_GAUSSIANS_AFTER_SAMPLING = MAX_GAUSSIANS

K_NEIGHBORS_ISO = 8

# ---- Step 6: 法线对齐各向异性（Normal-aligned anisotropy） ----
K_NEIGHBORS_NORMAL = 24
K_NEIGHBORS_TANGENT = 64

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
# 权重核宽度：sigma_w = factor * mean_neighbor_dist (≈ s_iso/0.75)
TANGENT_WEIGHT_SIGMA_FACTOR = 1.6
TANGENT_WEIGHT_SIGMA_W_MIN = 0.02
TANGENT_WEIGHT_SIGMA_W_MAX = 0.25   # ✅ 收紧（原来 0.50 容易把权重“抹平”）

# --- 切向 regularization：收紧范围，避免雾化 ---
TANGENT_REL_MIN = 0.35
TANGENT_REL_MAX = 1.1

# ---- NEW: 贴片绝对大小用“局部点间距”控制（避免 st 顶到 S_MAX 变雾） ----
# d_t = percentile(rho, DT_PERCENTILE)   (rho=sqrt(u^2+v^2) on tangent plane)
DT_PERCENTILE = 20

# st_base = clamp(DT_TO_SIGMA_FACTOR * d_t, S_MIN, S_MAX)
DT_TO_SIGMA_FACTOR = 0.75   # 0.6~1.0 可调：更大=更连续更糊；更小=更锐更易洞

# PCA 仅用于控制形状比（aspect ratio），而不是绝对大小
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

# ---- S_MAX policy (TLS-friendly) ----
# Scheme 1: S_MAX from s_iso distribution (fast, robust)
# Scheme 2: S_MAX from tangent-plane distance d_t distribution (more physical for surface splats)
# Final S_MAX can be chosen by policy.
S_MAX_POLICY = "max"   # "siso" | "dt" | "max"

# d_t-based S_MAX estimation (computed on a subset for speed)
USE_DT_BASED_SMAX = True
DT_SMAX_PERCENTILE = 90       # use dt percentile as a robust upper envelope
DT_SMAX_FACTOR = 0.75         # S_MAX_dt = DT_SMAX_FACTOR * percentile(d_t)
DT_STATS_MAX_POINTS = 50000   # compute dt stats on at most this many points

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
# If your input is in large world coordinates (e.g., UTM ~ 1e6), single-precision rendering in Unity
# can show banding/striping due to float quantization. Pre-shift moves the cloud near the origin
# BEFORE any float32 export.
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
        # ---- Preset-Verify: 小数据验证（清晰优先） ----
        # 推荐 voxel：比 random 更能保留局部密度结构
        SAMPLE_MODE = "voxel"
        VOXEL_SIZE = 0.05
        MAX_GAUSSIANS = 100000
        MAX_GAUSSIANS_AFTER_SAMPLING = MAX_GAUSSIANS

        # 切向更局部，避免稀疏被抹成雾
        TANGENT_WEIGHT_SIGMA_FACTOR = 1.0
        TANGENT_WEIGHT_SIGMA_W_MAX = 0.14

        # dt -> st 更保守（关键：清晰）
        DT_PERCENTILE = 10
        DT_TO_SIGMA_FACTOR = 0.35

        # 相对上限收紧
        TANGENT_REL_MAX = 0.80

        # 小样本验证：S_MAX 用 s_iso 更稳（别被 dt 放宽带雾）
        S_MAX_POLICY = "siso"
        DT_SMAX_PERCENTILE = 90
        DT_SMAX_FACTOR = 0.70

        # 法线更薄一点更锐（仍需防裂）
        SIGMA_N_FIXED = 0.006

    elif p == "final":
        # ---- Preset-Final: 全量/高密度（连续优先） ----
        SAMPLE_MODE = "voxel"
        VOXEL_SIZE = 0.02
        MAX_GAUSSIANS = -1
        MAX_GAUSSIANS_AFTER_SAMPLING = -1

        # 切向更宽，填洞更连续
        TANGENT_WEIGHT_SIGMA_FACTOR = 1.6
        TANGENT_WEIGHT_SIGMA_W_MAX = 0.25

        # dt -> st 更强（连续）
        DT_PERCENTILE = 20
        DT_TO_SIGMA_FACTOR = 0.75

        # 相对上限放宽
        TANGENT_REL_MAX = 1.10

        # 全量：允许 dt 放宽 S_MAX（更物理）
        S_MAX_POLICY = "max"
        DT_SMAX_PERCENTILE = 90
        DT_SMAX_FACTOR = 0.75

        SIGMA_N_FIXED = 0.008

    else:
        raise ValueError(f"Unknown preset: {preset}")

    print(f"[INFO] Preset applied: {PRESET}")
    print(f"[INFO] Sampling mode={SAMPLE_MODE}, voxel_size={VOXEL_SIZE}, max_gaussians={MAX_GAUSSIANS}, cap_after={MAX_GAUSSIANS_AFTER_SAMPLING}")
    print(f"[INFO] dt->st: DT_PERCENTILE={DT_PERCENTILE}, DT_TO_SIGMA_FACTOR={DT_TO_SIGMA_FACTOR:.2f}; tangent sigma_factor={TANGENT_WEIGHT_SIGMA_FACTOR:.2f}, w_max={TANGENT_WEIGHT_SIGMA_W_MAX:.2f}")
    print(f"[INFO] S_MAX_POLICY={S_MAX_POLICY}, SIGMA_N_FIXED={SIGMA_N_FIXED:.3f}")


# -------------------- 工具函数 --------------------

def compute_isotropic_scales(points, kdtree, k_neighbors):
    num = points.shape[0]
    scales_iso = np.zeros((num,), dtype=np.float32)
    dist_list = []

    print("[INFO] Estimating isotropic scales from neighbor distances...")
    for i in range(num):
        k, idx, dist2 = kdtree.search_knn_vector_3d(points[i], k_neighbors)
        if k > 1:
            mean_dist = float(np.sqrt(np.mean(dist2[1:])))
        else:
            mean_dist = 0.05

        s = mean_dist * 0.75
        scales_iso[i] = s
        dist_list.append(s)

        if i > 0 and i % 10000 == 0:
            print(f"  [ISO] processed {i}/{num} points...")

    dist_arr = np.asarray(dist_list)
    print("[INFO] Isotropic scale statistics (meters):")
    print(f"  min={dist_arr.min():.4f}, max={dist_arr.max():.4f}")
    print(f"  mean={dist_arr.mean():.4f}")
    for q in [5, 25, 50, 75, 95]:
        print(f"  {q}th percentile = {np.percentile(dist_arr, q):.4f}")

    neighbor_dist = dist_arr / 0.75
    print("[INFO] Neighbor distance vs Gaussian scale analysis:")
    print(f"  mean neighbor distance = {neighbor_dist.mean():.4f} m")
    print(f"  mean isotropic scale   = {dist_arr.mean():.4f} m")
    print(f"  ratio (neighbor / scale) = {neighbor_dist.mean() / dist_arr.mean():.2f}")

    print("  neighbor distance percentiles (m):")
    for q in [5, 25, 50, 75, 95]:
        print(f"    {q}th = {np.percentile(neighbor_dist, q):.4f}")

    print("  isotropic scale percentiles (m):")
    for q in [5, 25, 50, 75, 95]:
        print(f"    {q}th = {np.percentile(dist_arr, q):.4f}")

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
    """Return a sampled point cloud.

    - mode='all': return as-is
    - mode='voxel': voxel downsample (preserves local density better)
    - mode='random': uniform random sample
    """
    mode = (mode or "random").lower()

    if mode == "all" or max_points is None or max_points <= 0:
        return pcd

    if mode == "voxel":
        vs = float(max(1e-6, voxel_size))
        return pcd.voxel_down_sample(vs)

    # random
    n = np.asarray(pcd.points).shape[0]
    if n <= max_points:
        return pcd
    idx = np.random.choice(n, int(max_points), replace=False)
    return pcd.select_by_index(idx)


def estimate_dt_distribution(points: np.ndarray,
                             kdtree: o3d.geometry.KDTreeFlann,
                             scales_iso: np.ndarray,
                             max_points: int = 50000) -> dict:
    """Estimate tangent-plane distance d_t distribution on a subset.

    This is used to derive a TLS-friendly S_MAX from d_t rather than from s_iso.
    """
    n_total = points.shape[0]
    m = int(min(max_points, n_total))
    if m <= 0:
        return {"count": 0}

    if m < n_total:
        sel = np.random.choice(n_total, m, replace=False)
    else:
        sel = np.arange(n_total, dtype=np.int64)

    dt_vals = []
    fallback = 0

    print(f"[INFO] Estimating d_t distribution on {m}/{n_total} points (subset)...")

    for j, i in enumerate(sel):
        # normal neighbors
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

        if (j + 1) % 10000 == 0:
            print(f"  [d_t] processed {j+1}/{m}... (fallback={fallback})")

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

    print("[INFO] d_t statistics (meters) on subset:")
    print(f"  count={stats['count']}, fallback={stats['fallback']}")
    print(f"  min={stats['min']:.4f}, mean={stats['mean']:.4f}, max={stats['max']:.4f}")
    print(f"  p50={stats['p50']:.4f}, p75={stats['p75']:.4f}, p90={stats['p90']:.4f}, p95={stats['p95']:.4f}")

    return stats


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


def compute_normal_aligned_gaussians(points, kdtree, scales_iso):
    num = points.shape[0]
    scales_aniso = np.zeros((num, 3), dtype=np.float32)
    rotations = np.zeros((num, 9), dtype=np.float32)
    cov6 = np.zeros((num, 6), dtype=np.float32)

    sigma_t1_list = []
    sigma_t2_list = []
    sigma_n_list = []

    neff_list = []
    sigma_w_list = []

    # NEW stats: dt/ar/st_base and clamp hit ratio
    dt_list = []
    ar_list = []
    st_base_list = []
    hit_smax = 0
    hit_smin = 0

    fallback_normal = 0
    fallback_tangent = 0

    print("[INFO] Step 6: Estimating normal-aligned anisotropic Gaussians...")
    print(f"[INFO] K_NEIGHBORS_NORMAL={K_NEIGHBORS_NORMAL}, K_NEIGHBORS_TANGENT={K_NEIGHBORS_TANGENT}")
    print(f"[INFO] Tangent weighting={USE_TANGENT_WEIGHTING}, sigma_factor={TANGENT_WEIGHT_SIGMA_FACTOR:.2f}")
    print(f"[INFO] Tangent regularization: [{TANGENT_REL_MIN:.2f}, {TANGENT_REL_MAX:.2f}] * s_iso")
    print(f"[INFO] st_base from dt: DT_PERCENTILE={DT_PERCENTILE}, DT_TO_SIGMA_FACTOR={DT_TO_SIGMA_FACTOR:.2f}, AR=[{AR_MIN:.1f},{AR_MAX:.1f}]")
    print(f"[INFO] Sigma_n mode={SIGMA_N_MODE}, fixed={SIGMA_N_FIXED:.3f} m")
    print(f"[INFO] Using clamp range: S_MIN={S_MIN:.4f} m, S_MAX={S_MAX:.4f} m")

    for i in range(num):
        s_iso = float(scales_iso[i])
        s_iso_clamped = float(np.clip(s_iso, S_MIN, S_MAX))

        # 1) normal
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
        n_norm = np.linalg.norm(n)
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

        # 2) tangent
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

        # ---- 加权协方差（用于方向/形状比）----
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

        # 2D PCA: 只拿方向与形状比
        vals2, vecs2 = np.linalg.eigh(C2)
        order2 = np.argsort(vals2)[::-1]
        vals2 = np.clip(vals2[order2], 0.0, None)
        vecs2 = vecs2[:, order2]

        # ---- NEW: 用切平面距离分布估计局部点间距 d_t（绝对大小来源）----
        rho = np.sqrt(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)
        # 避免极小值干扰：去掉几乎 0 的（KNN 会包含非常近点）
        rho = rho[rho > 1e-9]
        if rho.size < 4:
            # 退化：回到 s_iso_clamped
            d_t = s_iso / 0.75 if s_iso > 1e-8 else 0.05
        else:
            d_t = float(np.percentile(rho, DT_PERCENTILE))

        # 形状比（aspect ratio）
        l1 = float(vals2[0] + 1e-12)
        l2 = float(vals2[1] + 1e-12)
        ar = float(np.sqrt(l1 / l2))
        ar = float(np.clip(ar, AR_MIN, AR_MAX))

        # st_base：真正控制“不会雾”的大小
        st_base = float(DT_TO_SIGMA_FACTOR * d_t)

        # 仍然允许用 s_iso 做一个轻度 regularization（防止局部 d_t 异常）
        st_base = float(np.clip(st_base, TANGENT_REL_MIN * s_iso, TANGENT_REL_MAX * s_iso))

        # clamp 到全局范围
        if st_base >= S_MAX - 1e-12:
            hit_smax += 1
        if st_base <= S_MIN + 1e-12:
            hit_smin += 1
        st_base = float(np.clip(st_base, S_MIN, S_MAX))

        # 用 ar 分解成 st1/st2
        s_ar = float(np.sqrt(ar))
        st1 = float(st_base * s_ar)
        st2 = float(st_base / max(s_ar, 1e-8))

        st1 = float(np.clip(st1, S_MIN, S_MAX))
        st2 = float(np.clip(st2, S_MIN, S_MAX))

        # normal sigma
        sn = compute_sigma_n(s_iso)
        sn = float(min(sn, 0.5 * max(min(st1, st2), 1e-6)))

        # 方向映射回 3D
        v1 = vecs2[:, 0].astype(np.float32)
        v2 = vecs2[:, 1].astype(np.float32)

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

        if i > 0 and i % 10000 == 0:
            print(f"  [N-ANISO] processed {i}/{num} points... "
                  f"(fallback_normal={fallback_normal}, fallback_tangent={fallback_tangent}, hitSmax={hit_smax})")

    # ---- 输出统计 ----
    st1_arr = np.asarray(sigma_t1_list, dtype=np.float64) if len(sigma_t1_list) else np.zeros((1,), dtype=np.float64)
    st2_arr = np.asarray(sigma_t2_list, dtype=np.float64) if len(sigma_t2_list) else np.zeros((1,), dtype=np.float64)
    sn_arr  = np.asarray(sigma_n_list, dtype=np.float64)  if len(sigma_n_list)  else np.zeros((1,), dtype=np.float64)

    print("[INFO] Step 6 done: normal-aligned anisotropic Gaussian estimation.")
    print("[INFO] Fallback counts:")
    print(f"  fallback_normal (too few neighbors / degenerate) = {fallback_normal}")
    print(f"  fallback_tangent (too few neighbors)            = {fallback_tangent}")

    print("[INFO] Tangent sigma statistics (meters):")
    for name, arr in [("st1", st1_arr), ("st2", st2_arr)]:
        print(f"  {name}: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
        for q in [5, 25, 50, 75, 95]:
            print(f"    {q}th percentile = {np.percentile(arr, q):.4f}")

    print("[INFO] Normal sigma statistics (meters):")
    print(f"  sn: min={sn_arr.min():.4f}, max={sn_arr.max():.4f}, mean={sn_arr.mean():.4f}")
    for q in [5, 25, 50, 75, 95]:
        print(f"    {q}th percentile = {np.percentile(sn_arr, q):.4f}")

    if len(dt_list) > 0:
        dt = np.asarray(dt_list, dtype=np.float64)
        ar = np.asarray(ar_list, dtype=np.float64)
        sb = np.asarray(st_base_list, dtype=np.float64)
        print("[INFO] dt/ar/st_base diagnostics:")
        print(f"  d_t (m): min={dt.min():.4f}, mean={dt.mean():.4f}, max={dt.max():.4f}")
        for q in [5, 25, 50, 75, 95]:
            print(f"    d_t {q}th = {np.percentile(dt, q):.4f}")
        print(f"  aspect ratio ar: min={ar.min():.2f}, mean={ar.mean():.2f}, max={ar.max():.2f}")
        print(f"  st_base (m): min={sb.min():.4f}, mean={sb.mean():.4f}, max={sb.max():.4f}")
        print(f"  clamp-hit rate: hitS_MAX={hit_smax}/{num} ({100.0*hit_smax/max(1,num):.2f}%), hitS_MIN={hit_smin}/{num} ({100.0*hit_smin/max(1,num):.2f}%)")

    if USE_TANGENT_WEIGHTING and len(neff_list) > 0:
        neff = np.asarray(neff_list, dtype=np.float64)
        sigw = np.asarray(sigma_w_list, dtype=np.float64)
        print("[INFO] Tangent weighting diagnostics:")
        print(f"  sigma_w (m): min={sigw.min():.4f}, mean={sigw.mean():.4f}, max={sigw.max():.4f}")
        print(f"  Neff (effective neighbors): min={neff.min():.1f}, mean={neff.mean():.1f}, max={neff.max():.1f}")
        for q in [5, 25, 50, 75, 95]:
            print(f"    Neff {q}th percentile = {np.percentile(neff, q):.1f}")

    print("[INFO] Anisotropic (st1,st2,sn) final stats (meters):")
    for axis, name in enumerate(["sx(st1)", "sy(st2)", "sz(sn)"]):
        col = scales_aniso[:, axis]
        print(f"  {name}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}")

    return scales_aniso, rotations, cov6


def main():
    print(f"[INFO] Loading point cloud from: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    pcd = o3d.io.read_point_cloud(str(INPUT_PATH))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    print(f"[INFO] Loaded {points.shape[0]} points")

    # ---- Pre-shift (float precision fix for large coordinates) ----
    origin_pre_shift = np.zeros((3,), dtype=np.float32)
    if ENABLE_PRE_SHIFT:
        # compute shift in the ORIGINAL input frame (before any ENU->Unity transform)
        origin_pre_shift = compute_recenter_origin(points.astype(np.float64), PRE_SHIFT_MODE).astype(np.float32)
        # translate the Open3D point cloud in-place (keeps colors aligned)
        pcd.translate((-origin_pre_shift).astype(np.float64), relative=True)
        # refresh numpy views
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        print("[INFO] Pre-shift enabled (float precision fix):")
        print(f"  PRE_SHIFT_MODE={PRE_SHIFT_MODE}")
        print(f"  origin_pre_shift (input frame) = [{origin_pre_shift[0]:.3f}, {origin_pre_shift[1]:.3f}, {origin_pre_shift[2]:.3f}]")


    # ---- Sampling (random / voxel / all) ----
    num_points_full = points.shape[0]

    pcd_full = pcd
    pcd_proc = sample_points_for_processing(
        pcd_full,
        max_points=MAX_GAUSSIANS,
        mode=SAMPLE_MODE,
        voxel_size=VOXEL_SIZE,
    )

    # If voxel sampling produced too many points, optionally cap with a final random sample.
    proc_n = np.asarray(pcd_proc.points).shape[0]
    cap_n = int(MAX_GAUSSIANS_AFTER_SAMPLING) if MAX_GAUSSIANS_AFTER_SAMPLING is not None else -1
    if cap_n > 0 and proc_n > cap_n:
        print(f"[INFO] Voxel/All sampling produced {proc_n} points; capping to {cap_n} by random sampling...")
        pcd_proc = sample_points_for_processing(pcd_proc, cap_n, mode="random", voxel_size=VOXEL_SIZE)
        proc_n = np.asarray(pcd_proc.points).shape[0]

    points = np.asarray(pcd_proc.points)
    colors = np.asarray(pcd_proc.colors)

    print(f"[INFO] Sampling mode={SAMPLE_MODE}, full={num_points_full} -> processed={points.shape[0]}")

    pcd_sample = o3d.geometry.PointCloud()
    pcd_sample.points = o3d.utility.Vector3dVector(points)

    if colors.size > 0:
        if colors.max() > 1.1:
            colors = colors / 255.0
        pcd_sample.colors = o3d.utility.Vector3dVector(colors)

    print("[INFO] Building KDTree...")
    kdtree = o3d.geometry.KDTreeFlann(pcd_sample)

    scales_iso = compute_isotropic_scales(points, kdtree, K_NEIGHBORS_ISO)

    global S_MIN, S_MAX
    if USE_ADAPTIVE_CLAMP:
        S_MIN, S_MAX = compute_adaptive_clamp(scales_iso)
        print("[INFO] Adaptive clamp enabled:")
        print(f"  ADAPTIVE_Q_MIN={ADAPTIVE_Q_MIN} -> S_MIN={S_MIN:.4f} m")
        print(f"  ADAPTIVE_Q_MAX={ADAPTIVE_Q_MAX} -> S_MAX={S_MAX:.4f} m")
    else:
        print("[INFO] Adaptive clamp disabled:")
        print(f"  Using fixed S_MIN={S_MIN:.4f} m, S_MAX={S_MAX:.4f} m")

    # ---- Optional: refine S_MAX using d_t distribution (Scheme 2) ----
    s_max_siso = float(S_MAX)
    s_max_dt = None

    if USE_DT_BASED_SMAX:
        dt_stats = estimate_dt_distribution(points, kdtree, scales_iso, max_points=DT_STATS_MAX_POINTS)
        if dt_stats.get("count", 0) > 0:
            key = f"p{int(DT_SMAX_PERCENTILE)}"
            # choose percentile value safely
            if key in dt_stats:
                dt_ref = float(dt_stats[key])
            else:
                # fallback to p90
                dt_ref = float(dt_stats.get("p90", dt_stats.get("p75", dt_stats.get("p50", 0.0))))

            s_max_dt = float(np.clip(DT_SMAX_FACTOR * dt_ref, ABS_S_MIN, ABS_S_MAX))
            print("[INFO] d_t-based S_MAX estimation:")
            print(f"  DT_SMAX_PERCENTILE={DT_SMAX_PERCENTILE} -> d_t_ref={dt_ref:.4f} m")
            print(f"  DT_SMAX_FACTOR={DT_SMAX_FACTOR:.2f} -> S_MAX_dt={s_max_dt:.4f} m")

    # Combine S_MAX (Scheme 1 + Scheme 2)
    policy = (S_MAX_POLICY or "max").lower()
    if policy == "dt" and s_max_dt is not None:
        S_MAX = float(s_max_dt)
        print(f"[INFO] S_MAX_POLICY=dt -> S_MAX={S_MAX:.4f} m")
    elif policy == "siso" or s_max_dt is None:
        S_MAX = float(s_max_siso)
        print(f"[INFO] S_MAX_POLICY=siso -> S_MAX={S_MAX:.4f} m")
    else:
        S_MAX = float(max(s_max_siso, s_max_dt))
        S_MAX = float(np.clip(S_MAX, ABS_S_MIN, ABS_S_MAX))
        print(f"[INFO] S_MAX_POLICY=max -> S_MAX=max({s_max_siso:.4f}, {s_max_dt:.4f}) = {S_MAX:.4f} m")

    # Ensure S_MAX > S_MIN
    if S_MAX <= S_MIN:
        S_MAX = float(min(ABS_S_MAX, S_MIN * 2.0))
        print(f"[WARN] Adjusted S_MAX to keep it > S_MIN: S_MAX={S_MAX:.4f} m")

    scales_aniso, rotations, cov6 = compute_normal_aligned_gaussians(points, kdtree, scales_iso)

    if APPLY_UNITY_FRAME:
        print("[INFO] Applying ENU -> Unity frame transform (Z=North, Y=Up)...")
        points, rotations, cov6 = apply_frame_transform(points, rotations, cov6, M_ENU_TO_UNITY)

    origin = np.zeros((3,), dtype=np.float32)
    if APPLY_RECENTER:
        origin = compute_recenter_origin(points, RECENTER_MODE)
        points = apply_recenter(points, origin)
        print("[INFO] Recentering enabled:")
        print(f"  RECENTER_MODE={RECENTER_MODE}")
        print(f"  origin (Unity frame) = [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]")

    print("[INFO] Covariance diag stats (meters^2):")
    print(f"  xx: min={cov6[:,0].min():.6e}, max={cov6[:,0].max():.6e}")
    print(f"  yy: min={cov6[:,3].min():.6e}, max={cov6[:,3].max():.6e}")
    print(f"  zz: min={cov6[:,5].min():.6e}, max={cov6[:,5].max():.6e}")

    opacity = np.ones((points.shape[0],), dtype=np.float32)
    print("[INFO] XYZ range (Unity frame): "
          f"X[{points[:,0].min():.2f}, {points[:,0].max():.2f}], "
          f"Y[{points[:,1].min():.2f}, {points[:,1].max():.2f}], "
          f"Z[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
    if colors.size > 0:
        print("[INFO] Colors range: "
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

    # Avoid overwriting between presets
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
    print(f"[INFO] Saved Gaussians to: {out_npz}")

    txt_path = out_npz.with_suffix(".txt")
    print(f"[INFO] Also writing simple text format to: {txt_path}")

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
    # Save both: pre-shift (input frame) and final recenter origin (Unity frame, before subtract)
    # Line1: origin_pre_shift (input frame)
    # Line2: origin (Unity frame)
    np.savetxt(origin_path, np.vstack([origin_pre_shift.reshape(1, 3), origin.reshape(1, 3)]), fmt="%.6f")
    print(f"[INFO] Saved origin sidecar (2 lines: pre_shift(input), origin(Unity)) to: {origin_path}")

    print("[INFO] Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=PRESET_DEFAULT, choices=["verify", "final"])
    args = ap.parse_args()
    apply_preset(args.preset)
    main()