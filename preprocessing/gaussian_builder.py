import numpy as np
import open3d as o3d
from pathlib import Path

# ==== 配置 ====

# CloudCompare 导出的、已经居中/清洗过的点云
INPUT_PATH = Path("data/SampleBlock1.ply")

# 输出的 Gaussian 参数（给以后研究用）
OUTPUT_PATH = Path("data/SampleBlock1_gaussians_demo.npz")

# 用于 demo 的最大高斯数量（开发阶段可以改小一点，比如 20000，加速调试）
MAX_GAUSSIANS = 200000

# 用于估计各向同性尺度的邻域大小
K_NEIGHBORS_ISO = 8

# 用于各向异性协方差估计的邻域大小（略大一点，更稳定）
K_NEIGHBORS_ANISO = 16

# 对尺度的 clamp，避免高斯过大或过小（单位：米）
S_MIN = 0.02   # 2 cm
S_MAX = 0.50   # 50 cm


def compute_isotropic_scales(points, kdtree, k_neighbors):
    """A2: 各向同性尺度估计 + 统计输出"""
    num = points.shape[0]
    scales_iso = np.zeros((num,), dtype=np.float32)
    dist_list = []

    print("[INFO] Estimating isotropic scales from neighbor distances...")
    for i in range(num):
        k, idx, dist2 = kdtree.search_knn_vector_3d(points[i], k_neighbors)
        if k > 1:
            mean_dist = float(np.sqrt(np.mean(dist2[1:])))  # 排除自身
        else:
            mean_dist = 0.05

        s = mean_dist * 0.75  # 经验系数
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

    return scales_iso


def orthonormalize_rotation(R):
    """
    用 SVD 对 3x3 矩阵做正交化，确保 R^T R ≈ I, det(R) ≈ 1.
    适合修正数值误差或略有噪声的特征向量矩阵。
    """
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt

    # 避免 det(R) = -1（反射），强制右手系
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1.0
        R_ortho = U @ Vt

    return R_ortho


def compute_anisotropic_gaussians(points, kdtree, k_neighbors, scales_iso):
    """
    A3: 基于邻域协方差 + PCA 估计各向异性 Gaussian
    返回：
      scales_aniso: (N,3) 每个点的 (sx, sy, sz)
      rotations:   (N,9) 每个点的 3x3 旋转矩阵（按列展平）
    """
    num = points.shape[0]
    scales_aniso = np.zeros((num, 3), dtype=np.float32)
    rotations = np.zeros((num, 9), dtype=np.float32)

    eigvals_all = []

    print("[INFO] Estimating anisotropic Gaussians (covariance + PCA)...")
    for i in range(num):
        # 找邻居
        k, idx, _ = kdtree.search_knn_vector_3d(points[i], k_neighbors)
        if k <= 3:
            # 邻居太少，退化到各向同性
            s_iso = scales_iso[i]
            s_clamped = np.clip(s_iso, S_MIN, S_MAX)
            scales_aniso[i] = np.array([s_clamped, s_clamped, s_clamped], dtype=np.float32)
            rotations[i] = np.eye(3, dtype=np.float32).reshape(-1)
            continue

        # 邻域点坐标（排除自身）
        neigh = points[idx[1:], :]  # (k-1, 3)
        center = neigh.mean(axis=0, keepdims=True)
        X = neigh - center  # 去中心化

        # 协方差矩阵 (3x3): cov = (X^T X) / (k-1)
        cov = (X.T @ X) / (X.shape[0] - 1)

        # 特征分解
        vals, vecs = np.linalg.eigh(cov)  # eigenvalues 升序，列为 eigenvectors

        # 从大到小排序，方便解释 principal / secondary / minor axes
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        # 防止负数（数值误差）
        vals = np.clip(vals, 0.0, None)

        # eigenvalues 对应 σ^2 → σ
        sigma = np.sqrt(vals + 1e-12)

        # 用各向同性 s_iso 做 regularization，避免某方向极端小或大
        s_iso = scales_iso[i]
        sigma = np.clip(sigma, 0.5 * s_iso, 2.0 * s_iso)

        # 再做全局 clamp，保证在 [S_MIN, S_MAX] 内
        sigma = np.clip(sigma, S_MIN, S_MAX)

        scales_aniso[i] = sigma.astype(np.float32)

        # 旋转矩阵：列向量为特征向量 → 做一次正交化，保证正交性 & det=1
        R = vecs.astype(np.float32)
        R = orthonormalize_rotation(R)
        rotations[i] = R.reshape(-1)

        eigvals_all.append(vals)

        if i > 0 and i % 10000 == 0:
            print(f"  [ANISO] processed {i}/{num} points...")

    eigvals_all = np.vstack(eigvals_all)
    print("[INFO] Anisotropic eigenvalues (lambda) statistics:")
    for axis, name in enumerate(["lambda1", "lambda2", "lambda3"]):
        col = eigvals_all[:, axis]
        print(f"  {name}: min={col.min():.6f}, max={col.max():.6f}, mean={col.mean():.6f}")

    print("[INFO] Anisotropic scales (sigma) statistics (meters):")
    for axis, name in enumerate(["sx", "sy", "sz"]):
        col = scales_aniso[:, axis]
        print(f"  {name}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}")

    return scales_aniso, rotations


def main():
    print(f"[INFO] Loading point cloud from: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    # A1: 读取点云
    pcd = o3d.io.read_point_cloud(str(INPUT_PATH))
    points = np.asarray(pcd.points)      # (N, 3)
    colors = np.asarray(pcd.colors)      # (N, 3) in [0,1] or [0,255]
    print(f"[INFO] Loaded {points.shape[0]} points")

    # A1.5: 随机子采样一部分点用于 demo / 开发
    num_points = points.shape[0]
    if num_points > MAX_GAUSSIANS:
        idx = np.random.choice(num_points, MAX_GAUSSIANS, replace=False)
        points = points[idx]
        colors = colors[idx] if colors.size > 0 else colors
        print(f"[INFO] Randomly sampled {points.shape[0]} points for demo (MAX_GAUSSIANS={MAX_GAUSSIANS})")
    else:
        print(f"[INFO] Using all {num_points} points")

    # 构建 Open3D 点云，KDTree
    pcd_sample = o3d.geometry.PointCloud()
    pcd_sample.points = o3d.utility.Vector3dVector(points)

    # 颜色归一化
    if colors.size > 0:
        if colors.max() > 1.1:
            colors = colors / 255.0
        pcd_sample.colors = o3d.utility.Vector3dVector(colors)

    print("[INFO] Building KDTree...")
    kdtree = o3d.geometry.KDTreeFlann(pcd_sample)

    # A2: 各向同性尺度 s_iso
    scales_iso = compute_isotropic_scales(points, kdtree, K_NEIGHBORS_ISO)

    # A3: 各向异性 Gaussian（协方差 + PCA）
    scales_aniso, rotations = compute_anisotropic_gaussians(
        points, kdtree, K_NEIGHBORS_ANISO, scales_iso
    )
    print("[INFO] Anisotropic Gaussian estimation done.")

    # A4: 基础数值检查
    opacity = np.ones((points.shape[0],), dtype=np.float32)
    print("[INFO] XYZ range: "
          f"X[{points[:,0].min():.2f}, {points[:,0].max():.2f}], "
          f"Y[{points[:,1].min():.2f}, {points[:,1].max():.2f}], "
          f"Z[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
    if colors.size > 0:
        print("[INFO] Colors range: "
              f"R[{colors[:,0].min():.2f}, {colors[:,0].max():.2f}], "
              f"G[{colors[:,1].min():.2f}, {colors[:,1].max():.2f}], "
              f"B[{colors[:,2].min():.2f}, {colors[:,2].max():.2f}]")

    # 保存 npz（研究主数据）
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        positions=points.astype(np.float32),
        scales=scales_aniso.astype(np.float32),    # ★ 各向异性 σx,σy,σz
        scales_iso=scales_iso.astype(np.float32),  # ★ 方便后续对比 / fallback
        rotations=rotations.astype(np.float32),    # ★ 正交化后的 R (列展平)
        colors=colors.astype(np.float32),
        opacity=opacity.astype(np.float32),
    )
    print(f"[INFO] Saved Gaussians to: {OUTPUT_PATH}")

    # Unity 简易版 TXT 输出：仍然用各向同性尺度，兼容当前 Unity demo
    txt_path = OUTPUT_PATH.with_suffix(".txt")
    print(f"[INFO] Also writing simple text format to: {txt_path}")

    if colors.size == 0:
        # 如果没有颜色，可以给个默认灰色
        colors_txt = np.tile(np.array([[0.7, 0.7, 0.7]], dtype=np.float32), (points.shape[0], 1))
    else:
        colors_txt = colors

    # x y z sx sy sz r g b，其中 sx=sy=sz=各向同性尺度
    data_mat = np.hstack([
        points,
        np.repeat(scales_iso[:, None], 3, axis=1),
        colors_txt
    ])
    np.savetxt(txt_path, data_mat, fmt="%.6f")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()