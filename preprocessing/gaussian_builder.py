import numpy as np
import open3d as o3d
from pathlib import Path

# ==== 配置 ====

# CloudCompare 导出的、已经居中/清洗过的点云
INPUT_PATH = Path("data/navvis_house2_centered.ply")

# 输出的 Gaussian 参数（给以后研究用）
OUTPUT_PATH = Path("data/navvis_house2_gaussians_demo.npz")

# 用于 Unity demo 的最大高斯数量
MAX_GAUSSIANS = 50000

# 用于估计局部尺度的邻域大小
K_NEIGHBORS = 8


def main():
    print(f"[INFO] Loading point cloud from: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    # 1) 读取点云
    pcd = o3d.io.read_point_cloud(str(INPUT_PATH))
    points = np.asarray(pcd.points)      # (N, 3)
    colors = np.asarray(pcd.colors)      # (N, 3) in [0,1] or [0,255]

    print(f"[INFO] Loaded {points.shape[0]} points")

    # 2) 随机子采样一部分点用于 demo
    num_points = points.shape[0]
    if num_points > MAX_GAUSSIANS:
        idx = np.random.choice(num_points, MAX_GAUSSIANS, replace=False)
        points = points[idx]
        colors = colors[idx]
        print(f"[INFO] Randomly sampled {MAX_GAUSSIANS} points for demo")
    else:
        print(f"[INFO] Using all {num_points} points")

    # 3) 构建 Open3D 点云，准备 KDTree
    pcd_sample = o3d.geometry.PointCloud()
    pcd_sample.points = o3d.utility.Vector3dVector(points)

    # 颜色归一化
    if colors.max() > 1.1:
        colors = colors / 255.0
    pcd_sample.colors = o3d.utility.Vector3dVector(colors)

    # 4) KDTree 找邻居，用于估计尺度
    print("[INFO] Building KDTree...")
    kdtree = o3d.geometry.KDTreeFlann(pcd_sample)

    num = points.shape[0]
    scales = np.zeros((num, 3), dtype=np.float32)

    print("[INFO] Estimating per-point scales from neighbor distances...")
    for i in range(num):
        # 搜索 K 近邻
        k, idx, dist2 = kdtree.search_knn_vector_3d(points[i], K_NEIGHBORS)
        if k > 1:
            mean_dist = np.sqrt(np.mean(dist2[1:]))  # 排除自身
        else:
            mean_dist = 0.05  # fallback

        s = mean_dist * 0.75
        scales[i] = np.array([s, s, s], dtype=np.float32)

    print("[INFO] Scale estimation done.")

    # 旋转：全部设单位矩阵（以后可升级为 PCA）
    rotations = np.tile(np.eye(3, dtype=np.float32).reshape(9), (num, 1))

    # 不透明度
    opacity = np.ones((num,), dtype=np.float32)

    # 保存 npz（正式研究时使用）
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        positions=points.astype(np.float32),
        scales=scales,
        rotations=rotations,
        colors=colors.astype(np.float32),
        opacity=opacity.astype(np.float32),
    )

    print(f"[INFO] Saved Gaussians to: {OUTPUT_PATH}")

    # ===== Unity 简易版 TXT 输出 =====
    txt_path = OUTPUT_PATH.with_suffix(".txt")
    print(f"[INFO] Also writing simple text format to: {txt_path}")

    data_mat = np.hstack([points, scales, colors])
    np.savetxt(txt_path, data_mat, fmt="%.6f")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()