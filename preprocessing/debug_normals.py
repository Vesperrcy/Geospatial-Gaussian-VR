import numpy as np
import open3d as o3d
from pathlib import Path

# ===== 配置 =====
PLY_PATH = Path("data/TumTLS_v2.ply")  # ← 改成你的 PLY 路径

def main():
    print("[DEBUG] Loading PLY:", PLY_PATH)
    if not PLY_PATH.exists():
        raise FileNotFoundError(f"PLY not found: {PLY_PATH}")

    pcd = o3d.io.read_point_cloud(str(PLY_PATH))

    # ---- 基本信息 ----
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    nrm  = np.asarray(pcd.normals)

    print("\n[DEBUG] Basic info")
    print("  points shape :", pts.shape)
    print("  colors shape :", cols.shape)
    print("  normals shape:", nrm.shape)
    print("  has_normals  :", pcd.has_normals())

    # ---- normals 详细检查 ----
    if not pcd.has_normals() or nrm.size == 0:
        print("\n[ERROR] No normals loaded by Open3D.")
        print("        -> CloudCompare has normals, but they were NOT read into Open3D.")
        print("        -> This means gaussian_builder WILL estimate normals.")
        return

    # 计算法线长度
    norms = np.linalg.norm(nrm, axis=1)

    print("\n[DEBUG] Normal statistics")
    print(f"  count        : {norms.size}")
    print(f"  min(norm)    : {norms.min():.6f}")
    print(f"  max(norm)    : {norms.max():.6f}")
    print(f"  mean(norm)   : {norms.mean():.6f}")
    print(f"  std(norm)    : {norms.std():.6f}")

    # 判断是否单位法线
    non_unit = np.sum((norms < 0.9) | (norms > 1.1))
    print(f"  non-unit normals (|n|<0.9 or >1.1): {non_unit}")

    # ---- 抽样打印几条 ----
    print("\n[DEBUG] Sample normals (first 5):")
    for i in range(min(5, nrm.shape[0])):
        nx, ny, nz = nrm[i]
        ln = norms[i]
        print(f"  {i:02d}: ({nx:+.4f}, {ny:+.4f}, {nz:+.4f}) |n|={ln:.4f}")

    # ---- 结论建议 ----
    print("\n[DEBUG] Conclusion:")
    if abs(norms.mean() - 1.0) < 1e-2:
        print("  ✔ Normals are unit-length. Safe to use directly.")
        print("  ✔ gaussian_builder SHOULD skip estimating normals.")
    else:
        print("  ⚠ Normals are NOT unit-length.")
        print("  → You should normalize normals after loading, e.g.:")
        print("    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)")

if __name__ == "__main__":
    main()
