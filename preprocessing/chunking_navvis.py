import numpy as np
from pathlib import Path
import json

# ====== 配置 ======

# 输入：Stage A 生成的 NPZ（positions + colors + cov6/cov0/cov1 ...）
NPZ_PATH = Path("data/TumTLS_v2_gaussians_demo_verify.npz")

# 输出：chunk txt 文件目录
CHUNK_DIR = Path("data/navvis_chunks_TUMv2")

# chunk 文件命名前缀
CHUNK_PREFIX = "navvis_chunk"

# chunk 尺寸（单位：米），可按需要调整
CHUNK_SIZE = np.array([10.0, 10.0, 10.0], dtype=np.float32)  # dx, dy, dz

# 允许边界上点恰好落在 max 时归到最后一个格子
EPS = 1e-5


def main():
    print("[B] Loading NPZ:", NPZ_PATH)
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"NPZ file not found: {NPZ_PATH}")

    data = np.load(NPZ_PATH)
    P = data["positions"]   # (N,3)
    C = data["colors"]      # (N,3)

    # New anisotropic ellipse splatting expects covariance per Gaussian.
    # Prefer cov6 (xx,xy,xz,yy,yz,zz). Fall back to cov0/cov1 if needed.
    if "cov6" in data:
        cov6 = data["cov6"].astype(np.float32)  # (N,6)
    elif "cov0" in data and "cov1" in data:
        cov0 = data["cov0"].astype(np.float32)  # (N,4) (xx,xy,xz,yy)
        cov1 = data["cov1"].astype(np.float32)  # (N,4) (yz,zz,0,0)
        cov6 = np.stack([
            cov0[:, 0], cov0[:, 1], cov0[:, 2], cov0[:, 3], cov1[:, 0], cov1[:, 1]
        ], axis=1).astype(np.float32)
    else:
        raise KeyError("NPZ must contain 'cov6' or both 'cov0' and 'cov1' for anisotropic splatting.")

    # Optional: keep scales if present (not used for chunk TXT anymore)
    S = data["scales"] if "scales" in data else None

    N = P.shape[0]
    print(f"[B1] Loaded {N} Gaussians")

    # ===== B1: 计算全局 bounding box + 网格大小 =====
    xyz_min = P.min(axis=0)
    xyz_max = P.max(axis=0)

    print("[B1] XYZ range:")
    print(f"  X[{xyz_min[0]:.2f}, {xyz_max[0]:.2f}]")
    print(f"  Y[{xyz_min[1]:.2f}, {xyz_max[1]:.2f}]")
    print(f"  Z[{xyz_min[2]:.2f}, {xyz_max[2]:.2f}]")

    # chunk 原点从全局最小值开始
    origin = xyz_min.copy()

    # 网格数量（nx, ny, nz）
    extent = xyz_max - xyz_min
    grid = np.ceil((extent + EPS) / CHUNK_SIZE).astype(int)
    nx, ny, nz = grid
    print(f"[B1] Chunk grid (nx,ny,nz) = {nx} {ny} {nz}")

    # ===== B2: 为每个点计算 chunk 索引 (ix,iy,iz) =====
    # 相对坐标
    rel = P - origin[None, :]           # (N,3)
    ijk = np.floor(rel / CHUNK_SIZE[None, :]).astype(int)

    # Clamp，确保在 [0, n-1] 范围内
    ijk[:, 0] = np.clip(ijk[:, 0], 0, nx - 1)
    ijk[:, 1] = np.clip(ijk[:, 1], 0, ny - 1)
    ijk[:, 2] = np.clip(ijk[:, 2], 0, nz - 1)

    # 用字典记录各 chunk 中点的 index
    chunk_dict = {}
    for idx_point in range(N):
        ix, iy, iz = ijk[idx_point]
        key = (int(ix), int(iy), int(iz))
        if key not in chunk_dict:
            chunk_dict[key] = []
        chunk_dict[key].append(idx_point)

    non_empty_chunks = sorted(chunk_dict.keys())
    print(f"[B2] Number of non-empty chunks: {len(non_empty_chunks)}")

    total_pts = 0
    for key in non_empty_chunks[:10]:   # 只打印前 10 个
        n_pts = len(chunk_dict[key])
        ix, iy, iz = key
        print(f"  chunk ({ix}, {iy}, {iz}) has {n_pts} points")
        total_pts += n_pts

    # 统计所有 chunk 的总点数（用于 sanity check）
    total_pts = sum(len(v) for v in chunk_dict.values())
    print(f"[B2] Total points over all chunks = {total_pts}")

    # ===== B3: 写出每个 chunk 对应的 txt 文件 =====
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    # 用于 B4：收集元数据，基于协方差格式
    chunk_meta = []

    print("[B3] Writing chunk files to:", CHUNK_DIR)
    for (ix, iy, iz), idx_list in chunk_dict.items():
        idx_arr = np.array(idx_list, dtype=int)

        P_chunk = P[idx_arr]      # (M,3)
        C_chunk = C[idx_arr]      # (M,3)
        cov6_chunk = cov6[idx_arr]  # (M,6)

        # New 12-column format: x y z r g b xx xy xz yy yz zz
        mat = np.hstack([P_chunk, C_chunk, cov6_chunk])

        fname = f"{CHUNK_PREFIX}_{ix}_{iy}_{iz}.txt"
        fpath = CHUNK_DIR / fname

        np.savetxt(fpath, mat, fmt="%.6f")

        # 为 B4 记录 chunk 的元数据（bbox、中心、点数等）
        bbox_min = P_chunk.min(axis=0)
        bbox_max = P_chunk.max(axis=0)
        center   = 0.5 * (bbox_min + bbox_max)

        meta = {
            "ijk": [int(ix), int(iy), int(iz)],
            "filename": fname,
            "count": int(P_chunk.shape[0]),
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "center": center.tolist(),
        }
        chunk_meta.append(meta)

    print(f"[B3] Wrote {len(chunk_meta)} chunk files.")

    # ===== B4: 写出 chunk 索引 JSON =====
    index = {
        "npz_source": str(NPZ_PATH),
        "origin": origin.tolist(),
        "chunk_size": CHUNK_SIZE.tolist(),
        "grid_shape": [int(nx), int(ny), int(nz)],
        "num_points": int(N),
        "num_chunks": len(chunk_meta),
        "chunks": chunk_meta,
    }

    index_path = CHUNK_DIR / "navvis_chunks_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"[B4] Wrote chunk index JSON to: {index_path}")
    print("[B4] Done.")


if __name__ == "__main__":
    main()