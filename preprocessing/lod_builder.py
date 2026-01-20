import json
import numpy as np
from pathlib import Path

# ===== 配置 =====

# Stage B 生成的 chunk 索引和目录
CHUNK_DIR = Path("data/navvis_chunks_TUMv2")
CHUNK_INDEX_PATH = CHUNK_DIR / "navvis_chunks_index.json"

# LOD 参数
# L0 = 原始数据，不做子采样
LOD_LEVELS = [0, 1, 2]                # 想要的 LOD 层级
LOD_FACTORS = {0: 1, 1: 4, 2: 16}     # 每级大致的下采样因子（L1 ~ 1/4，L2 ~ 1/16）

MIN_POINTS_PER_LEVEL = {
    0: 1,     # L0 其实用不到
    1: 100,   # 每个 chunk 在 L1 至少保留 100 点
    2: 50,    # L2 至少保留 50 点
}

CHUNK_PREFIX = "navvis_chunk_"
CHUNK_SUFFIX = ".txt"

# 输出的 LOD 索引 JSON
LOD_INDEX_PATH = CHUNK_DIR / "navvis_chunks_lod_index.json"


# ===== C1: 构建单个 chunk 的多级 LOD =====

def build_lods_for_chunk(chunk_meta):
    """
    输入：一个来自 navvis_chunks_index.json 的 chunk meta 字典
    输出：dict，记录每个 LOD 的 filename 与 count（对输入行做子采样，整行原样保留；兼容 9/12 列格式）
    """
    ijk = chunk_meta["ijk"]
    ix, iy, iz = ijk
    base_fname = chunk_meta["filename"]

    base_path = CHUNK_DIR / base_fname
    if not base_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {base_path}")

    # 加载原始 L0 数据
    # Old format (9 cols):  x y z sx sy sz r g b
    # New format (12 cols): x y z r g b xx xy xz yy yz zz
    arr = np.loadtxt(base_path)
    if arr.ndim == 1:
        arr = arr[None, :]   # 单行情况

    # Validate expected column count. LOD sampling must preserve entire rows.
    n_cols = arr.shape[1]
    if n_cols not in (9, 12):
        raise ValueError(f"Unexpected column count in {base_path}: {n_cols} (expected 9 or 12)")

    # If still using old 9-col chunks, remind the user to re-run Stage B with covariance output.
    if n_cols == 9:
        print(f"[C1][WARN] Chunk TXT is 9 columns (legacy). For anisotropic ellipse splatting, regenerate chunks to 12 columns: {base_path.name}")

    n_total = arr.shape[0]

    lod_info = {}

    for level in LOD_LEVELS:
        if level == 0:
            # L0 = 原始数据，单独写一份 *_L0.txt
            fname_lod = f"{CHUNK_PREFIX}{ix}_{iy}_{iz}_L0{CHUNK_SUFFIX}"
            path_lod = CHUNK_DIR / fname_lod

            np.savetxt(path_lod, arr, fmt="%.6f")
            lod_info["L0"] = {
                "filename": fname_lod,
                "count": int(n_total),
            }
            continue

        # 其它 LOD：做随机子采样
        factor = LOD_FACTORS[level]
        target = max(int(n_total / factor), MIN_POINTS_PER_LEVEL[level])
        target = min(target, n_total)  # 不超过原始数量

        if target == n_total:
            # 点数太少，不做下采样
            arr_lod = arr
        else:
            idx = np.random.choice(n_total, target, replace=False)
            arr_lod = arr[idx]

        fname_lod = f"{CHUNK_PREFIX}{ix}_{iy}_{iz}_L{level}{CHUNK_SUFFIX}"
        path_lod = CHUNK_DIR / fname_lod

        np.savetxt(path_lod, arr_lod, fmt="%.6f")

        lod_info[f"L{level}"] = {
            "filename": fname_lod,
            "count": int(arr_lod.shape[0]),
        }

    return lod_info


# ===== C2: 处理所有 chunk + 写 LOD 索引 JSON =====

def main():
    print("=== Stage C: Building LODs for NavVis chunks ===")

    if not CHUNK_INDEX_PATH.exists():
        raise FileNotFoundError(f"Chunk index JSON not found: {CHUNK_INDEX_PATH}")

    with open(CHUNK_INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)

    chunks = index["chunks"]
    print(f"[C2] Loaded chunk index with {len(chunks)} chunks.")

    lod_chunks_meta = []
    total_L0 = 0
    total_L1 = 0
    total_L2 = 0

    for i, meta in enumerate(chunks):
        ijk = meta["ijk"]
        print(f"  [C1] Processing chunk {i+1}/{len(chunks)} ijk={ijk} ...")

        lod_info = build_lods_for_chunk(meta)

        # ===== 关键修改：把 lod_info 映射成数组 lod_files / lod_counts =====
        lod_files = []
        lod_counts = []
        for level in LOD_LEVELS:
            key = f"L{level}"
            if key in lod_info:
                lod_files.append(lod_info[key]["filename"])
                lod_counts.append(int(lod_info[key]["count"]))
            else:
                # 如果某个 LOD 缺失，用占位
                lod_files.append("")
                lod_counts.append(0)

        # 合并原 meta + lod 信息
        meta_lod = dict(meta)
        meta_lod["lod"] = lod_info
        meta_lod["lod_files"] = lod_files   # ← JSON: ["L0file","L1file","L2file"]
        meta_lod["lod_counts"] = lod_counts # ← JSON: [count0,count1,count2]
        lod_chunks_meta.append(meta_lod)
        # ========================================

        total_L0 += lod_info["L0"]["count"]
        if "L1" in lod_info:
            total_L1 += lod_info["L1"]["count"]
        if "L2" in lod_info:
            total_L2 += lod_info["L2"]["count"]

    # 构建新的 LOD 索引结构
    lod_index = {
        "npz_source": index.get("npz_source", ""),
        "origin": index.get("origin", []),
        "chunk_size": index.get("chunk_size", []),
        "grid_shape": index.get("grid_shape", []),
        "num_points": index.get("num_points", 0),
        "num_chunks": len(lod_chunks_meta),
        "lod_levels": LOD_LEVELS,
        "lod_factors": LOD_FACTORS,
        "min_points_per_level": MIN_POINTS_PER_LEVEL,
        "chunks": lod_chunks_meta,
    }

    LOD_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOD_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(lod_index, f, indent=2)

    print("\n=== Summary ===")
    print(f"  Total L0 points over all chunks = {total_L0}")
    print(f"  Total L1 points over all chunks = {total_L1}")
    print(f"  Total L2 points over all chunks = {total_L2}")
    print(f"[C2] Wrote LOD index JSON to: {LOD_INDEX_PATH}")
    print("[C2] Done.")


if __name__ == "__main__":
    main()