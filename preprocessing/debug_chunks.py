import numpy as np
from pathlib import Path

NPZ_PATH = Path("data/navvis_house2_gaussians_demo.npz")
CHUNK_DIR = Path("data/navvis_chunks")
CHUNK_PREFIX = "navvis_chunk_"
CHUNK_SUFFIX = ".txt"


def main():
    print("=== Debug Chunk Statistics ===")

    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"NPZ file not found: {NPZ_PATH}")

    data = np.load(NPZ_PATH)
    P = data["positions"]
    n_orig = P.shape[0]
    print(f"[INFO] Original Gaussians: {n_orig} points\n")

    # 找到所有 chunk txt 文件
    chunk_files = sorted(CHUNK_DIR.glob(f"{CHUNK_PREFIX}*{CHUNK_SUFFIX}"))
    print(f"[INFO] Found {len(chunk_files)} chunk files.")

    total = 0
    for f in chunk_files:
        arr = np.loadtxt(f)

        # 根据维度安全地统计行数
        if arr.size == 0:
            n_pts = 0
        elif arr.ndim == 1:
            # 一行 9 列被挤成 shape=(9,)
            n_pts = 1
        else:
            n_pts = arr.shape[0]

        total += n_pts
        print(f"  {f.name}: {n_pts} points")

    print("\n=== Summary ===")
    print(f"Total points across all chunks = {total}")
    if total == n_orig:
        print("[OK] Chunking is consistent with original NPZ.")
    else:
        print("[WARNING] Mismatch detected!")
        print(f"Original = {n_orig}, but chunks sum = {total}")


if __name__ == "__main__":
    main()