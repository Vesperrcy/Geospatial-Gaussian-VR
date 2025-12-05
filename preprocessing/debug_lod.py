import json
from pathlib import Path
import numpy as np

CHUNK_DIR = Path("data/navvis_chunks")
LOD_INDEX_PATH = CHUNK_DIR / "navvis_chunks_lod_index.json"

def main():
    print("=== Debug LOD Index ===")

    if not LOD_INDEX_PATH.exists():
        raise FileNotFoundError(f"LOD index not found: {LOD_INDEX_PATH}")

    with open(LOD_INDEX_PATH, "r", encoding="utf-8") as f:
        lod_index = json.load(f)

    chunks = lod_index["chunks"]
    print(f"[INFO] LOD index has {len(chunks)} chunks.")
    print(f"[INFO] LOD levels: {lod_index['lod_levels']}")

    total_L0 = total_L1 = total_L2 = 0

    for meta in chunks:
        ijk = meta["ijk"]
        lod = meta["lod"]

        # 检查 L0/L1/L2 点数单调递减
        n0 = lod["L0"]["count"]
        n1 = lod.get("L1", {}).get("count", n0)
        n2 = lod.get("L2", {}).get("count", n1)

        if not (n0 >= n1 >= n2):
            print(f"[WARN] Non-monotonic LOD for chunk {ijk}: "
                  f"L0={n0}, L1={n1}, L2={n2}")

        total_L0 += n0
        total_L1 += n1
        total_L2 += n2

    print("\n=== Total counts (from index) ===")
    print("  L0 =", total_L0)
    print("  L1 =", total_L1)
    print("  L2 =", total_L2)

    # 也可以 spot check 某个 chunk 的文件行数是否和 index 一致
    sample = chunks[0]
    ijk = sample["ijk"]
    print(f"\n[INFO] Spot check first chunk ijk={ijk}")
    for level in ["L0", "L1", "L2"]:
        info = sample["lod"][level]
        path = CHUNK_DIR / info["filename"]
        arr = np.loadtxt(path)
        if arr.ndim == 1:
            n_file = 1
        else:
            n_file = arr.shape[0]
        print(f"  {level}: index={info['count']}, file_rows={n_file}")

if __name__ == "__main__":
    main()