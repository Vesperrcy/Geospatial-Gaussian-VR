import json
from pathlib import Path
import numpy as np

LOD_INDEX_PATH = Path("data/navvis_chunks/navvis_chunks_lod_index.json")
CHUNK_DIR = LOD_INDEX_PATH.parent

def main():
    print("=== Debug LOD JSON ===")

    if not LOD_INDEX_PATH.exists():
        raise FileNotFoundError(LOD_INDEX_PATH)

    with open(LOD_INDEX_PATH, "r", encoding="utf-8") as f:
        lod_index = json.load(f)

    chunks = lod_index["chunks"]
    print(f"[INFO] chunks in LOD index: {len(chunks)}\n")

    # 检查前 3 个 chunk
    for meta in chunks[:3]:
        ijk = meta["ijk"]
        lod_files = meta.get("lod_files", {})
        lod_counts = meta.get("lod_counts", {})
        print(f"Chunk ijk={ijk}")
        print(f"  lod_files: {lod_files}")
        print(f"  lod_counts: {lod_counts}")

        # 确认文件存在、行数匹配
        for level_str, fname in lod_files.items():
            path = CHUNK_DIR / fname
            if not path.exists():
                print(f"  [WARN] file not found for L{level_str}: {path}")
                continue

            arr = np.loadtxt(path)
            if arr.ndim == 1:
                n = 1
            else:
                n = arr.shape[0]

            expected = lod_counts.get(level_str, None)
            print(f"    L{level_str}: file={fname}, rows={n}, expected={expected}")
        print()

if __name__ == "__main__":
    main()