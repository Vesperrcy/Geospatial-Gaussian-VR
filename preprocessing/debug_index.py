import json
from pathlib import Path

INDEX_PATH = Path("data/navvis_chunks_Block1/navvis_chunks_index.json")

def main():
    print("=== Debug Chunk Index ===")
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Index not found: {INDEX_PATH}")

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        idx = json.load(f)

    print("[INFO] npz_source:", idx["npz_source"])
    print("[INFO] origin:", idx["origin"])
    print("[INFO] chunk_size:", idx["chunk_size"])
    print("[INFO] grid_shape:", idx["grid_shape"])
    print("[INFO] num_points:", idx["num_points"])
    print("[INFO] num_chunks:", idx["num_chunks"])

    print("\nFirst 5 chunks:")
    for c in idx["chunks"][:5]:
        print("  ijk =", c["ijk"],
              "count =", c["count"],
              "bbox_min =", [round(v,2) for v in c["bbox_min"]],
              "bbox_max =", [round(v,2) for v in c["bbox_max"]],
              "file =", c["filename"])

if __name__ == "__main__":
    main()