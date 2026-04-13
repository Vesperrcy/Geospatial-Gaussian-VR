import argparse
import json
import time
from pathlib import Path

# Usage examples for assembling Unity-facing LOD indices:
#
# Indoor random LOD index:
#   python preprocessing/lod_builder.py --chunk_dir data/chunks_indoor_random
#
# Outdoor random LOD index:
#   python preprocessing/lod_builder.py --chunk_dir data/chunks_outdoor_random
#
# Indoor uniform-resolution LOD index:
#   python preprocessing/lod_builder.py --chunk_dir data/chunks_indoor_uniform
#
# Outdoor uniform-resolution LOD index:
#   python preprocessing/lod_builder.py --chunk_dir data/chunks_outdoor_uniform
#
# This script only validates and assembles the existing chunk_*_L*.txt files
# into chunks_lod_index.json. It does not perform random or uniform sampling,
# and it does not generate duplicate L0 files.

# Refs: REF-PAPER-H3DGS, REF-PAPER-OCTREEGS, REF-PAPER-POTREE. See /REFERENCES.md.

CHUNK_DIR = Path("data/chunks_TUMv2")
CHUNK_INDEX_PATH = CHUNK_DIR / "chunks_index.json"
LOD_INDEX_PATH = CHUNK_DIR / "chunks_lod_index.json"


def log_elapsed(step_name: str, t0: float) -> float:
    elapsed = time.perf_counter() - t0
    print(f"[TIME] {step_name}: {elapsed:.2f}s")
    return elapsed


def load_chunk_index(index_path: Path) -> dict:
    if not index_path.exists():
        raise FileNotFoundError(f"Chunk index JSON not found: {index_path}")
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_lod_files(index: dict, chunk_dir: Path) -> tuple[list[dict], dict[str, int]]:
    lod_levels = [int(level) for level in index.get("lod_levels", [])]
    if not lod_levels:
        raise ValueError("Chunk index does not contain lod_levels. Re-run chunking_builder.py with a Gaussian LOD manifest.")

    totals = {f"L{level}": 0 for level in lod_levels}
    chunks_out = []

    for meta in index.get("chunks", []):
        lod_files = list(meta.get("lod_files", []))
        lod_counts = [int(v) for v in meta.get("lod_counts", [])]

        if len(lod_files) != len(lod_levels) or len(lod_counts) != len(lod_levels):
            raise ValueError(f"Chunk {meta.get('ijk')} has inconsistent lod_files/lod_counts length.")

        lod_info = {}
        for i, level in enumerate(lod_levels):
            filename = str(lod_files[i])
            count = int(lod_counts[i])
            if filename:
                path = chunk_dir / filename
                if not path.exists():
                    raise FileNotFoundError(f"LOD chunk file not found: {path}")
            lod_info[f"L{level}"] = {
                "filename": filename,
                "count": count,
            }
            totals[f"L{level}"] += count

        meta_out = dict(meta)
        meta_out["lod"] = lod_info
        meta_out["lod_files"] = lod_files
        meta_out["lod_counts"] = lod_counts
        chunks_out.append(meta_out)

    return chunks_out, totals


def build_lod_index(index: dict, chunks: list[dict], totals: dict[str, int], timing: dict[str, float]) -> dict:
    lod_levels = [int(level) for level in index.get("lod_levels", [])]
    return {
        "stage": "lod_index_assembly",
        "gaussian_lod_manifest": index.get("gaussian_lod_manifest", {}),
        "lod_levels": lod_levels,
        "lod_descriptors": index.get("lod_descriptors", []),
        "origin": index.get("origin", []),
        "chunk_size": index.get("chunk_size", []),
        "grid_shape": index.get("grid_shape", []),
        "num_points": int(sum(totals.values())),
        "num_chunks": len(chunks),
        "lod_point_totals": totals,
        "timing": timing,
        "chunks": chunks,
    }


def main() -> None:
    total_t0 = time.perf_counter()
    timing: dict[str, float] = {}

    print("[INFO] Stage C: Assembling sampling-based LOD index")

    t0 = time.perf_counter()
    index = load_chunk_index(CHUNK_INDEX_PATH)
    timing["load_chunk_index_sec"] = log_elapsed("Load chunk index JSON", t0)

    t0 = time.perf_counter()
    chunks, totals = validate_lod_files(index, CHUNK_DIR)
    timing["validate_lod_files_sec"] = log_elapsed("Validate LOD files", t0)

    timing["total_sec"] = float(time.perf_counter() - total_t0)
    lod_index = build_lod_index(index, chunks, totals, timing)
    LOD_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    with open(LOD_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(lod_index, f, indent=2)
    timing["write_lod_index_sec"] = log_elapsed("Write LOD index JSON", t0)
    timing["total_sec"] = float(time.perf_counter() - total_t0)
    lod_index["timing"] = timing
    with open(LOD_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(lod_index, f, indent=2)

    print("[INFO] LOD point totals:")
    for key in sorted(totals.keys()):
        print(f"  {key}: {totals[key]}")
    print(f"[INFO] Wrote LOD index JSON to: {LOD_INDEX_PATH}")
    print(f"[TIME] Total: {time.perf_counter() - total_t0:.2f}s")
    print("[INFO] Stage C done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_dir", type=str, default=str(CHUNK_DIR), help="Chunk directory")
    parser.add_argument("--chunk_index", type=str, default="", help="Path to chunks_index.json")
    parser.add_argument("--lod_index", type=str, default="", help="Output path for chunks_lod_index.json")
    args = parser.parse_args()

    CHUNK_DIR = Path(args.chunk_dir)
    CHUNK_INDEX_PATH = Path(args.chunk_index) if args.chunk_index else CHUNK_DIR / "chunks_index.json"
    LOD_INDEX_PATH = Path(args.lod_index) if args.lod_index else CHUNK_DIR / "chunks_lod_index.json"

    main()
