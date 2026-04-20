import argparse
import json
import time
from pathlib import Path

import numpy as np

# Usage examples for converting Gaussian LOD manifests into chunk LOD files:
#
# Indoor random chunks:
#   python preprocessing/chunking_builder.py --lod_manifest data/indoor_random_gaussians_final_lod_manifest.json --chunk_dir data/chunks_indoor_random
#
# Outdoor random chunks:
#   python preprocessing/chunking_builder.py --lod_manifest data/outdoor_random_gaussians_final_lod_manifest.json --chunk_dir data/chunks_outdoor_random
#
# Indoor uniform-resolution chunks:
#   python preprocessing/chunking_builder.py --lod_manifest data/indoor_uniform_gaussians_final_lod_manifest.json --chunk_dir data/chunks_indoor_uniform
#
# Outdoor uniform-resolution chunks:
#   python preprocessing/chunking_builder.py --lod_manifest data/outdoor_uniform_gaussians_final_lod_manifest.json --chunk_dir data/chunks_outdoor_uniform
#
#
# After this step, run lod_builder.py on the same chunk directory.



NPZ_PATH = Path("data/TumTLS_v2_gaussians_demo_final.npz")
LOD_MANIFEST_PATH = Path("")
CHUNK_DIR = Path("data/chunks_TUMv2")
CHUNK_PREFIX = "chunk"
CHUNK_SIZE = np.array([10.0, 10.0, 10.0], dtype=np.float32)
EPS = 1e-5


def log_elapsed(step_name: str, t0: float) -> float:
    elapsed = time.perf_counter() - t0
    print(f"[TIME] {step_name}: {elapsed:.2f}s")
    return elapsed


def load_gaussian_npz(npz_path: Path) -> dict[str, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path)
    positions = data["positions"].astype(np.float32)
    colors = data["colors"].astype(np.float32) if "colors" in data else np.empty((positions.shape[0], 0), dtype=np.float32)

    if "cov6" in data:
        cov6 = data["cov6"].astype(np.float32)
    elif "cov0" in data and "cov1" in data:
        cov0 = data["cov0"].astype(np.float32)
        cov1 = data["cov1"].astype(np.float32)
        cov6 = np.stack(
            [cov0[:, 0], cov0[:, 1], cov0[:, 2], cov0[:, 3], cov1[:, 0], cov1[:, 1]],
            axis=1,
        ).astype(np.float32)
    else:
        raise KeyError("NPZ must contain 'cov6' or both 'cov0' and 'cov1' for anisotropic splatting.")

    return {
        "positions": positions,
        "colors": colors,
        "cov6": cov6,
    }


def load_lod_levels(manifest_path: Path, fallback_npz_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    if manifest_path and str(manifest_path) and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        levels = []
        for raw in manifest.get("levels", []):
            level = int(raw["level"])
            levels.append({
                "level": level,
                "label": raw.get("label", f"L{level}"),
                "npz_path": str(raw["npz_path"]),
                "sampling_method": raw.get("sampling_method", manifest.get("sampling_method", "")),
                "sampling_parameter_name": raw.get("sampling_parameter_name", ""),
                "sampling_parameter_value": raw.get("sampling_parameter_value", ""),
                "sampling_parameter_label": raw.get("sampling_parameter_label", ""),
                "gaussian_points": raw.get("num_points", 0),
                "gaussian_timing": raw.get("timing", {}),
                "gaussian_total_sec": raw.get("total_sec", 0.0),
            })
        if not levels:
            raise ValueError(f"LOD manifest has no levels: {manifest_path}")
        return sorted(levels, key=lambda x: int(x["level"])), manifest

    return [
        {
            "level": 0,
            "label": "L0",
            "npz_path": str(fallback_npz_path),
            "sampling_method": "",
            "sampling_parameter_name": "",
            "sampling_parameter_value": "",
            "sampling_parameter_label": "single",
            "gaussian_points": 0,
            "gaussian_timing": {},
            "gaussian_total_sec": 0.0,
        }
    ], {}


def compute_global_grid(levels: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    for level in levels:
        arrays = load_gaussian_npz(Path(str(level["npz_path"])))
        points = arrays["positions"]
        mins.append(points.min(axis=0))
        maxs.append(points.max(axis=0))

    xyz_min = np.min(np.vstack(mins), axis=0)
    xyz_max = np.max(np.vstack(maxs), axis=0)
    extent = xyz_max - xyz_min
    grid = np.ceil((extent + EPS) / CHUNK_SIZE).astype(int)
    grid = np.maximum(grid, 1)
    return xyz_min.astype(np.float32), xyz_max.astype(np.float32), grid.astype(np.int32)


def assign_chunk_indices(points: np.ndarray, origin: np.ndarray, grid: np.ndarray) -> dict[tuple[int, int, int], np.ndarray]:
    rel = points - origin[None, :]
    ijk = np.floor(rel / CHUNK_SIZE[None, :]).astype(np.int32)
    ijk[:, 0] = np.clip(ijk[:, 0], 0, int(grid[0]) - 1)
    ijk[:, 1] = np.clip(ijk[:, 1], 0, int(grid[1]) - 1)
    ijk[:, 2] = np.clip(ijk[:, 2], 0, int(grid[2]) - 1)

    linear = ijk[:, 0] + ijk[:, 1] * int(grid[0]) + ijk[:, 2] * int(grid[0]) * int(grid[1])
    order = np.argsort(linear, kind="stable")
    sorted_linear = linear[order]
    unique, starts = np.unique(sorted_linear, return_index=True)

    chunks: dict[tuple[int, int, int], np.ndarray] = {}
    for pos, key_linear in enumerate(unique):
        start = starts[pos]
        end = starts[pos + 1] if pos + 1 < len(starts) else len(order)
        idx = order[start:end]
        iz = int(key_linear // (int(grid[0]) * int(grid[1])))
        rem = int(key_linear - iz * int(grid[0]) * int(grid[1]))
        iy = int(rem // int(grid[0]))
        ix = int(rem - iy * int(grid[0]))
        chunks[(ix, iy, iz)] = idx
    return chunks


def write_level_chunks(level: dict[str, object],
                       origin: np.ndarray,
                       grid: np.ndarray,
                       chunk_registry: dict[tuple[int, int, int], dict[str, object]]) -> dict[str, object]:
    arrays = load_gaussian_npz(Path(str(level["npz_path"])))
    points = arrays["positions"]
    colors = arrays["colors"]
    cov6 = arrays["cov6"]
    if colors.size == 0:
        colors = np.tile(np.array([[0.7, 0.7, 0.7]], dtype=np.float32), (points.shape[0], 1))

    level_num = int(level["level"])
    chunks = assign_chunk_indices(points, origin, grid)
    total_written = 0

    for (ix, iy, iz), idx in chunks.items():
        p_chunk = points[idx]
        c_chunk = colors[idx]
        cov6_chunk = cov6[idx]
        mat = np.hstack([p_chunk, c_chunk, cov6_chunk])
        fname = f"{CHUNK_PREFIX}_{ix}_{iy}_{iz}_L{level_num}.txt"
        np.savetxt(CHUNK_DIR / fname, mat, fmt="%.6f")

        bbox_min = p_chunk.min(axis=0)
        bbox_max = p_chunk.max(axis=0)
        center = 0.5 * (bbox_min + bbox_max)

        meta = chunk_registry.setdefault(
            (ix, iy, iz),
            {
                "ijk": [int(ix), int(iy), int(iz)],
                "bbox_min": bbox_min.tolist(),
                "bbox_max": bbox_max.tolist(),
                "center": center.tolist(),
                "lod": {},
            },
        )

        old_min = np.asarray(meta["bbox_min"], dtype=np.float32)
        old_max = np.asarray(meta["bbox_max"], dtype=np.float32)
        new_min = np.minimum(old_min, bbox_min)
        new_max = np.maximum(old_max, bbox_max)
        meta["bbox_min"] = new_min.tolist()
        meta["bbox_max"] = new_max.tolist()
        meta["center"] = (0.5 * (new_min + new_max)).tolist()
        meta["lod"][f"L{level_num}"] = {
            "filename": fname,
            "count": int(p_chunk.shape[0]),
            "sampling_method": level.get("sampling_method", ""),
            "sampling_parameter_name": level.get("sampling_parameter_name", ""),
            "sampling_parameter_value": level.get("sampling_parameter_value", ""),
            "sampling_parameter_label": level.get("sampling_parameter_label", ""),
        }
        total_written += int(p_chunk.shape[0])

    return {
        "level": level_num,
        "num_points": int(points.shape[0]),
        "num_chunks": int(len(chunks)),
        "written_points": int(total_written),
    }


def build_chunk_index(levels: list[dict[str, object]],
                      manifest: dict[str, object],
                      origin: np.ndarray,
                      xyz_max: np.ndarray,
                      grid: np.ndarray,
                      chunk_registry: dict[tuple[int, int, int], dict[str, object]],
                      timing: dict[str, float]) -> dict[str, object]:
    lod_levels = [int(level["level"]) for level in levels]
    chunks = []
    for key in sorted(chunk_registry.keys()):
        meta = chunk_registry[key]
        lod_files = []
        lod_counts = []
        for level in lod_levels:
            lod_meta = meta["lod"].get(f"L{level}", {})
            lod_files.append(str(lod_meta.get("filename", "")))
            lod_counts.append(int(lod_meta.get("count", 0)))
        first_file = next((name for name in lod_files if name), "")
        first_count = next((count for count in lod_counts if count > 0), 0)
        meta_out = dict(meta)
        meta_out["filename"] = first_file
        meta_out["count"] = int(first_count)
        meta_out["lod_files"] = lod_files
        meta_out["lod_counts"] = lod_counts
        chunks.append(meta_out)

    return {
        "stage": "chunking",
        "gaussian_lod_manifest": manifest,
        "lod_levels": lod_levels,
        "lod_descriptors": levels,
        "origin": origin.tolist(),
        "chunk_size": CHUNK_SIZE.tolist(),
        "grid_shape": [int(grid[0]), int(grid[1]), int(grid[2])],
        "xyz_max": xyz_max.tolist(),
        "num_points": int(sum(int(level.get("gaussian_points") or 0) for level in levels)),
        "num_chunks": len(chunks),
        "timing": timing,
        "chunks": chunks,
    }


def main() -> None:
    total_t0 = time.perf_counter()
    timing: dict[str, float] = {}

    print("[INFO] Stage B: Building sampling-based LOD chunks")
    levels, manifest = load_lod_levels(LOD_MANIFEST_PATH, NPZ_PATH)
    print(f"[INFO] Loaded {len(levels)} LOD level input(s).")

    t0 = time.perf_counter()
    origin, xyz_max, grid = compute_global_grid(levels)
    timing["compute_global_grid_sec"] = log_elapsed("Compute global chunk grid", t0)
    print(f"[INFO] Global grid shape: {int(grid[0])} {int(grid[1])} {int(grid[2])}")

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    chunk_registry: dict[tuple[int, int, int], dict[str, object]] = {}
    level_summaries = []

    for level in levels:
        level_t0 = time.perf_counter()
        print(
            "[INFO] Writing chunks for "
            f"L{int(level['level'])} ({level.get('sampling_method', '')}, "
            f"{level.get('sampling_parameter_label', '')})"
        )
        summary = write_level_chunks(level, origin, grid, chunk_registry)
        summary["chunking_sec"] = log_elapsed(f"Write chunks for L{int(level['level'])}", level_t0)
        level_summaries.append(summary)

    timing["total_chunk_file_write_sec"] = float(sum(float(s["chunking_sec"]) for s in level_summaries))

    timing["total_sec"] = float(time.perf_counter() - total_t0)
    index = build_chunk_index(levels, manifest, origin, xyz_max, grid, chunk_registry, timing)
    index["level_summaries"] = level_summaries

    t0 = time.perf_counter()
    index_path = CHUNK_DIR / "chunks_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    timing["write_index_sec"] = log_elapsed("Write chunk index JSON", t0)
    timing["total_sec"] = float(time.perf_counter() - total_t0)
    index["timing"] = timing
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"[INFO] Wrote chunk index JSON to: {index_path}")
    print(f"[INFO] Non-empty chunks across all levels: {len(chunk_registry)}")
    print(f"[TIME] Total: {time.perf_counter() - total_t0:.2f}s")
    print("[INFO] Stage B done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default=str(NPZ_PATH), help="Fallback single Gaussian NPZ input")
    parser.add_argument("--lod_manifest", type=str, default="", help="Gaussian LOD manifest from gaussian_builder.py")
    parser.add_argument("--chunk_dir", type=str, default=str(CHUNK_DIR), help="Chunk output directory")
    parser.add_argument("--chunk_prefix", type=str, default=CHUNK_PREFIX, help="Chunk filename prefix")
    parser.add_argument("--chunk_size", type=float, nargs=3, default=CHUNK_SIZE.tolist(), help="Chunk size dx dy dz")
    args = parser.parse_args()

    NPZ_PATH = Path(args.npz)
    LOD_MANIFEST_PATH = Path(args.lod_manifest) if args.lod_manifest else Path("")
    CHUNK_DIR = Path(args.chunk_dir)
    CHUNK_PREFIX = str(args.chunk_prefix)
    CHUNK_SIZE = np.array(args.chunk_size, dtype=np.float32)

    main()
