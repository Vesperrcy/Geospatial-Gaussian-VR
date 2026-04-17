import numpy as np
import open3d as o3d
from pathlib import Path
import argparse
import json
from typing import Optional
import os
import platform
import subprocess
import time
import gc

# Usage examples for sampling-based Gaussian LOD generation:
#
# Indoor random LOD pyramid:
#   python preprocessing/gaussian_builder.py --input data/IndoorOfficeData/IndoorMergedClouds.ply --output data/indoor_random_gaussians.npz --preset final --sampling_method random --lod_pyramid
#
# Outdoor random LOD pyramid:
#   python preprocessing/gaussian_builder.py --input data/TumTLS_v2.ply --output data/outdoor_random_gaussians.npz --preset final --sampling_method random --lod_pyramid
#
# Indoor uniform-resolution LOD pyramid:
#   python preprocessing/gaussian_builder.py --input data/IndoorOfficeData/IndoorMergedClouds.ply --output data/indoor_uniform_gaussians.npz --preset final --sampling_method uniform --lod_pyramid
#
# Outdoor uniform-resolution LOD pyramid:
#   python preprocessing/gaussian_builder.py --input data/TumTLS_v2.ply --output data/outdoor_uniform_gaussians.npz --preset final --sampling_method uniform --lod_pyramid
#
# Smaller random test on memory-limited machines:
#   python preprocessing/gaussian_builder.py --input data/TumTLS_v2.ply --output data/outdoor_random_test_gaussians.npz --preset final --sampling_method random --lod_specs L0=1M,L1=100K,L2=10K --no_autotune
#
# Output manifests used by chunking_builder.py:
#   data/indoor_random_gaussians_final_lod_manifest.json
#   data/outdoor_random_gaussians_final_lod_manifest.json
#   data/indoor_uniform_gaussians_final_lod_manifest.json
#   data/outdoor_uniform_gaussians_final_lod_manifest.json

# Optional (fast CPU KNN)
try:
    from scipy.spatial import cKDTree  # type: ignore
    _HAS_SCIPY = True
except Exception:
    cKDTree = None
    _HAS_SCIPY = False

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:
    psutil = None
    _HAS_PSUTIL = False

# ==============================
# Paths
# ==============================
INPUT_PATH = Path("data/IndoorOfficeData/IndoorMergedClouds.ply")
OUTPUT_PATH = Path("data/Indoordata_demo.npz")

# ==============================
# Preset switch
# ==============================
PRESET_DEFAULT = "verify"
PRESET = PRESET_DEFAULT  # overwritten by apply_preset()

# ==============================
# Logging
# ==============================
VERBOSE = True
LOG_EVERY = 50000

def log(msg: str):
    if VERBOSE:
        print(msg)

# ==============================
# PLY color attributes
# ==============================
USE_INTENSITY_AUX_COLOR = True
INTENSITY_COLOR_STRENGTH = 0.25
INTENSITY_LOW_PERCENTILE = 2.0
INTENSITY_HIGH_PERCENTILE = 98.0
INTENSITY_PERCENTILE_SAMPLE = 1_000_000

# ==============================
# Sampling strategy
# ==============================
# Refs: REF-PAPER-3DGS, REF-PAPER-H3DGS, REF-LIB-OPEN3D. See /REFERENCES.md.
SAMPLE_MODE = "random"   # random | uniform
MAX_GAUSSIANS = -1
MAX_GAUSSIANS_AFTER_SAMPLING = MAX_GAUSSIANS
UNIFORM_RESOLUTION = 0.05
GAUSSIAN_LOD_LEVELS: Optional[list[dict[str, object]]] = None

# KNN
K_NEIGHBORS_ISO = 8
K_NEIGHBORS_NORMAL = 24
K_NEIGHBORS_TANGENT = 64
KNN_BACKEND = "auto"   # auto | cuda | cpu
KNN_QUERY_BATCH = 262144
FRAME_TRANSFORM_BACKEND = "auto"  # auto | cuda | cpu
FRAME_TRANSFORM_BLOCK_SIZE = 262144
CKDTREE_WORKERS = -1
KNN_CACHE_MODE = "auto"  # auto | full | iso | none

# Auto profiling / autotune
AUTO_PROFILE = True
AUTO_TUNE = True
AUTO_TUNE_PROBE_POINTS = 30000
MAX_CACHE_RAM_FRACTION = 0.22
UNIFIED_MEMORY_CACHE_RAM_FRACTION = 0.14
VERIFY_AVAILABLE_RAM_FRACTION = 0.60
VERIFY_UNIFIED_AVAILABLE_RAM_FRACTION = 0.45
VERIFY_MEMORY_SAFETY_FACTOR = 1.30

USER_OVERRIDE_MAX_POINTS = False
USER_OVERRIDE_ANISO_BLOCK = False
USER_OVERRIDE_KNN_BACKEND = False
USER_OVERRIDE_KNN_QUERY_BATCH = False
USER_OVERRIDE_FRAME_BACKEND = False
USER_OVERRIDE_FRAME_BLOCK = False
USER_OVERRIDE_CKDTREE_WORKERS = False
USER_OVERRIDE_KNN_CACHE_MODE = False

# ==============================
# Normals strategy (B): prefer file normals
# ==============================
USE_FILE_NORMALS_FIRST = True
ALLOW_O3D_ESTIMATE_NORMALS = True

# ==============================
# Batch anisotropy (million+ points)
# ==============================
USE_BATCH_ANISO = True
ANISO_BLOCK_SIZE = 50000
ANISO_MIN_BLOCK_SIZE = 4096
DT_USE_PARTITION = True

MIN_NEIGHBORS_NORMAL = 10
MIN_NEIGHBORS_TANGENT = 16
KNN_DISTANCE_CLIP_ALPHA = 2.5
RADIUS_FACTOR_FROM_ISO = 6.0

# --- normal thickness ---
SIGMA_N_MODE = "fixed"   # "fixed" | "relative"
SIGMA_N_FIXED = 0.008
SIGMA_N_REL_FACTOR = 0.25
SIGMA_N_MIN = 0.004
SIGMA_N_MAX = 0.015

# --- tangent weighting ---
USE_TANGENT_WEIGHTING = True
TANGENT_WEIGHT_SIGMA_FACTOR = 1.6
TANGENT_WEIGHT_SIGMA_W_MIN = 0.02
TANGENT_WEIGHT_SIGMA_W_MAX = 0.25

# --- tangent regularization ---
TANGENT_REL_MIN = 0.35
TANGENT_REL_MAX = 1.1

# --- dt -> st_base ---
DT_PERCENTILE = 20
DT_TO_SIGMA_FACTOR = 0.75

# aspect ratio clamp
AR_MIN = 1.0
AR_MAX = 3.0

# clamp
USE_ADAPTIVE_CLAMP = True
S_MIN = 0.02
S_MAX = 0.0875

ADAPTIVE_Q_MIN = 2
ADAPTIVE_Q_MAX = 95
ADAPTIVE_MIN_FACTOR = 0.35
ADAPTIVE_MAX_FACTOR = 0.9

ABS_S_MIN = 0.005
ABS_S_MAX = 0.50

# ---- S_MAX policy ----
S_MAX_POLICY = "max"   # "siso" | "dt" | "max"
USE_DT_BASED_SMAX = True
DT_SMAX_PERCENTILE = 90
DT_SMAX_FACTOR = 0.75
DT_STATS_MAX_POINTS = 50000

# ==============================
# Frame transform (ENU -> Unity)
# ==============================
APPLY_UNITY_FRAME = True
M_ENU_TO_UNITY = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0],
    [ 0.0,  1.0,  0.0],
], dtype=np.float32)

# ==============================
# Recenter
# ==============================
APPLY_RECENTER = True
RECENTER_MODE = "bbox_center"

# ==============================
# Pre-shift large coordinates (UTM float precision fix)
# ==============================
ENABLE_PRE_SHIFT = True
PRE_SHIFT_MODE = "bbox_center"

# ==============================
# Chunk writing (integrated)
# ==============================
WRITE_CHUNKS = False
CHUNK_DIR = Path("data/chunks_Indoordata")
CHUNK_PREFIX = "chunk"
CHUNK_SIZE = np.array([10.0, 10.0, 10.0], dtype=np.float32)
CHUNK_EPS = 1e-5


def apply_preset(preset: str):
    global PRESET
    global MAX_GAUSSIANS, MAX_GAUSSIANS_AFTER_SAMPLING
    global TANGENT_WEIGHT_SIGMA_FACTOR, TANGENT_WEIGHT_SIGMA_W_MAX
    global DT_PERCENTILE, DT_TO_SIGMA_FACTOR
    global TANGENT_REL_MAX
    global S_MAX_POLICY, DT_SMAX_PERCENTILE, DT_SMAX_FACTOR
    global SIGMA_N_FIXED

    p = (preset or PRESET_DEFAULT).lower().strip()
    PRESET = p

    if p == "verify":
        MAX_GAUSSIANS = 4_433_050
        MAX_GAUSSIANS_AFTER_SAMPLING = MAX_GAUSSIANS

        TANGENT_WEIGHT_SIGMA_FACTOR = 1.0
        TANGENT_WEIGHT_SIGMA_W_MAX = 0.14

        DT_PERCENTILE = 10
        DT_TO_SIGMA_FACTOR = 0.35

        TANGENT_REL_MAX = 0.80

        S_MAX_POLICY = "siso"
        DT_SMAX_PERCENTILE = 90
        DT_SMAX_FACTOR = 0.70

        SIGMA_N_FIXED = 0.006

    elif p == "final":
        MAX_GAUSSIANS = -1
        MAX_GAUSSIANS_AFTER_SAMPLING = -1

        TANGENT_WEIGHT_SIGMA_FACTOR = 1.6
        TANGENT_WEIGHT_SIGMA_W_MAX = 0.25

        DT_PERCENTILE = 20
        DT_TO_SIGMA_FACTOR = 0.75

        TANGENT_REL_MAX = 1.10

        S_MAX_POLICY = "max"
        DT_SMAX_PERCENTILE = 90
        DT_SMAX_FACTOR = 0.75

        SIGMA_N_FIXED = 0.008

    else:
        raise ValueError(f"Unknown preset: {preset}")

    log(f"[INFO] Preset applied: {PRESET}")
    log(f"[INFO] Sampling defaults: mode={SAMPLE_MODE}, max_points={MAX_GAUSSIANS}, uniform_resolution={UNIFORM_RESOLUTION:.4f} m")
    log(f"[INFO] dt->st: DT_PERCENTILE={DT_PERCENTILE}, DT_TO_SIGMA_FACTOR={DT_TO_SIGMA_FACTOR:.2f}; "
        f"tangent sigma_factor={TANGENT_WEIGHT_SIGMA_FACTOR:.2f}, w_max={TANGENT_WEIGHT_SIGMA_W_MAX:.2f}")
    log(f"[INFO] S_MAX_POLICY={S_MAX_POLICY}, SIGMA_N_FIXED={SIGMA_N_FIXED:.3f}")


def is_apple_silicon_mac() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64")


def detect_total_memory_gb() -> float:
    if _HAS_PSUTIL:
        try:
            return float(psutil.virtual_memory().total) / (1024.0 ** 3)
        except Exception:
            pass

    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return float(int(out)) / (1024.0 ** 3)
        except Exception:
            pass

    return 8.0


def detect_available_memory_gb() -> float:
    if _HAS_PSUTIL:
        try:
            return float(psutil.virtual_memory().available) / (1024.0 ** 3)
        except Exception:
            pass

    total_gb = detect_total_memory_gb()
    return total_gb * 0.5


def detect_device_profile() -> dict[str, object]:
    cpu_count = int(os.cpu_count() or 8)
    mem_gb = detect_total_memory_gb()
    has_cuda = False
    try:
        has_cuda = bool(o3d.core.cuda.is_available())
    except Exception:
        has_cuda = False

    apple_silicon = is_apple_silicon_mac()
    unified_memory = apple_silicon

    if has_cuda:
        tier = "gpu"
    elif mem_gb >= 48 and cpu_count >= 16:
        tier = "cpu_xl"
    elif mem_gb >= 24 and cpu_count >= 8:
        tier = "cpu_l"
    elif mem_gb >= 16 and cpu_count >= 6:
        tier = "cpu_m"
    else:
        tier = "cpu_s"

    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": cpu_count,
        "mem_gb": float(mem_gb),
        "avail_mem_gb": float(detect_available_memory_gb()),
        "has_cuda": has_cuda,
        "apple_silicon": apple_silicon,
        "unified_memory": unified_memory,
        "tier": tier,
    }


def estimate_knn_cache_bytes(num_points: int, k: int) -> int:
    n = int(max(0, num_points))
    kk = int(max(0, k))
    bytes_per_neighbor = np.dtype(np.int32).itemsize + np.dtype(np.float32).itemsize
    return n * kk * bytes_per_neighbor


def estimate_pipeline_bytes_per_point(cache_mode: str,
                                      include_colors: bool = True,
                                      include_normals: bool = True) -> float:
    bytes_per_point = 0.0

    # Resident arrays across the full pipeline.
    bytes_per_point += 12.0  # points
    bytes_per_point += 4.0   # scales_iso
    bytes_per_point += 12.0  # scales_aniso
    bytes_per_point += 36.0  # rotations
    bytes_per_point += 24.0  # cov6
    bytes_per_point += 16.0  # cov0
    bytes_per_point += 16.0  # cov1
    bytes_per_point += 4.0   # opacity

    if include_colors:
        bytes_per_point += 12.0
    if include_normals:
        bytes_per_point += 12.0

    # Spatial indexing / library overhead.
    bytes_per_point += 24.0 if _HAS_SCIPY else 16.0

    cm = (cache_mode or "none").lower().strip()
    if cm == "full":
        bytes_per_point += float(estimate_knn_cache_bytes(1, max(K_NEIGHBORS_ISO, K_NEIGHBORS_NORMAL, K_NEIGHBORS_TANGENT) + 1))
    elif cm == "iso":
        bytes_per_point += float(estimate_knn_cache_bytes(1, K_NEIGHBORS_ISO + 1))

    return float(bytes_per_point * VERIFY_MEMORY_SAFETY_FACTOR)


def estimate_verify_max_points(total_points: int,
                               profile: dict[str, object],
                               cache_mode: str) -> dict[str, object]:
    available_gb = float(profile.get("avail_mem_gb", detect_available_memory_gb()))
    unified_memory = bool(profile.get("unified_memory", False))
    usable_ratio = VERIFY_UNIFIED_AVAILABLE_RAM_FRACTION if unified_memory else VERIFY_AVAILABLE_RAM_FRACTION
    usable_bytes = max(0.25, available_gb * usable_ratio) * (1024.0 ** 3)

    bytes_per_point = estimate_pipeline_bytes_per_point(cache_mode)
    max_points = int(max(1, usable_bytes // max(bytes_per_point, 1.0)))
    max_points = int(min(int(total_points), max_points))

    return {
        "available_gb": available_gb,
        "usable_gb": usable_bytes / (1024.0 ** 3),
        "bytes_per_point": bytes_per_point,
        "max_points": max_points,
        "cache_mode": str(cache_mode),
    }


def choose_max_points_for_profile(profile: dict[str, object], preset: str) -> int:
    p = (preset or PRESET_DEFAULT).lower().strip()
    if p == "final":
        return -1

    mem_gb = float(profile["mem_gb"])
    tier = str(profile["tier"])
    has_cuda = bool(profile["has_cuda"])
    unified_memory = bool(profile["unified_memory"])

    if has_cuda and mem_gb >= 48.0:
        verify_max, final_max = 4_433_050, -1
    elif tier == "gpu":
        verify_max, final_max = 4_433_050, 6_000_000
    elif tier == "cpu_xl":
        verify_max, final_max = 3_800_000, 5_000_000
    elif tier == "cpu_l":
        verify_max, final_max = 2_400_000, 3_400_000
    elif tier == "cpu_m":
        verify_max, final_max = 1_400_000, 2_200_000
    else:
        verify_max, final_max = 850_000, 1_300_000

    if unified_memory:
        verify_max = int(verify_max * 0.9)
        if final_max > 0:
            final_max = int(final_max * 0.85)

    return int(verify_max)


def choose_runtime_defaults(profile: dict[str, object], effective_points: int) -> dict[str, object]:
    mem_gb = float(profile["mem_gb"])
    cpu_count = int(profile["cpu_count"])
    has_cuda = bool(profile["has_cuda"])
    unified_memory = bool(profile["unified_memory"])

    if has_cuda:
        knn_backend = "cuda"
        frame_backend = "cuda"
        knn_query_batch = 524288 if mem_gb >= 32.0 else 262144
        frame_block = 524288 if mem_gb >= 32.0 else 262144
        aniso_block = 50000 if mem_gb >= 32.0 else 28000
        workers = max(1, min(12, cpu_count))
    elif unified_memory:
        knn_backend = "cpu"
        frame_backend = "cpu"
        knn_query_batch = 131072 if mem_gb >= 16.0 else 65536
        frame_block = 131072 if mem_gb >= 16.0 else 65536
        aniso_block = 18000 if mem_gb >= 16.0 else 10000
        workers = max(1, min(8, max(2, cpu_count - 2)))
    else:
        knn_backend = "cpu"
        frame_backend = "cpu"
        knn_query_batch = 262144 if mem_gb >= 24.0 else 131072
        frame_block = 262144 if mem_gb >= 24.0 else 131072
        aniso_block = 35000 if mem_gb >= 24.0 else 18000
        workers = max(1, min(12, max(2, cpu_count - 2)))

    k_full = int(max(K_NEIGHBORS_ISO, K_NEIGHBORS_NORMAL, K_NEIGHBORS_TANGENT) + 1)
    k_iso = int(K_NEIGHBORS_ISO + 1)
    cache_ratio = UNIFIED_MEMORY_CACHE_RAM_FRACTION if unified_memory else MAX_CACHE_RAM_FRACTION
    budget_bytes = float(mem_gb) * cache_ratio * (1024.0 ** 3)
    full_cache_bytes = float(estimate_knn_cache_bytes(effective_points, k_full))
    iso_cache_bytes = float(estimate_knn_cache_bytes(effective_points, k_iso))

    if full_cache_bytes <= budget_bytes:
        cache_mode = "full"
    elif iso_cache_bytes <= budget_bytes:
        cache_mode = "iso"
    else:
        cache_mode = "none"

    return {
        "knn_backend": knn_backend,
        "knn_query_batch": int(knn_query_batch),
        "frame_backend": frame_backend,
        "frame_block": int(frame_block),
        "aniso_block": int(aniso_block),
        "workers": int(workers),
        "cache_mode": cache_mode,
        "full_cache_gb": full_cache_bytes / (1024.0 ** 3),
        "iso_cache_gb": iso_cache_bytes / (1024.0 ** 3),
        "cache_budget_gb": budget_bytes / (1024.0 ** 3),
    }


def apply_device_profile_if_needed(total_points: int) -> dict[str, object]:
    global MAX_GAUSSIANS, MAX_GAUSSIANS_AFTER_SAMPLING
    global ANISO_BLOCK_SIZE, KNN_BACKEND, KNN_QUERY_BATCH
    global FRAME_TRANSFORM_BACKEND, FRAME_TRANSFORM_BLOCK_SIZE
    global CKDTREE_WORKERS, KNN_CACHE_MODE

    profile = detect_device_profile()
    effective_points = int(total_points)

    if AUTO_PROFILE:
        if not USER_OVERRIDE_MAX_POINTS:
            max_points = choose_max_points_for_profile(profile, PRESET)
            MAX_GAUSSIANS = int(max_points)
            MAX_GAUSSIANS_AFTER_SAMPLING = int(max_points)

        effective_points = int(total_points if MAX_GAUSSIANS <= 0 else min(total_points, MAX_GAUSSIANS))
        defaults = choose_runtime_defaults(profile, effective_points)

        if not USER_OVERRIDE_KNN_BACKEND:
            KNN_BACKEND = str(defaults["knn_backend"])
        if not USER_OVERRIDE_KNN_QUERY_BATCH:
            KNN_QUERY_BATCH = int(defaults["knn_query_batch"])
        if not USER_OVERRIDE_FRAME_BACKEND:
            FRAME_TRANSFORM_BACKEND = str(defaults["frame_backend"])
        if not USER_OVERRIDE_FRAME_BLOCK:
            FRAME_TRANSFORM_BLOCK_SIZE = int(defaults["frame_block"])
        if not USER_OVERRIDE_ANISO_BLOCK:
            ANISO_BLOCK_SIZE = int(defaults["aniso_block"])
        if not USER_OVERRIDE_CKDTREE_WORKERS:
            CKDTREE_WORKERS = int(defaults["workers"])
        if not USER_OVERRIDE_KNN_CACHE_MODE:
            KNN_CACHE_MODE = str(defaults["cache_mode"])

        if PRESET == "verify" and not USER_OVERRIDE_MAX_POINTS:
            verify_test = estimate_verify_max_points(total_points, profile, KNN_CACHE_MODE)
            MAX_GAUSSIANS = int(verify_test["max_points"])
            MAX_GAUSSIANS_AFTER_SAMPLING = int(verify_test["max_points"])
            effective_points = int(MAX_GAUSSIANS)
            defaults = choose_runtime_defaults(profile, effective_points)
            if not USER_OVERRIDE_KNN_BACKEND:
                KNN_BACKEND = str(defaults["knn_backend"])
            if not USER_OVERRIDE_KNN_QUERY_BATCH:
                KNN_QUERY_BATCH = int(defaults["knn_query_batch"])
            if not USER_OVERRIDE_FRAME_BACKEND:
                FRAME_TRANSFORM_BACKEND = str(defaults["frame_backend"])
            if not USER_OVERRIDE_FRAME_BLOCK:
                FRAME_TRANSFORM_BLOCK_SIZE = int(defaults["frame_block"])
            if not USER_OVERRIDE_ANISO_BLOCK:
                ANISO_BLOCK_SIZE = int(defaults["aniso_block"])
            if not USER_OVERRIDE_CKDTREE_WORKERS:
                CKDTREE_WORKERS = int(defaults["workers"])
            if not USER_OVERRIDE_KNN_CACHE_MODE:
                KNN_CACHE_MODE = str(defaults["cache_mode"])
                verify_test = estimate_verify_max_points(total_points, profile, KNN_CACHE_MODE)
                MAX_GAUSSIANS = int(verify_test["max_points"])
                MAX_GAUSSIANS_AFTER_SAMPLING = int(verify_test["max_points"])
                effective_points = int(MAX_GAUSSIANS)
                defaults = choose_runtime_defaults(profile, effective_points)
                if not USER_OVERRIDE_KNN_BACKEND:
                    KNN_BACKEND = str(defaults["knn_backend"])
                if not USER_OVERRIDE_KNN_QUERY_BATCH:
                    KNN_QUERY_BATCH = int(defaults["knn_query_batch"])
                if not USER_OVERRIDE_FRAME_BACKEND:
                    FRAME_TRANSFORM_BACKEND = str(defaults["frame_backend"])
                if not USER_OVERRIDE_FRAME_BLOCK:
                    FRAME_TRANSFORM_BLOCK_SIZE = int(defaults["frame_block"])
                if not USER_OVERRIDE_ANISO_BLOCK:
                    ANISO_BLOCK_SIZE = int(defaults["aniso_block"])
                if not USER_OVERRIDE_CKDTREE_WORKERS:
                    CKDTREE_WORKERS = int(defaults["workers"])

        if (not _HAS_SCIPY) and KNN_CACHE_MODE != "full" and not USER_OVERRIDE_KNN_CACHE_MODE:
            k_full = int(max(K_NEIGHBORS_ISO, K_NEIGHBORS_NORMAL, K_NEIGHBORS_TANGENT) + 1)
            cache_ratio = UNIFIED_MEMORY_CACHE_RAM_FRACTION if bool(profile["unified_memory"]) else MAX_CACHE_RAM_FRACTION
            budget_bytes = float(profile["mem_gb"]) * cache_ratio * (1024.0 ** 3)
            max_points_for_full = int(max(1, budget_bytes // max(1, estimate_knn_cache_bytes(1, k_full))))
            if not USER_OVERRIDE_MAX_POINTS and MAX_GAUSSIANS > 0:
                adjusted = min(int(MAX_GAUSSIANS), max_points_for_full)
                if adjusted < MAX_GAUSSIANS:
                    MAX_GAUSSIANS = adjusted
                    MAX_GAUSSIANS_AFTER_SAMPLING = adjusted
                    effective_points = min(effective_points, adjusted)
                    log(f"[WARN] SciPy unavailable; reducing max_points to {adjusted} so full KNN cache stays within budget.")
            KNN_CACHE_MODE = "full"

        log("[INFO] Device profile:")
        log(f"  system={profile['system']}, machine={profile['machine']}, tier={profile['tier']}")
        log(f"  cpu_count={profile['cpu_count']}, mem_gb={float(profile['mem_gb']):.1f}, avail_mem_gb={float(profile['avail_mem_gb']):.1f}, cuda={bool(profile['has_cuda'])}")
        log("[INFO] Runtime defaults:")
        log(f"  max_points={MAX_GAUSSIANS}, knn_backend={KNN_BACKEND}, knn_query_batch={KNN_QUERY_BATCH}")
        log(f"  frame_backend={FRAME_TRANSFORM_BACKEND}, frame_block={FRAME_TRANSFORM_BLOCK_SIZE}")
        log(f"  aniso_block={ANISO_BLOCK_SIZE}, ckdtree_workers={CKDTREE_WORKERS}, knn_cache_mode={KNN_CACHE_MODE}")
        log(f"  cache_budget={float(defaults['cache_budget_gb']):.2f} GB, "
            f"full_cache_est={float(defaults['full_cache_gb']):.2f} GB, iso_cache_est={float(defaults['iso_cache_gb']):.2f} GB")
        if PRESET == "verify" and not USER_OVERRIDE_MAX_POINTS:
            verify_test = estimate_verify_max_points(total_points, profile, KNN_CACHE_MODE)
            log("[INFO] Verify max-point test:")
            log(f"  cache_mode={verify_test['cache_mode']}, usable_mem={float(verify_test['usable_gb']):.2f} GB")
            log(f"  est_bytes_per_point={float(verify_test['bytes_per_point']):.1f} B, max_points={int(verify_test['max_points'])}")

    return profile


def _tree_query(tree, p: np.ndarray, k: int, workers: int):
    try:
        return tree.query(p, k=k, workers=int(workers))
    except TypeError:
        try:
            return tree.query(p, k=k, n_jobs=int(workers))
        except TypeError:
            return tree.query(p, k=k)


def build_probe_indices(num_points: int, probe_points: int) -> np.ndarray:
    n = int(num_points)
    m = int(min(max(1, probe_points), n))
    if m >= n:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, m, dtype=np.int64)


def autotune_knn_runtime(points_probe: np.ndarray,
                         k: int,
                         profile: dict[str, object]) -> tuple[str, int, np.ndarray, np.ndarray]:
    backend_candidates = [str(KNN_BACKEND)]
    if not USER_OVERRIDE_KNN_BACKEND:
        if bool(profile["has_cuda"]):
            backend_candidates = ["cuda", "cpu"]
        else:
            backend_candidates = ["cpu"]

    batch_candidates = [int(KNN_QUERY_BATCH)]
    if not USER_OVERRIDE_KNN_QUERY_BATCH:
        base = int(max(4096, KNN_QUERY_BATCH))
        batch_candidates = sorted({
            max(4096, base // 2),
            base,
            max(4096, base * 2),
        })

    best = None
    best_idx = None
    best_dist2 = None
    for backend in backend_candidates:
        for batch in batch_candidates:
            t0 = time.perf_counter()
            try:
                idx, dist2 = batch_knn_search(points_probe, k, backend=backend, query_batch=batch)
                dt = max(1e-6, time.perf_counter() - t0)
                speed = points_probe.shape[0] / dt
                log(f"[INFO] Auto-tune KNN: backend={backend}, batch={batch}, speed={speed:.1f} pts/s")
                if (best is None) or (speed > best[0]):
                    best = (speed, backend, batch)
                    best_idx = idx
                    best_dist2 = dist2
            except Exception as e:
                log(f"[WARN] Auto-tune KNN failed for backend={backend}, batch={batch}: {e}")

    if best is None or best_idx is None or best_dist2 is None:
        raise RuntimeError("KNN auto-tune could not find a working configuration.")

    _, backend_best, batch_best = best
    log(f"[INFO] Auto-tune KNN selected: backend={backend_best}, batch={batch_best}")
    return str(backend_best), int(batch_best), best_idx, best_dist2


def autotune_aniso_runtime(points_probe: np.ndarray,
                           scales_probe: np.ndarray,
                           normals_probe: np.ndarray,
                           profile: dict[str, object],
                           knn_probe_full_idx: Optional[np.ndarray]) -> tuple[int, int, np.ndarray, np.ndarray]:
    block_candidates = [int(ANISO_BLOCK_SIZE)]
    if not USER_OVERRIDE_ANISO_BLOCK:
        base = int(max(ANISO_MIN_BLOCK_SIZE, ANISO_BLOCK_SIZE))
        block_candidates = sorted({
            max(ANISO_MIN_BLOCK_SIZE, base * 3 // 4),
            base,
            max(ANISO_MIN_BLOCK_SIZE, base * 5 // 4),
        })

    worker_candidates = [int(CKDTREE_WORKERS)]
    use_cached_knn = knn_probe_full_idx is not None
    if (not use_cached_knn) and (not USER_OVERRIDE_CKDTREE_WORKERS):
        cpu_count = int(profile["cpu_count"])
        base_w = max(1, int(CKDTREE_WORKERS))
        worker_candidates = sorted({
            max(1, base_w - 2),
            base_w,
            max(1, min(cpu_count, base_w + 2)),
        })

    best = None
    best_rot = None
    best_cov6 = None
    for block in block_candidates:
        for workers in worker_candidates:
            t0 = time.perf_counter()
            try:
                _s, rot, cov6 = compute_normal_aligned_gaussians_batched(
                    points=points_probe,
                    scales_iso=scales_probe,
                    normals=normals_probe,
                    k_tangent=K_NEIGHBORS_TANGENT,
                    block_size=block,
                    query_workers=workers,
                    knn_idx_all=knn_probe_full_idx,
                )
                dt = max(1e-6, time.perf_counter() - t0)
                speed = points_probe.shape[0] / dt
                src = "cached_knn" if use_cached_knn else f"cKDTree workers={workers}"
                log(f"[INFO] Auto-tune aniso: block={block}, source={src}, speed={speed:.1f} pts/s")
                if (best is None) or (speed > best[0]):
                    best = (speed, block, workers)
                    best_rot = rot
                    best_cov6 = cov6
            except Exception as e:
                log(f"[WARN] Auto-tune aniso failed for block={block}, workers={workers}: {e}")

    if best is None or best_rot is None or best_cov6 is None:
        raise RuntimeError("Anisotropy auto-tune could not find a working configuration.")

    _, block_best, workers_best = best
    log(f"[INFO] Auto-tune aniso selected: block={block_best}, workers={workers_best}")
    return int(block_best), int(workers_best), best_rot, best_cov6


def autotune_frame_runtime(points_probe: np.ndarray,
                           rotations_probe: np.ndarray,
                           cov6_probe: np.ndarray,
                           profile: dict[str, object]) -> tuple[str, int]:
    backend_candidates = [str(FRAME_TRANSFORM_BACKEND)]
    if not USER_OVERRIDE_FRAME_BACKEND:
        if bool(profile["has_cuda"]):
            backend_candidates = ["cuda", "cpu"]
        else:
            backend_candidates = ["cpu"]

    block_candidates = [int(FRAME_TRANSFORM_BLOCK_SIZE)]
    if not USER_OVERRIDE_FRAME_BLOCK:
        base = int(max(4096, FRAME_TRANSFORM_BLOCK_SIZE))
        block_candidates = sorted({
            max(4096, base // 2),
            base,
            max(4096, base * 2),
        })

    best = None
    for backend in backend_candidates:
        for block in block_candidates:
            t0 = time.perf_counter()
            try:
                old_block = FRAME_TRANSFORM_BLOCK_SIZE
                try:
                    globals()["FRAME_TRANSFORM_BLOCK_SIZE"] = int(block)
                    apply_frame_transform(
                        points_probe.copy(),
                        rotations_probe.copy(),
                        cov6_probe.copy(),
                        M_ENU_TO_UNITY,
                        backend=backend,
                    )
                finally:
                    globals()["FRAME_TRANSFORM_BLOCK_SIZE"] = old_block

                dt = max(1e-6, time.perf_counter() - t0)
                speed = points_probe.shape[0] / dt
                log(f"[INFO] Auto-tune frame: backend={backend}, block={block}, speed={speed:.1f} pts/s")
                if (best is None) or (speed > best[0]):
                    best = (speed, backend, block)
            except Exception as e:
                log(f"[WARN] Auto-tune frame failed for backend={backend}, block={block}: {e}")

    if best is None:
        raise RuntimeError("Frame transform auto-tune could not find a working configuration.")

    _, backend_best, block_best = best
    log(f"[INFO] Auto-tune frame selected: backend={backend_best}, block={block_best}")
    return str(backend_best), int(block_best)


def maybe_autotune_runtime(points: np.ndarray,
                           normals: np.ndarray,
                           profile: dict[str, object]) -> None:
    global KNN_BACKEND, KNN_QUERY_BATCH
    global ANISO_BLOCK_SIZE, CKDTREE_WORKERS
    global FRAME_TRANSFORM_BACKEND, FRAME_TRANSFORM_BLOCK_SIZE

    if not AUTO_TUNE:
        return

    probe_idx = build_probe_indices(points.shape[0], AUTO_TUNE_PROBE_POINTS)
    if probe_idx.size < max(K_NEIGHBORS_TANGENT + 1, 2048):
        log("[INFO] Auto-tune skipped: probe sample too small.")
        return

    log(f"[INFO] Auto-tune starting on {probe_idx.size} probe points...")
    points_probe = points[probe_idx].astype(np.float32, copy=False)
    normals_probe = normals[probe_idx].astype(np.float32, copy=False)

    k_probe = int(max(K_NEIGHBORS_ISO + 1, K_NEIGHBORS_TANGENT + 1 if KNN_CACHE_MODE == "full" else K_NEIGHBORS_ISO + 1))
    knn_backend_best, knn_batch_best, knn_probe_idx, knn_probe_dist2 = autotune_knn_runtime(points_probe, k_probe, profile)
    KNN_BACKEND = knn_backend_best
    KNN_QUERY_BATCH = knn_batch_best

    scales_probe = compute_isotropic_scales(points_probe, None, K_NEIGHBORS_ISO, knn_probe_idx, knn_probe_dist2)
    knn_probe_full_idx = knn_probe_idx if (KNN_CACHE_MODE == "full" and knn_probe_idx.shape[1] >= (K_NEIGHBORS_TANGENT + 1)) else None

    aniso_block_best, workers_best, rot_probe, cov6_probe = autotune_aniso_runtime(
        points_probe=points_probe,
        scales_probe=scales_probe,
        normals_probe=normals_probe,
        profile=profile,
        knn_probe_full_idx=knn_probe_full_idx,
    )
    ANISO_BLOCK_SIZE = aniso_block_best
    CKDTREE_WORKERS = workers_best

    frame_backend_best, frame_block_best = autotune_frame_runtime(
        points_probe=points_probe,
        rotations_probe=rot_probe,
        cov6_probe=cov6_probe,
        profile=profile,
    )
    FRAME_TRANSFORM_BACKEND = frame_backend_best
    FRAME_TRANSFORM_BLOCK_SIZE = frame_block_best

    log("[INFO] Auto-tune summary:")
    log(f"  knn_backend={KNN_BACKEND}, knn_query_batch={KNN_QUERY_BATCH}")
    log(f"  aniso_block={ANISO_BLOCK_SIZE}, ckdtree_workers={CKDTREE_WORKERS}")
    log(f"  frame_backend={FRAME_TRANSFORM_BACKEND}, frame_block={FRAME_TRANSFORM_BLOCK_SIZE}")


# -------------------- helpers --------------------
def compute_recenter_origin(points: np.ndarray, mode: str = "bbox_center") -> np.ndarray:
    if points.size == 0:
        return np.zeros((3,), dtype=np.float32)
    mode = (mode or "bbox_center").lower()
    if mode == "mean":
        origin = points.mean(axis=0)
    elif mode == "first_point":
        origin = points[0]
    else:
        pmin = points.min(axis=0)
        pmax = points.max(axis=0)
        origin = (pmin + pmax) * 0.5
    return origin.astype(np.float32)

def apply_recenter(points: np.ndarray, origin: np.ndarray) -> np.ndarray:
    return (points - origin.reshape(1, 3)).astype(np.float32)

def orthonormalize_rotation(R):
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1.0
        R_ortho = U @ Vt
    return R_ortho.astype(np.float32)

def cov6_to_mat(cov6_row: np.ndarray) -> np.ndarray:
    xx, xy, xz, yy, yz, zz = cov6_row
    return np.array([[xx, xy, xz],
                     [xy, yy, yz],
                     [xz, yz, zz]], dtype=np.float32)

def mat_to_cov6(Sigma: np.ndarray) -> np.ndarray:
    return np.array([Sigma[0, 0], Sigma[0, 1], Sigma[0, 2],
                     Sigma[1, 1], Sigma[1, 2], Sigma[2, 2]], dtype=np.float32)

def build_cov6_transform_matrix(M: np.ndarray) -> np.ndarray:
    rows = np.asarray(M, dtype=np.float32)
    pairs = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    T = np.empty((6, 6), dtype=np.float32)
    for out_i, (r, c) in enumerate(pairs):
        u = rows[r]
        v = rows[c]
        T[out_i] = np.array([
            u[0] * v[0],
            u[0] * v[1] + u[1] * v[0],
            u[0] * v[2] + u[2] * v[0],
            u[1] * v[1],
            u[1] * v[2] + u[2] * v[1],
            u[2] * v[2],
        ], dtype=np.float32)
    return T

def apply_frame_transform(points: np.ndarray,
                          rotations: np.ndarray,
                          cov6: np.ndarray,
                          M: np.ndarray,
                          backend: str = FRAME_TRANSFORM_BACKEND) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    M32 = np.asarray(M, dtype=np.float32)
    MT32 = M32.T
    cov6_transform = build_cov6_transform_matrix(M32)

    points = np.asarray(points, dtype=np.float32)
    rotations = np.asarray(rotations, dtype=np.float32)
    cov6 = np.asarray(cov6, dtype=np.float32)

    n = int(points.shape[0])
    block_size = int(max(1, min(FRAME_TRANSFORM_BLOCK_SIZE, n)))

    def _maybe_log_progress(b1: int):
        if VERBOSE and n > block_size and (b1 == n or (b1 % max(LOG_EVERY, block_size) == 0)):
            log(f"  [FRAME-XFORM] processed {b1}/{n}")

    def _cpu_blocked() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for b0 in range(0, n, block_size):
            b1 = min(n, b0 + block_size)
            points[b0:b1] = points[b0:b1] @ MT32

            R_blk = rotations[b0:b1].reshape(-1, 3, 3)
            rotations[b0:b1] = np.einsum('ij,bjk->bik', M32, R_blk, optimize=True).reshape(-1, 9).astype(np.float32, copy=False)

            cov6[b0:b1] = (cov6[b0:b1] @ cov6_transform.T).astype(np.float32, copy=False)
            _maybe_log_progress(b1)

        return points, rotations, cov6

    b = (backend or "auto").lower().strip()
    if b not in ("auto", "cuda", "cpu"):
        b = "auto"

    if b in ("auto", "cuda"):
        try:
            if o3d.core.cuda.is_available():
                log("[INFO] Frame transform backend: Open3D Tensor CUDA")
                dev = o3d.core.Device("CUDA:0")
                t_M = o3d.core.Tensor(M32, dtype=o3d.core.Dtype.Float32, device=dev)
                t_MT = t_M.transpose(0, 1)
                for b0 in range(0, n, block_size):
                    b1 = min(n, b0 + block_size)

                    t_pts = o3d.core.Tensor(points[b0:b1], dtype=o3d.core.Dtype.Float32, device=dev)
                    points[b0:b1] = t_pts.matmul(t_MT).cpu().numpy().astype(np.float32, copy=False)

                    R_blk = rotations[b0:b1].reshape(-1, 3, 3)
                    t_R = o3d.core.Tensor(R_blk, dtype=o3d.core.Dtype.Float32, device=dev)
                    rotations[b0:b1] = t_M.matmul(t_R).cpu().numpy().reshape(-1, 9).astype(np.float32, copy=False)

                    cov6[b0:b1] = (cov6[b0:b1] @ cov6_transform.T).astype(np.float32, copy=False)
                    _maybe_log_progress(b1)

                return points, rotations, cov6
            elif b == "cuda":
                log("[WARN] Frame transform requested CUDA but CUDA unavailable; falling back to CPU.")
        except Exception as e:
            log(f"[WARN] Frame transform CUDA path failed: {e}; falling back to CPU.")

    log("[INFO] Frame transform backend: NumPy CPU")
    return _cpu_blocked()


# Refs: REF-SPEC-PLY. See /REFERENCES.md.
PLY_NUMPY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def read_ply_vertex_attribute_layout(ply_path: Path) -> Optional[dict[str, object]]:
    """Read the PLY header layout needed for direct RGB/intensity access."""
    with open(ply_path, "rb") as f:
        first = f.readline().decode("ascii", errors="replace").strip()
        if first != "ply":
            return None

        fmt = ""
        vertex_count = 0
        in_vertex = False
        vertex_props: list[tuple[str, str]] = []

        while True:
            raw = f.readline()
            if not raw:
                return None
            line = raw.decode("ascii", errors="replace").strip()
            if line == "end_header":
                data_offset = f.tell()
                break

            parts = line.split()
            if not parts:
                continue

            if parts[0] == "format" and len(parts) >= 2:
                fmt = parts[1]
            elif parts[0] == "element" and len(parts) >= 3:
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and parts[0] == "property":
                if len(parts) >= 5 and parts[1] == "list":
                    log("[WARN] Direct PLY RGB/intensity read skipped because vertex list properties are unsupported.")
                    return None
                if len(parts) >= 3:
                    vertex_props.append((parts[2], parts[1]))

    return {
        "format": fmt,
        "vertex_count": vertex_count,
        "vertex_props": vertex_props,
        "data_offset": data_offset,
    }


def build_ply_vertex_dtype(vertex_props: list[tuple[str, str]], endian: str) -> Optional[np.dtype]:
    dtype_fields = []
    used_names: set[str] = set()
    for name, ply_type in vertex_props:
        type_key = ply_type.lower()
        if type_key not in PLY_NUMPY_TYPES:
            log(f"[WARN] Direct PLY RGB/intensity read skipped because property type is unsupported: {ply_type}")
            return None
        field_name = name
        if field_name in used_names:
            suffix = 1
            while f"{field_name}_{suffix}" in used_names:
                suffix += 1
            field_name = f"{field_name}_{suffix}"
        used_names.add(field_name)
        dtype_fields.append((field_name, endian + PLY_NUMPY_TYPES[type_key]))
    return np.dtype(dtype_fields)


def normalize_rgb_channels(rgb_raw: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_raw, dtype=np.float32)
    if rgb.size == 0:
        return rgb.reshape((-1, 3))
    if np.nanmax(rgb) > 1.1:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


def normalize_intensity_for_color(intensity: np.ndarray) -> Optional[np.ndarray]:
    values = np.asarray(intensity, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None

    if finite.size > INTENSITY_PERCENTILE_SAMPLE:
        step = max(1, finite.size // INTENSITY_PERCENTILE_SAMPLE)
        finite = finite[::step]

    lo = float(np.percentile(finite, INTENSITY_LOW_PERCENTILE))
    hi = float(np.percentile(finite, INTENSITY_HIGH_PERCENTILE))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return np.full(values.shape, 0.5, dtype=np.float32)

    norm = (values - lo) / (hi - lo)
    norm = np.nan_to_num(norm, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(norm, 0.0, 1.0).astype(np.float32, copy=False)


def apply_intensity_aux_coloring(colors: np.ndarray, intensity: Optional[np.ndarray]) -> np.ndarray:
    if not USE_INTENSITY_AUX_COLOR or intensity is None or INTENSITY_COLOR_STRENGTH <= 0.0:
        return colors

    norm = normalize_intensity_for_color(intensity)
    if norm is None or norm.shape[0] != colors.shape[0]:
        log("[WARN] scalar_Intensity was found but could not be aligned with RGB colors; using RGB only.")
        return colors

    strength = float(np.clip(INTENSITY_COLOR_STRENGTH, 0.0, 1.0))
    factor = 1.0 + strength * ((norm * 2.0) - 1.0)
    shaded = colors * factor[:, None]
    log(
        "[INFO] Applied scalar_Intensity auxiliary coloring "
        f"(strength={strength:.2f}, percentiles={INTENSITY_LOW_PERCENTILE:.1f}-{INTENSITY_HIGH_PERCENTILE:.1f})."
    )
    return np.clip(shaded, 0.0, 1.0).astype(np.float32, copy=False)


def load_direct_ply_rgb_with_intensity(ply_path: Path, expected_points: int) -> Optional[np.ndarray]:
    layout = read_ply_vertex_attribute_layout(ply_path)
    if not layout:
        return None

    fmt = str(layout["format"]).lower().strip()
    if fmt != "binary_little_endian":
        log(f"[WARN] Direct PLY RGB/intensity read skipped for unsupported PLY format: {fmt}")
        return None

    vertex_count = int(layout["vertex_count"])
    if vertex_count != int(expected_points):
        log(
            "[WARN] Direct PLY RGB/intensity read skipped because vertex count "
            f"({vertex_count}) does not match Open3D point count ({expected_points})."
        )
        return None

    vertex_props = list(layout["vertex_props"])
    prop_names = {name for name, _ in vertex_props}
    required_rgb = {"red", "green", "blue"}
    if not required_rgb.issubset(prop_names):
        missing = ", ".join(sorted(required_rgb - prop_names))
        log(f"[WARN] Direct RGB fields missing from PLY ({missing}); falling back to Open3D colors.")
        return None

    dtype = build_ply_vertex_dtype(vertex_props, "<")
    if dtype is None:
        return None

    data = np.memmap(ply_path, dtype=dtype, mode="r", offset=int(layout["data_offset"]), shape=(vertex_count,))
    rgb_raw = np.column_stack([data["red"], data["green"], data["blue"]])
    colors = normalize_rgb_channels(rgb_raw)

    intensity = None
    if "scalar_Intensity" in data.dtype.names:
        intensity = np.asarray(data["scalar_Intensity"], dtype=np.float32)
    elif "intensity" in data.dtype.names:
        intensity = np.asarray(data["intensity"], dtype=np.float32)

    colors = apply_intensity_aux_coloring(colors, intensity)
    log("[INFO] Loaded RGB directly from PLY red/green/blue fields; f_dc_* fields are not used for ordinary point clouds.")
    return colors


def apply_direct_ply_color_attributes(pcd: o3d.geometry.PointCloud, ply_path: Path) -> None:
    colors = load_direct_ply_rgb_with_intensity(ply_path, int(np.asarray(pcd.points).shape[0]))
    if colors is None:
        return
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64, copy=False))


def random_downsample_pcd(pcd: o3d.geometry.PointCloud, max_points: int) -> o3d.geometry.PointCloud:
    n = np.asarray(pcd.points).shape[0]
    if max_points is None or max_points <= 0 or n <= max_points:
        return pcd
    idx = np.random.choice(n, int(max_points), replace=False)
    return pcd.select_by_index(idx)


def uniform_downsample_pcd(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    if voxel_size is None or voxel_size <= 0:
        return pcd
    return pcd.voxel_down_sample(float(voxel_size))


def apply_sampling_strategy(pcd: o3d.geometry.PointCloud) -> tuple[o3d.geometry.PointCloud, dict[str, object]]:
    n_full = int(np.asarray(pcd.points).shape[0])
    mode = (SAMPLE_MODE or "random").lower().strip()

    if mode == "random":
        target = int(MAX_GAUSSIANS) if MAX_GAUSSIANS is not None else -1
        pcd_proc = random_downsample_pcd(pcd, target)
        n_out = int(np.asarray(pcd_proc.points).shape[0])
        label = "full" if target <= 0 else format_count_label(target)
        return pcd_proc, {
            "sampling_method": "random",
            "sampling_parameter_name": "point_count",
            "sampling_parameter_value": int(target),
            "sampling_parameter_label": label,
            "input_points": n_full,
            "output_points": n_out,
        }

    if mode == "uniform":
        resolution = float(UNIFORM_RESOLUTION)
        pcd_proc = uniform_downsample_pcd(pcd, resolution)
        n_out = int(np.asarray(pcd_proc.points).shape[0])
        return pcd_proc, {
            "sampling_method": "uniform",
            "sampling_parameter_name": "resolution_m",
            "sampling_parameter_value": resolution,
            "sampling_parameter_label": format_resolution_label(resolution),
            "input_points": n_full,
            "output_points": n_out,
        }

    raise ValueError(f"Unknown sampling mode: {SAMPLE_MODE}")


def format_count_label(value: int) -> str:
    if value <= 0:
        return "full"
    if value % 1_000_000 == 0:
        return f"{value // 1_000_000}M"
    if value % 1_000 == 0:
        return f"{value // 1_000}K"
    return str(value)


def format_resolution_label(value_m: float) -> str:
    cm = value_m * 100.0
    if abs(cm - round(cm)) < 1e-6:
        return f"{int(round(cm))}cm"
    return f"{cm:g}cm"


def parse_count_token(token: str) -> int:
    t = token.strip().lower().replace("_", "")
    if t in ("full", "all", "-1"):
        return -1
    if t.endswith("m"):
        return int(float(t[:-1]) * 1_000_000)
    if t.endswith("k"):
        return int(float(t[:-1]) * 1_000)
    return int(float(t))


def parse_resolution_token(token: str) -> float:
    t = token.strip().lower().replace("_", "")
    if t.endswith("cm"):
        return float(t[:-2]) / 100.0
    if t.endswith("m"):
        return float(t[:-1])
    return float(t)


def default_lod_specs(method: str) -> list[dict[str, object]]:
    m = (method or "random").lower().strip()
    if m == "random":
        values = [-1, 10_000_000, 1_000_000, 100_000, 10_000]
        return [
            {
                "level": i,
                "sampling_method": "random",
                "sampling_parameter_name": "point_count",
                "sampling_parameter_value": v,
                "sampling_parameter_label": format_count_label(v),
            }
            for i, v in enumerate(values)
        ]
    if m == "uniform":
        values = [0.01, 0.02, 0.05, 0.10, 0.20]
        return [
            {
                "level": i,
                "sampling_method": "uniform",
                "sampling_parameter_name": "resolution_m",
                "sampling_parameter_value": v,
                "sampling_parameter_label": format_resolution_label(v),
            }
            for i, v in enumerate(values)
        ]
    raise ValueError(f"Unknown sampling method for default LOD specs: {method}")


def parse_lod_specs(spec: str, method: str) -> list[dict[str, object]]:
    if not spec:
        return default_lod_specs(method)

    levels: list[dict[str, object]] = []
    for i, raw_item in enumerate(spec.split(",")):
        item = raw_item.strip()
        if not item:
            continue
        if "=" in item:
            level_token, value_token = item.split("=", 1)
            level = int(level_token.strip().upper().lstrip("L"))
        elif ":" in item:
            level_token, value_token = item.split(":", 1)
            level = int(level_token.strip().upper().lstrip("L"))
        else:
            level = i
            value_token = item

        m = (method or "random").lower().strip()
        if m == "random":
            value = parse_count_token(value_token)
            label = format_count_label(value)
            name = "point_count"
        elif m == "uniform":
            value = parse_resolution_token(value_token)
            label = format_resolution_label(value)
            name = "resolution_m"
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        levels.append({
            "level": level,
            "sampling_method": m,
            "sampling_parameter_name": name,
            "sampling_parameter_value": value,
            "sampling_parameter_label": label,
        })

    if not levels:
        raise ValueError("LOD spec did not contain any levels.")
    return sorted(levels, key=lambda x: int(x["level"]))


def batch_knn_search(points_np: np.ndarray,
                     k: int,
                     backend: str = KNN_BACKEND,
                     query_batch: int = KNN_QUERY_BATCH) -> tuple[np.ndarray, np.ndarray]:
    k = int(k)
    if k <= 0:
        raise ValueError("k must be > 0")

    pts = np.asarray(points_np, dtype=np.float32)
    n = pts.shape[0]

    backend = (backend or "auto").lower().strip()
    query_batch = int(max(1, query_batch))

    def _try_tensor_knn(device):
        t_pts = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32, device=device)
        nns = o3d.core.nns.NearestNeighborSearch(t_pts)
        nns.knn_index()

        idx_out = np.empty((n, k), dtype=np.int32)
        dist2_out = np.empty((n, k), dtype=np.float32)

        for b0 in range(0, n, query_batch):
            b1 = min(n, b0 + query_batch)
            q = t_pts[b0:b1]
            idx_t, dist2_t = nns.knn_search(q, k)
            idx_out[b0:b1] = idx_t.cpu().numpy().astype(np.int32, copy=False)
            dist2_out[b0:b1] = dist2_t.cpu().numpy().astype(np.float32, copy=False)

        return idx_out, dist2_out

    if backend not in ("auto", "cuda", "cpu"):
        backend = "auto"

    if backend in ("auto", "cuda"):
        try:
            if o3d.core.cuda.is_available():
                log("[INFO] KNN backend: Open3D Tensor CUDA")
                return _try_tensor_knn(o3d.core.Device("CUDA:0"))
            elif backend == "cuda":
                log("[WARN] Requested CUDA KNN, but CUDA is unavailable; falling back.")
        except Exception as e:
            log(f"[WARN] CUDA Tensor KNN failed: {e}; falling back.")

    try:
        log("[INFO] KNN backend: Open3D Tensor CPU")
        return _try_tensor_knn(o3d.core.Device("CPU:0"))
    except Exception as e:
        log(f"[WARN] Open3D tensor NNS unavailable: {e}; falling back to KDTreeFlann loop (slow).")

    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    kdt = o3d.geometry.KDTreeFlann(pcd_tmp)

    idx = np.empty((n, k), dtype=np.int32)
    dist2 = np.empty((n, k), dtype=np.float32)

    for i in range(n):
        kk, ii, dd = kdt.search_knn_vector_3d(pts[i], k)
        if kk < k:
            ii = list(ii) + [i] * (k - kk)
            dd = list(dd) + [0.0] * (k - kk)
        idx[i, :] = np.asarray(ii[:k], dtype=np.int32)
        dist2[i, :] = np.asarray(dd[:k], dtype=np.float32)
        if VERBOSE and (i > 0 and i % LOG_EVERY == 0):
            log(f"  [KNN-FALLBACK] processed {i}/{n} points...")

    return idx, dist2


def compute_isotropic_scales(points,
                            kdtree,
                            k_neighbors,
                            knn_idx: Optional[np.ndarray] = None,
                            knn_dist2: Optional[np.ndarray] = None,
                            query_workers: int = CKDTREE_WORKERS,
                            query_block: int = KNN_QUERY_BATCH):
    num = points.shape[0]
    scales_iso = np.zeros((num,), dtype=np.float32)

    log("[INFO] Estimating isotropic scales from neighbor distances...")

    if knn_idx is not None and knn_dist2 is not None:
        d = np.sqrt(knn_dist2[:, 1:k_neighbors + 1].astype(np.float64))
        mean_dist = np.sqrt(np.mean((d ** 2), axis=1))
        s = (mean_dist * 0.75).astype(np.float32)
        scales_iso[:] = s
        if VERBOSE:
            dist_arr = s.astype(np.float64)
            log("[INFO] Isotropic scale summary (m): "
                f"min={dist_arr.min():.4f}, p50={np.percentile(dist_arr, 50):.4f}, "
                f"p95={np.percentile(dist_arr, 95):.4f}, max={dist_arr.max():.4f}, mean={dist_arr.mean():.4f}")
        return scales_iso

    if _HAS_SCIPY:
        pts = np.asarray(points, dtype=np.float32)
        tree = cKDTree(pts)
        kk = int(k_neighbors + 1)
        block = int(max(1024, query_block))

        for b0 in range(0, num, block):
            b1 = min(num, b0 + block)
            dist, _idx = _tree_query(tree, pts[b0:b1], k=kk, workers=query_workers)
            dist = np.asarray(dist, dtype=np.float64)
            if dist.ndim == 1:
                dist = dist.reshape(-1, 1)

            if dist.shape[1] > 1:
                d = dist[:, 1:kk]
                mean_dist = np.sqrt(np.mean((d ** 2), axis=1))
            else:
                mean_dist = np.full((b1 - b0,), 0.05, dtype=np.float64)

            scales_iso[b0:b1] = (mean_dist * 0.75).astype(np.float32, copy=False)
            if VERBOSE and num > block and (b1 == num or (b1 % max(LOG_EVERY, block) == 0)):
                log(f"  [ISO-CKDTREE] processed {b1}/{num} points...")

        dist_arr = scales_iso.astype(np.float64)
        if VERBOSE:
            log("[INFO] Isotropic scale summary (m): "
                f"min={dist_arr.min():.4f}, p50={np.percentile(dist_arr, 50):.4f}, "
                f"p95={np.percentile(dist_arr, 95):.4f}, max={dist_arr.max():.4f}, mean={dist_arr.mean():.4f}")
        return scales_iso

    dist_list = []
    for i in range(num):
        k, idx, dist2 = kdtree.search_knn_vector_3d(points[i], k_neighbors + 1)
        if k > 1:
            mean_dist = float(np.sqrt(np.mean(dist2[1:])))
        else:
            mean_dist = 0.05
        s = mean_dist * 0.75
        scales_iso[i] = s
        dist_list.append(s)
        if VERBOSE and (i > 0 and i % LOG_EVERY == 0):
            log(f"  [ISO] processed {i}/{num} points...")

    dist_arr = np.asarray(dist_list, dtype=np.float64)
    if VERBOSE:
        log("[INFO] Isotropic scale summary (m): "
            f"min={dist_arr.min():.4f}, p50={np.percentile(dist_arr, 50):.4f}, "
            f"p95={np.percentile(dist_arr, 95):.4f}, max={dist_arr.max():.4f}, mean={dist_arr.mean():.4f}")
    return scales_iso


def compute_adaptive_clamp(scales_iso: np.ndarray) -> tuple[float, float]:
    q_min = float(np.percentile(scales_iso, ADAPTIVE_Q_MIN))
    q_max = float(np.percentile(scales_iso, ADAPTIVE_Q_MAX))

    s_min = q_min * ADAPTIVE_MIN_FACTOR
    s_max = q_max * ADAPTIVE_MAX_FACTOR

    s_min = float(np.clip(s_min, ABS_S_MIN, ABS_S_MAX))
    s_max = float(np.clip(s_max, ABS_S_MIN, ABS_S_MAX))

    if s_max <= s_min:
        s_max = min(ABS_S_MAX, s_min * 2.0)

    return s_min, s_max


def make_tangent_basis_from_normals_batch(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = n.astype(np.float32, copy=False)
    B = n.shape[0]

    a = np.zeros((B, 3), dtype=np.float32)
    use_z = (np.abs(n[:, 2]) < 0.9)
    a[use_z, 2] = 1.0
    a[~use_z, 1] = 1.0

    t0 = np.cross(a, n)
    t0n = np.linalg.norm(t0, axis=1, keepdims=True)

    bad = (t0n[:, 0] < 1e-8)
    if np.any(bad):
        a2 = np.zeros((np.count_nonzero(bad), 3), dtype=np.float32)
        a2[:, 0] = 1.0
        t0_bad = np.cross(a2, n[bad])
        t0[bad] = t0_bad
        t0n[bad] = np.linalg.norm(t0_bad, axis=1, keepdims=True)

    t0 = t0 / np.maximum(t0n, 1e-8)
    t1 = np.cross(n, t0)
    t1 = t1 / np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-8)
    return t0.astype(np.float32, copy=False), t1.astype(np.float32, copy=False)


def eig2x2_sym_batch(C2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = C2[:, 0, 0].astype(np.float64, copy=False)
    b = C2[:, 0, 1].astype(np.float64, copy=False)
    c = C2[:, 1, 1].astype(np.float64, copy=False)

    tr = a + c
    det = a * c - b * b
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    s = np.sqrt(disc)
    l1 = 0.5 * (tr + s)
    l2 = 0.5 * (tr - s)

    v1 = np.zeros((C2.shape[0], 2), dtype=np.float64)
    use_b = (np.abs(b) > 1e-18)
    v1[use_b, 0] = (l1[use_b] - c[use_b])
    v1[use_b, 1] = b[use_b]
    diag = ~use_b
    v1[diag, 0] = 1.0

    v1n = np.linalg.norm(v1, axis=1, keepdims=True)
    v1 = v1 / np.maximum(v1n, 1e-18)
    v2 = np.stack([-v1[:, 1], v1[:, 0]], axis=1)

    return l1.astype(np.float64), l2.astype(np.float64), v1.astype(np.float32), v2.astype(np.float32)


def compute_normal_aligned_gaussians_batched(points: np.ndarray,
                                             scales_iso: np.ndarray,
                                             normals: np.ndarray,
                                             k_tangent: int = K_NEIGHBORS_TANGENT,
                                             block_size: int = 50000,
                                             query_workers: int = CKDTREE_WORKERS,
                                             knn_idx_all: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    use_cached_knn = knn_idx_all is not None
    if (not use_cached_knn) and (not _HAS_SCIPY):
        raise RuntimeError("SciPy not available and no cached KNN provided.")

    pts = np.asarray(points, dtype=np.float32)
    N = pts.shape[0]
    k = int(k_tangent)
    if k <= 2:
        raise ValueError("k_tangent must be >= 3")

    nrm = np.asarray(normals, dtype=np.float32)
    nn = np.linalg.norm(nrm, axis=1)
    good = nn > 1e-8
    nrm = nrm.copy()
    nrm[good] /= nn[good].reshape(-1, 1)
    nrm[~good] = np.array([0, 0, 1], dtype=np.float32)

    scales_aniso = np.zeros((N, 3), dtype=np.float32)
    rotations = np.zeros((N, 9), dtype=np.float32)
    cov6 = np.zeros((N, 6), dtype=np.float32)

    tree = cKDTree(pts) if not use_cached_knn else None

    log("[INFO] Step 6 (BATCH): anisotropy + numpy broadcasting...")
    if use_cached_knn:
        log("[INFO] Step 6 KNN source: cached batched KNN")
    else:
        log("[INFO] Step 6 KNN source: SciPy cKDTree query")
    log(f"[INFO] N={N}, k_tangent={k}, block_size={block_size}, query_workers={query_workers}")

    q = float(DT_PERCENTILE)
    block_size_cur = int(max(ANISO_MIN_BLOCK_SIZE, block_size))

    def _process_block(b0: int, b1: int):
        B = b1 - b0

        p = pts[b0:b1]
        n = nrm[b0:b1]
        s_iso = scales_iso[b0:b1].astype(np.float32, copy=False)

        if use_cached_knn:
            idx = np.asarray(knn_idx_all[b0:b1, :k], dtype=np.int64)
            if idx.shape[1] < k:
                raise RuntimeError(f"Cached KNN has insufficient columns: need {k}, got {idx.shape[1]}")
        else:
            _d, idx = _tree_query(tree, p, k=k, workers=query_workers)
            idx = idx.astype(np.int64, copy=False)

        idx_n = idx[:, 1:]
        neigh = pts[idx_n]

        t0, t1 = make_tangent_basis_from_normals_batch(n)
        p_t0 = np.einsum('bc,bc->b', p, t0, optimize=True).astype(np.float64, copy=False)
        p_t1 = np.einsum('bc,bc->b', p, t1, optimize=True).astype(np.float64, copy=False)

        u = np.einsum('bkc,bc->bk', neigh, t0, optimize=True).astype(np.float64, copy=False)
        v = np.einsum('bkc,bc->bk', neigh, t1, optimize=True).astype(np.float64, copy=False)
        u -= p_t0[:, None]
        v -= p_t1[:, None]

        d2 = u * u + v * v

        if USE_TANGENT_WEIGHTING:
            mean_neighbor_dist = (s_iso / 0.75).astype(np.float64)
            sigma_w = (TANGENT_WEIGHT_SIGMA_FACTOR * mean_neighbor_dist)
            sigma_w = np.clip(sigma_w, TANGENT_WEIGHT_SIGMA_W_MIN, TANGENT_WEIGHT_SIGMA_W_MAX)
            w = np.exp(-0.5 * d2 / np.maximum(sigma_w[:, None] * sigma_w[:, None], 1e-12))
            w_sum = np.maximum(np.sum(w, axis=1), 1e-12)

            mu_u = np.sum(w * u, axis=1) / w_sum
            mu_v = np.sum(w * v, axis=1) / w_sum

            c00 = np.sum(w * u * u, axis=1) / w_sum - mu_u * mu_u
            c01 = np.sum(w * u * v, axis=1) / w_sum - mu_u * mu_v
            c11 = np.sum(w * v * v, axis=1) / w_sum - mu_v * mu_v
        else:
            mu_u = np.mean(u, axis=1)
            mu_v = np.mean(v, axis=1)
            m = max(1, u.shape[1])
            denom = max(1, m - 1)
            c00 = (np.sum(u * u, axis=1) - m * mu_u * mu_u) / denom
            c01 = (np.sum(u * v, axis=1) - m * mu_u * mu_v) / denom
            c11 = (np.sum(v * v, axis=1) - m * mu_v * mu_v) / denom

        C2 = np.zeros((B, 2, 2), dtype=np.float64)
        C2[:, 0, 0] = c00
        C2[:, 0, 1] = c01
        C2[:, 1, 0] = c01
        C2[:, 1, 1] = c11

        l1, l2, v1_2d, v2_2d = eig2x2_sym_batch(C2)
        l1 = np.maximum(l1, 0.0) + 1e-12
        l2 = np.maximum(l2, 0.0) + 1e-12

        if DT_USE_PARTITION:
            m = d2.shape[1]
            pos = int(np.clip(np.round((q / 100.0) * (m - 1)), 0, m - 1))
            d2_part = np.partition(d2, pos, axis=1)
            d_t = np.sqrt(np.maximum(d2_part[:, pos], 0.0))
        else:
            d_t = np.sqrt(np.maximum(np.percentile(d2, q, axis=1), 0.0))

        ar = np.sqrt(l1 / l2)
        ar = np.clip(ar, AR_MIN, AR_MAX)

        st_base = (DT_TO_SIGMA_FACTOR * d_t).astype(np.float64)
        st_base = np.clip(
            st_base,
            (TANGENT_REL_MIN * s_iso).astype(np.float64),
            (TANGENT_REL_MAX * s_iso).astype(np.float64),
        )
        st_base = np.clip(st_base, S_MIN, S_MAX)

        s_ar = np.sqrt(ar)
        st1 = np.clip(st_base * s_ar, S_MIN, S_MAX)
        st2 = np.clip(st_base / np.maximum(s_ar, 1e-8), S_MIN, S_MAX)

        if SIGMA_N_MODE.lower() == "relative":
            sn = (SIGMA_N_REL_FACTOR * s_iso).astype(np.float64)
        else:
            sn = np.full((B,), float(SIGMA_N_FIXED), dtype=np.float64)
        sn = np.clip(sn, SIGMA_N_MIN, SIGMA_N_MAX)
        sn = np.minimum(sn, 0.5 * np.maximum(np.minimum(st1, st2), 1e-6))

        t_major = (v1_2d[:, 0:1] * t0 + v1_2d[:, 1:2] * t1).astype(np.float32)
        t_minor = (v2_2d[:, 0:1] * t0 + v2_2d[:, 1:2] * t1).astype(np.float32)

        t_major = t_major / np.maximum(np.linalg.norm(t_major, axis=1, keepdims=True), 1e-8)
        proj = np.sum(t_minor * t_major, axis=1, keepdims=True)
        t_minor = t_minor - proj * t_major
        t_minor = t_minor / np.maximum(np.linalg.norm(t_minor, axis=1, keepdims=True), 1e-8)

        R = np.zeros((B, 3, 3), dtype=np.float32)
        R[:, :, 0] = t_major
        R[:, :, 1] = t_minor
        R[:, :, 2] = n

        sigma = np.stack([st1.astype(np.float32), st2.astype(np.float32), sn.astype(np.float32)], axis=1)
        scales_aniso[b0:b1] = sigma
        rotations[b0:b1] = R.reshape(B, 9)

        s1_2 = sigma[:, 0] * sigma[:, 0]
        s2_2 = sigma[:, 1] * sigma[:, 1]
        sn_2 = sigma[:, 2] * sigma[:, 2]
        cov_blk = cov6[b0:b1]
        cov_blk[:, 0] = s1_2 * t_major[:, 0] * t_major[:, 0] + s2_2 * t_minor[:, 0] * t_minor[:, 0] + sn_2 * n[:, 0] * n[:, 0]
        cov_blk[:, 1] = s1_2 * t_major[:, 0] * t_major[:, 1] + s2_2 * t_minor[:, 0] * t_minor[:, 1] + sn_2 * n[:, 0] * n[:, 1]
        cov_blk[:, 2] = s1_2 * t_major[:, 0] * t_major[:, 2] + s2_2 * t_minor[:, 0] * t_minor[:, 2] + sn_2 * n[:, 0] * n[:, 2]
        cov_blk[:, 3] = s1_2 * t_major[:, 1] * t_major[:, 1] + s2_2 * t_minor[:, 1] * t_minor[:, 1] + sn_2 * n[:, 1] * n[:, 1]
        cov_blk[:, 4] = s1_2 * t_major[:, 1] * t_major[:, 2] + s2_2 * t_minor[:, 1] * t_minor[:, 2] + sn_2 * n[:, 1] * n[:, 2]
        cov_blk[:, 5] = s1_2 * t_major[:, 2] * t_major[:, 2] + s2_2 * t_minor[:, 2] * t_minor[:, 2] + sn_2 * n[:, 2] * n[:, 2]

    b0 = 0
    while b0 < N:
        b1 = min(N, b0 + block_size_cur)
        try:
            _process_block(b0, b1)
        except MemoryError:
            if block_size_cur <= ANISO_MIN_BLOCK_SIZE:
                raise
            new_block = max(ANISO_MIN_BLOCK_SIZE, block_size_cur // 2)
            log(f"[WARN] Step 6 block {b0}:{b1} ran out of memory; reducing aniso block_size from {block_size_cur} to {new_block} and retrying.")
            block_size_cur = new_block
            gc.collect()
            continue

        if VERBOSE and (b0 > 0 and (b0 % LOG_EVERY == 0)):
            log(f"  [BATCH-ANISO] processed {b1}/{N}")
        b0 = b1

    if VERBOSE:
        log("[INFO] Step 6 (BATCH) done.")
        log(f"[INFO] aniso st1 p50={np.percentile(scales_aniso[:,0],50):.4f}, p95={np.percentile(scales_aniso[:,0],95):.4f}")
        log(f"[INFO] aniso st2 p50={np.percentile(scales_aniso[:,1],50):.4f}, p95={np.percentile(scales_aniso[:,1],95):.4f}")
        log(f"[INFO] aniso sn  p50={np.percentile(scales_aniso[:,2],50):.4f}, p95={np.percentile(scales_aniso[:,2],95):.4f}")

    return scales_aniso, rotations, cov6


def write_chunks_from_arrays(P: np.ndarray,
                             C: np.ndarray,
                             cov6: np.ndarray,
                             chunk_dir: Path,
                             chunk_prefix: str,
                             chunk_size: np.ndarray,
                             eps: float = 1e-5):
    chunk_dir.mkdir(parents=True, exist_ok=True)

    N = P.shape[0]
    xyz_min = P.min(axis=0)
    xyz_max = P.max(axis=0)

    origin = xyz_min.copy()
    extent = xyz_max - xyz_min
    grid = np.ceil((extent + eps) / chunk_size).astype(int)
    nx, ny, nz = map(int, grid.tolist())

    rel = P - origin[None, :]
    ijk = np.floor(rel / chunk_size[None, :]).astype(int)
    ijk[:, 0] = np.clip(ijk[:, 0], 0, nx - 1)
    ijk[:, 1] = np.clip(ijk[:, 1], 0, ny - 1)
    ijk[:, 2] = np.clip(ijk[:, 2], 0, nz - 1)

    chunk_dict: dict[tuple[int,int,int], list[int]] = {}
    for i in range(N):
        key = (int(ijk[i, 0]), int(ijk[i, 1]), int(ijk[i, 2]))
        chunk_dict.setdefault(key, []).append(i)

    chunk_meta = []
    for (ix, iy, iz), idx_list in chunk_dict.items():
        idx_arr = np.asarray(idx_list, dtype=np.int64)
        P_chunk = P[idx_arr]
        C_chunk = C[idx_arr] if C.size else np.tile(np.array([[0.7,0.7,0.7]], dtype=np.float32), (P_chunk.shape[0],1))
        cov6_chunk = cov6[idx_arr]

        mat = np.hstack([P_chunk, C_chunk, cov6_chunk])
        fname = f"{chunk_prefix}_{ix}_{iy}_{iz}.txt"
        fpath = chunk_dir / fname
        np.savetxt(fpath, mat, fmt="%.6f")

        bbox_min = P_chunk.min(axis=0)
        bbox_max = P_chunk.max(axis=0)
        center = 0.5 * (bbox_min + bbox_max)

        chunk_meta.append({
            "ijk": [ix, iy, iz],
            "filename": fname,
            "count": int(P_chunk.shape[0]),
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "center": center.tolist(),
        })

    index = {
        "origin": origin.tolist(),
        "chunk_size": chunk_size.tolist(),
        "grid_shape": [nx, ny, nz],
        "num_points": int(N),
        "num_chunks": int(len(chunk_meta)),
        "chunks": chunk_meta,
    }

    index_path = chunk_dir / "chunks_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    log(f"[INFO] Wrote {len(chunk_meta)} chunk files to: {chunk_dir}")
    log(f"[INFO] Wrote chunk index JSON: {index_path}")


def build_gaussians_once(output_npz_override: Optional[Path] = None,
                         lod_metadata: Optional[dict[str, object]] = None) -> dict[str, object]:
    t_start = time.perf_counter()
    stage_times: dict[str, float] = {}
    device_profile: Optional[dict[str, object]] = None

    t_stage = time.perf_counter()
    log(f"[INFO] Loading point cloud from: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    pcd = o3d.io.read_point_cloud(str(INPUT_PATH))
    if pcd.is_empty():
        raise RuntimeError("Open3D loaded empty point cloud. Check file path/format.")

    log(f"[INFO] Loaded {np.asarray(pcd.points).shape[0]} points")
    apply_direct_ply_color_attributes(pcd, INPUT_PATH)
    stage_times["load_pcd"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    device_profile = apply_device_profile_if_needed(np.asarray(pcd.points).shape[0])
    stage_times["device_profile"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    origin_pre_shift = np.zeros((3,), dtype=np.float32)
    if ENABLE_PRE_SHIFT:
        pts0 = np.asarray(pcd.points).astype(np.float64)
        origin_pre_shift = compute_recenter_origin(pts0, PRE_SHIFT_MODE).astype(np.float32)
        pcd.translate((-origin_pre_shift).astype(np.float64), relative=True)
        log("[INFO] Pre-shift enabled (float precision fix):")
        log(f"  PRE_SHIFT_MODE={PRE_SHIFT_MODE}")
        log(f"  origin_pre_shift (input frame) = [{origin_pre_shift[0]:.3f}, {origin_pre_shift[1]:.3f}, {origin_pre_shift[2]:.3f}]")
    stage_times["pre_shift"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    pcd_proc, sampling_metadata = apply_sampling_strategy(pcd)
    if lod_metadata:
        sampling_metadata.update(lod_metadata)
    proc_n = int(np.asarray(pcd_proc.points).shape[0])

    log(
        "[INFO] Sampling "
        f"method={sampling_metadata.get('sampling_method')} "
        f"parameter={sampling_metadata.get('sampling_parameter_label')} "
        f"input={sampling_metadata.get('input_points')} -> output={proc_n}"
    )

    points = np.asarray(pcd_proc.points).astype(np.float32, copy=False)
    colors = np.asarray(pcd_proc.colors).astype(np.float32, copy=False)

    if colors.size > 0 and colors.max() > 1.1:
        colors = colors / 255.0

    normals = None
    if USE_FILE_NORMALS_FIRST and pcd_proc.has_normals():
        normals = np.asarray(pcd_proc.normals).astype(np.float32, copy=False)
        log("[INFO] Using normals from input PLY (no estimation).")
    elif ALLOW_O3D_ESTIMATE_NORMALS:
        log(f"[INFO] Estimating normals with Open3D (C++) KNN={K_NEIGHBORS_NORMAL} ...")
        pcd_proc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=K_NEIGHBORS_NORMAL))
        pcd_proc.normalize_normals()
        normals = np.asarray(pcd_proc.normals).astype(np.float32, copy=False)
    else:
        raise RuntimeError("No normals in file and estimation disabled.")
    stage_times["sampling_and_normals"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    if device_profile is None:
        device_profile = detect_device_profile()
    maybe_autotune_runtime(points, normals, device_profile)
    stage_times["autotune"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    K_MAX_CACHE = int(max(K_NEIGHBORS_ISO, K_NEIGHBORS_NORMAL, K_NEIGHBORS_TANGENT) + 1)
    K_ISO_CACHE = int(K_NEIGHBORS_ISO + 1)
    knn_idx_all = None
    knn_dist2_all = None
    kdtree = None

    cache_mode = (KNN_CACHE_MODE or "auto").lower().strip()
    if cache_mode == "auto":
        cache_mode = "full"

    if cache_mode not in ("full", "iso", "none"):
        cache_mode = "full"

    if cache_mode == "full":
        log("[INFO] Building KDTree...")
        kdtree = o3d.geometry.KDTreeFlann(pcd_proc)
        log(f"[INFO] Building cached batched KNN for isotropic+anisotropy (K={K_MAX_CACHE})...")
        knn_idx_all, knn_dist2_all = batch_knn_search(points, K_MAX_CACHE, backend=KNN_BACKEND, query_batch=KNN_QUERY_BATCH)
    elif cache_mode == "iso":
        log("[INFO] Building KDTree...")
        kdtree = o3d.geometry.KDTreeFlann(pcd_proc)
        log(f"[INFO] Building cached batched KNN for isotropic scales only (K={K_ISO_CACHE})...")
        knn_idx_all, knn_dist2_all = batch_knn_search(points, K_ISO_CACHE, backend=KNN_BACKEND, query_batch=KNN_QUERY_BATCH)
    else:
        if not _HAS_SCIPY:
            log("[INFO] Building KDTree...")
            kdtree = o3d.geometry.KDTreeFlann(pcd_proc)
        log("[INFO] KNN cache disabled; isotropic scales and anisotropy will use fallback paths as needed.")

    scales_iso = compute_isotropic_scales(
        points,
        kdtree,
        K_NEIGHBORS_ISO,
        knn_idx_all,
        knn_dist2_all,
        query_workers=CKDTREE_WORKERS,
        query_block=KNN_QUERY_BATCH,
    )
    stage_times["knn_and_iso"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    global S_MIN, S_MAX
    if USE_ADAPTIVE_CLAMP:
        S_MIN, S_MAX = compute_adaptive_clamp(scales_iso)
        log("[INFO] Adaptive clamp enabled:")
        log(f"  ADAPTIVE_Q_MIN={ADAPTIVE_Q_MIN} -> S_MIN={S_MIN:.4f} m")
        log(f"  ADAPTIVE_Q_MAX={ADAPTIVE_Q_MAX} -> S_MAX={S_MAX:.4f} m")
    else:
        log("[INFO] Adaptive clamp disabled:")
        log(f"  Using fixed S_MIN={S_MIN:.4f} m, S_MAX={S_MAX:.4f} m")

    if S_MAX <= S_MIN:
        S_MAX = float(min(ABS_S_MAX, S_MIN * 2.0))
        log(f"[WARN] Adjusted S_MAX to keep it > S_MIN: S_MAX={S_MAX:.4f} m")
    stage_times["adaptive_clamp"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    if USE_BATCH_ANISO:
        scales_aniso, rotations, cov6 = compute_normal_aligned_gaussians_batched(
            points=points,
            scales_iso=scales_iso,
            normals=normals,
            k_tangent=K_NEIGHBORS_TANGENT,
            block_size=ANISO_BLOCK_SIZE,
            query_workers=CKDTREE_WORKERS,
            knn_idx_all=knn_idx_all if (knn_idx_all is not None and knn_idx_all.shape[1] >= K_NEIGHBORS_TANGENT) else None,
        )
    else:
        raise RuntimeError("This script version expects SciPy batched anisotropy. Install scipy or extend fallback path.")
    stage_times["anisotropy"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    if APPLY_UNITY_FRAME:
        log("[INFO] Applying ENU -> Unity frame transform (Z=North, Y=Up)...")
        points, rotations, cov6 = apply_frame_transform(points, rotations, cov6, M_ENU_TO_UNITY, backend=FRAME_TRANSFORM_BACKEND)
    stage_times["frame_transform"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    origin = np.zeros((3,), dtype=np.float32)
    if APPLY_RECENTER:
        origin = compute_recenter_origin(points, RECENTER_MODE)
        points = apply_recenter(points, origin)
        log("[INFO] Recentering enabled:")
        log(f"  RECENTER_MODE={RECENTER_MODE}")
        log(f"  origin (Unity frame) = [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]")
    stage_times["recenter"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    if VERBOSE:
        log("[INFO] Covariance diag stats (m^2): "
            f"xx[{cov6[:,0].min():.3e},{cov6[:,0].max():.3e}] "
            f"yy[{cov6[:,3].min():.3e},{cov6[:,3].max():.3e}] "
            f"zz[{cov6[:,5].min():.3e},{cov6[:,5].max():.3e}]")

    opacity = np.ones((points.shape[0],), dtype=np.float32)

    cov0 = np.zeros((points.shape[0], 4), dtype=np.float32)
    cov1 = np.zeros((points.shape[0], 4), dtype=np.float32)
    cov0[:, 0] = cov6[:, 0]
    cov0[:, 1] = cov6[:, 1]
    cov0[:, 2] = cov6[:, 2]
    cov0[:, 3] = cov6[:, 3]
    cov1[:, 0] = cov6[:, 4]
    cov1[:, 1] = cov6[:, 5]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    out_npz = output_npz_override if output_npz_override is not None else OUTPUT_PATH
    if output_npz_override is None and PRESET in ("verify", "final"):
        out_npz = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_{PRESET}{OUTPUT_PATH.suffix}")

    np.savez(
        out_npz,
        positions=points.astype(np.float32, copy=False),
        origin=origin.astype(np.float32, copy=False),
        origin_pre_shift=origin_pre_shift.astype(np.float32, copy=False),
        scales=scales_aniso.astype(np.float32, copy=False),
        scales_iso=scales_iso.astype(np.float32, copy=False),
        rotations=rotations.astype(np.float32, copy=False),
        cov6=cov6.astype(np.float32, copy=False),
        cov0=cov0,
        cov1=cov1,
        colors=colors.astype(np.float32, copy=False),
        opacity=opacity.astype(np.float32, copy=False),
        sampling_metadata=np.array(json.dumps(sampling_metadata), dtype=np.str_),
    )
    log(f"[INFO] Saved Gaussians to: {out_npz}")
    stage_times["pack_and_save"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    if WRITE_CHUNKS:
        log("[INFO] Writing chunks...")
        write_chunks_from_arrays(
            P=points.astype(np.float32, copy=False),
            C=colors.astype(np.float32, copy=False),
            cov6=cov6.astype(np.float32, copy=False),
            chunk_dir=CHUNK_DIR,
            chunk_prefix=CHUNK_PREFIX,
            chunk_size=CHUNK_SIZE.astype(np.float32, copy=False),
            eps=CHUNK_EPS,
        )
    stage_times["chunk_writing"] = time.perf_counter() - t_stage

    elapsed = time.perf_counter() - t_start

    log("[INFO] Stage runtime breakdown:")
    for key in ["load_pcd", "device_profile", "pre_shift", "sampling_and_normals", "autotune", "knn_and_iso", "adaptive_clamp", "anisotropy", "frame_transform", "recenter", "pack_and_save", "chunk_writing"]:
        if key in stage_times:
            v = stage_times[key]
            log(f"  - {key}: {v:.2f} s ({(100.0 * v / max(elapsed, 1e-9)):.1f}%)")

    log(f"[INFO] Total runtime: {elapsed:.2f} s ({elapsed / 60.0:.2f} min)")
    log("[INFO] Done.")

    return {
        "npz_path": str(out_npz),
        "preset": PRESET,
        "sampling": sampling_metadata,
        "num_points": int(points.shape[0]),
        "timing": stage_times,
        "total_sec": float(elapsed),
    }


def run_lod_gaussian_builds(levels: list[dict[str, object]]) -> None:
    global SAMPLE_MODE, MAX_GAUSSIANS, MAX_GAUSSIANS_AFTER_SAMPLING, UNIFORM_RESOLUTION, USER_OVERRIDE_MAX_POINTS, WRITE_CHUNKS

    manifest_t0 = time.perf_counter()
    manifest_entries = []
    base_output = OUTPUT_PATH
    if WRITE_CHUNKS:
        log("[WARN] --write_chunks is ignored during Gaussian LOD builds. Run chunking_builder.py with the generated manifest instead.")
        WRITE_CHUNKS = False

    for level_meta in levels:
        level = int(level_meta["level"])
        method = str(level_meta["sampling_method"]).lower().strip()
        label = str(level_meta["sampling_parameter_label"]).lower().replace(" ", "")

        SAMPLE_MODE = method
        if method == "random":
            MAX_GAUSSIANS = int(level_meta["sampling_parameter_value"])
            MAX_GAUSSIANS_AFTER_SAMPLING = -1
            USER_OVERRIDE_MAX_POINTS = True
        elif method == "uniform":
            UNIFORM_RESOLUTION = float(level_meta["sampling_parameter_value"])
            MAX_GAUSSIANS_AFTER_SAMPLING = -1
        else:
            raise ValueError(f"Unknown sampling method in LOD level: {method}")

        out_npz = base_output.with_name(f"{base_output.stem}_{PRESET}_L{level}_{label}{base_output.suffix}")
        log(f"[INFO] Building Gaussian L{level} ({method}, {label}) -> {out_npz}")
        summary = build_gaussians_once(
            output_npz_override=out_npz,
            lod_metadata={
                "lod_level": level,
                "lod_label": f"L{level}",
            },
        )
        manifest_entries.append({
            "level": level,
            "label": f"L{level}",
            "npz_path": str(out_npz),
            "sampling_method": method,
            "sampling_parameter_name": level_meta["sampling_parameter_name"],
            "sampling_parameter_value": level_meta["sampling_parameter_value"],
            "sampling_parameter_label": level_meta["sampling_parameter_label"],
            "num_points": summary["num_points"],
            "timing": summary["timing"],
            "total_sec": summary["total_sec"],
        })

    manifest = {
        "stage": "gaussian_lod_build",
        "input_path": str(INPUT_PATH),
        "output_root": str(base_output.parent),
        "preset": PRESET,
        "sampling_method": str(levels[0]["sampling_method"]) if levels else "",
        "levels": manifest_entries,
        "total_sec": float(time.perf_counter() - manifest_t0),
    }

    manifest_path = base_output.with_name(f"{base_output.stem}_{PRESET}_lod_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log(f"[INFO] Wrote Gaussian LOD manifest: {manifest_path}")


def main():
    if GAUSSIAN_LOD_LEVELS:
        run_lod_gaussian_builds(GAUSSIAN_LOD_LEVELS)
    else:
        build_gaussians_once()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(INPUT_PATH), help="Input point cloud path")
    ap.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="Output Gaussian NPZ path or base path for LOD manifests")
    ap.add_argument("--preset", default=PRESET_DEFAULT, choices=["verify", "final"])
    ap.add_argument("--quiet", action="store_true", help="Disable most logs for speed")
    ap.add_argument("--log_every", type=int, default=LOG_EVERY, help="Progress log interval")

    ap.add_argument("--no_batch_aniso", action="store_true", help="Disable SciPy batched anisotropy")
    ap.add_argument("--aniso_block", type=int, default=ANISO_BLOCK_SIZE, help="Batch size for anisotropy")
    ap.add_argument("--ckdtree_workers", type=int, default=None, help="Override SciPy cKDTree query workers")
    ap.add_argument("--knn_backend", type=str, default=KNN_BACKEND, choices=["auto", "cuda", "cpu"], help="KNN backend for Open3D tensor NNS")
    ap.add_argument("--knn_query_batch", type=int, default=KNN_QUERY_BATCH, help="Query batch size for tensor KNN")
    ap.add_argument("--knn_cache_mode", type=str, default=KNN_CACHE_MODE, choices=["auto", "full", "iso", "none"], help="How much KNN to cache for later stages")
    ap.add_argument("--frame_backend", type=str, default=FRAME_TRANSFORM_BACKEND, choices=["auto", "cuda", "cpu"], help="Frame transform backend")
    ap.add_argument("--frame_block", type=int, default=FRAME_TRANSFORM_BLOCK_SIZE, help="Batch size for frame transform")
    ap.add_argument("--no_auto_profile", action="store_true", help="Disable hardware-aware default configuration")
    ap.add_argument("--no_autotune", action="store_true", help="Disable startup micro-benchmark auto-tuning")
    ap.add_argument("--autotune_probe_points", type=int, default=AUTO_TUNE_PROBE_POINTS, help="Probe point count for startup auto-tuning")

    ap.add_argument("--sampling_method", default=SAMPLE_MODE, choices=["random", "uniform"], help="Sampling strategy used before Gaussian generation")
    ap.add_argument("--max_points", type=int, default=None, help="Random sampling target point count (-1=all)")
    ap.add_argument("--uniform_resolution", type=float, default=UNIFORM_RESOLUTION, help="Uniform voxel sampling resolution in meters")
    ap.add_argument("--no_intensity_color", action="store_true", help="Disable scalar_Intensity auxiliary RGB shading")
    ap.add_argument("--intensity_color_strength", type=float, default=INTENSITY_COLOR_STRENGTH, help="Strength for scalar_Intensity auxiliary RGB shading (0..1)")
    ap.add_argument("--lod_pyramid", action="store_true", help="Build the default sampling-based Gaussian LOD pyramid")
    ap.add_argument("--lod_specs", type=str, default="", help="Comma-separated LOD specs, for example L0=full,L1=10M,L2=1M or L0=1cm,L1=2cm")

    ap.add_argument("--write_chunks", action="store_true", help="Write chunk txt files + chunks_index.json")
    ap.add_argument("--chunk_dir", type=str, default=str(CHUNK_DIR), help="Chunk output directory")
    ap.add_argument("--chunk_prefix", type=str, default=CHUNK_PREFIX, help="Chunk filename prefix")
    ap.add_argument("--chunk_size", type=float, nargs=3, default=CHUNK_SIZE.tolist(), help="Chunk size dx dy dz")

    args = ap.parse_args()

    INPUT_PATH = Path(args.input)
    OUTPUT_PATH = Path(args.output)

    if args.quiet:
        VERBOSE = False
    LOG_EVERY = int(max(1, args.log_every))

    if args.no_batch_aniso:
        USE_BATCH_ANISO = False
    if args.aniso_block != ANISO_BLOCK_SIZE:
        USER_OVERRIDE_ANISO_BLOCK = True
    ANISO_BLOCK_SIZE = int(max(1000, args.aniso_block))

    if str(args.knn_backend).lower().strip() != KNN_BACKEND:
        USER_OVERRIDE_KNN_BACKEND = True
    KNN_BACKEND = str(args.knn_backend).lower().strip()

    if int(args.knn_query_batch) != KNN_QUERY_BATCH:
        USER_OVERRIDE_KNN_QUERY_BATCH = True
    KNN_QUERY_BATCH = int(max(4096, args.knn_query_batch))

    if str(args.knn_cache_mode).lower().strip() != KNN_CACHE_MODE:
        USER_OVERRIDE_KNN_CACHE_MODE = True
    KNN_CACHE_MODE = str(args.knn_cache_mode).lower().strip()

    if str(args.frame_backend).lower().strip() != FRAME_TRANSFORM_BACKEND:
        USER_OVERRIDE_FRAME_BACKEND = True
    FRAME_TRANSFORM_BACKEND = str(args.frame_backend).lower().strip()

    if int(args.frame_block) != FRAME_TRANSFORM_BLOCK_SIZE:
        USER_OVERRIDE_FRAME_BLOCK = True
    FRAME_TRANSFORM_BLOCK_SIZE = int(max(4096, args.frame_block))

    AUTO_PROFILE = not bool(args.no_auto_profile)
    AUTO_TUNE = not bool(args.no_autotune)
    AUTO_TUNE_PROBE_POINTS = int(max(2000, args.autotune_probe_points))

    apply_preset(args.preset)

    SAMPLE_MODE = str(args.sampling_method).lower().strip()
    UNIFORM_RESOLUTION = float(args.uniform_resolution)
    USE_INTENSITY_AUX_COLOR = not bool(args.no_intensity_color)
    INTENSITY_COLOR_STRENGTH = float(np.clip(args.intensity_color_strength, 0.0, 1.0))

    if args.max_points is not None:
        USER_OVERRIDE_MAX_POINTS = True
        MAX_GAUSSIANS = int(args.max_points)
        MAX_GAUSSIANS_AFTER_SAMPLING = -1
        log(f"[INFO] Overriding max_points -> {MAX_GAUSSIANS}")

    if args.ckdtree_workers is not None:
        USER_OVERRIDE_CKDTREE_WORKERS = True
        CKDTREE_WORKERS = int(max(1, args.ckdtree_workers))

    WRITE_CHUNKS = bool(args.write_chunks)
    CHUNK_DIR = Path(args.chunk_dir)
    CHUNK_PREFIX = str(args.chunk_prefix)
    CHUNK_SIZE = np.array(args.chunk_size, dtype=np.float32)

    if args.lod_pyramid or args.lod_specs:
        GAUSSIAN_LOD_LEVELS = parse_lod_specs(str(args.lod_specs), SAMPLE_MODE)
        log(f"[INFO] Gaussian LOD build enabled with {len(GAUSSIAN_LOD_LEVELS)} levels.")

    main()
