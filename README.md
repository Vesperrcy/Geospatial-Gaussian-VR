# Geospatial Gaussian Point Cloud Visualization in VR

This repository contains the implementation of a master thesis project on **large-scale geospatial point cloud visualization in Virtual Reality**, based on **Gaussian primitives**, spatial chunking, level-of-detail (LOD) management, and runtime streaming.

The system is designed to support immersive VR exploration of indoor and urban-scale datasets (e.g. NavVis, TUM2Twin), with a focus on scalability, performance, and visual quality.

---

## Project Title

**Immersive Geospatial Point Cloud Visualization in Virtual Reality Based on Gaussian Splatting**

---

## Overview

This project builds a complete pipeline from raw point cloud data to an interactive VR visualization system:

- Offline preprocessing and Gaussian primitive construction (Python)
- Spatial chunking and multi-level LOD generation
- Runtime chunk management with frustum culling and streaming
- GPU-based point cloud rendering in Unity
- (Upcoming) Gaussian splatting and VR integration via OpenXR

The current version focuses on a **NavVis indoor scan demo**, used as a minimal, controllable test case before scaling up to the full **TUM2Twin** dataset.

---

## Pipeline Summary

### 1. Data Preparation (CloudCompare)

Raw point cloud data is preprocessed using CloudCompare:

- Segmentation (extracting a single room / block)
- Statistical Outlier Removal (SOR)
- Spatial Subsampling (uniform density control)

**Output:** cleaned point cloud (`.ply`)

---

### 2. Gaussian Primitive Construction (Python)

`gaussian_builder.py` converts the cleaned point cloud into Gaussian primitives suitable for rendering and research.

Each point is represented as a Gaussian with:

- Position
- Scale (isotropic, with support for anisotropic extension)
- Color

**Input:** `.ply`  
**Output:**
- `.npz` — full Gaussian parameters (positions, scales, colors)
- `.txt` — simplified format for direct Unity loading (demo)

Key parameters:
- `MAX_GAUSSIANS`: limit point count during development
- `K_NEIGHBORS_ISO / ANISO`: neighborhood size for scale estimation
- `S_MIN / S_MAX`: scale clamping for numerical stability

---

### 3. Spatial Chunking

`chunking_navvis.py` partitions the Gaussian point cloud into fixed-size 3D grid chunks.

For each chunk, it outputs:
- A chunk-specific `.txt` file
- Metadata including bounding box, center, and point count

A global **chunk index JSON** is generated, which serves as the runtime directory for Unity.

**Input:** `.npz`  
**Output:** `navvis_chunk_ix_iy_iz.txt`, `chunk_index.json`

---

### 4. LOD Generation

`lod_builder.py` generates multiple levels of detail (LOD) for each chunk:

- **L0**: full-resolution Gaussian points
- **L1**: downsampled version
- **L2**: aggressively downsampled version

A unified **LOD index JSON** is produced, extending the original chunk index with per-chunk LOD information.

**Output:**
- `navvis_chunk_ix_iy_iz_L0.txt`
- `navvis_chunk_ix_iy_iz_L1.txt`
- `navvis_chunk_ix_iy_iz_L2.txt`
- `navvis_chunks_lod_index.json`

---

## Unity Runtime System

### GaussianLoader.cs

`GaussianLoader` is the lowest-level runtime rendering component.  
It is responsible for:

- Loading a single Gaussian point cloud file (`.txt`)
- Uploading positions, colors, and scales to GPU `ComputeBuffer`s
- Rendering via `DrawProceduralNow`
- Managing GPU memory lifecycle (load / unload)

Each loader maintains its **own material instance** to avoid GPU buffer conflicts when rendering multiple chunks simultaneously.

---

### GaussianChunkManager.cs

`GaussianChunkManager` orchestrates all chunks at runtime:

1. Loads the LOD index JSON
2. Creates one GameObject + `GaussianLoader` per chunk
3. Performs **frustum culling** based on chunk bounding boxes
4. Performs **distance-based LOD switching** (L0 / L1 / L2)
5. Implements **chunk streaming** with hysteresis:
   - Load chunks within `loadRadius`
   - Unload chunks beyond `unloadRadius`

This design enables scalable rendering of large point clouds while controlling GPU memory usage.

---

### Current Rendering

- GPU-based point rendering using `MeshTopology.Points`
- Screen-space point size adapts to distance and per-point Gaussian scale
- Corrected clip-space transformation to avoid double application of model matrices

This stage serves as a stable baseline before introducing full Gaussian splatting.

---

## Current Status

### Completed
- CloudCompare preprocessing pipeline
- Gaussian primitive construction
- Spatial chunking and LOD generation
- Unity runtime chunk loading
- Frustum culling
- Distance-based LOD switching
- Chunk streaming with GPU buffer unloading
- Stable NavVis indoor demo scene

### In Progress / Upcoming
- Gaussian splatting (billboard / compute-shader based)
- Stereo rendering and VR integration (OpenXR, Vive Pro 2)
- Performance evaluation (FPS, GPU time, memory)
- Scale-up to TUM2Twin datasets

---

## Repository Structure
