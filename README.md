# Geospatial Gaussian Point Cloud Visualization in VR

This repository contains the implementation of a master thesis project on **large-scale geospatial point cloud visualization in Virtual Reality**, based on **Gaussian primitives**, spatial chunking, level-of-detail (LOD) management, and GPU-based Gaussian splatting.

The system is designed to support immersive exploration of indoor and urban-scale datasets (e.g. NavVis, TUM2Twin), with a focus on scalability, performance, and perceptual visual quality.

---

## Project Title

**Immersive Geospatial Point Cloud Visualization in Virtual Reality Based on Gaussian Splatting**

---

## Overview

This project implements an end-to-end pipeline from raw point cloud data to an interactive real-time visualization system in Unity:

- Offline preprocessing and Gaussian primitive construction (Python)
- Spatial chunking and multi-level LOD generation
- Runtime chunk management with frustum culling and streaming
- GPU-based rendering using Gaussian primitives
- Quad-based Gaussian splatting (MVP implementation)

The current implementation focuses on a **NavVis indoor scan demo**, used as a minimal and controllable test case before scaling up to the full **TUM2Twin** datasets.

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

`gaussian_builder.py` converts the cleaned point cloud into Gaussian primitives.

Each point is represented as a Gaussian with:

- 3D position
- Scale (currently isotropic, with support for anisotropic extension)
- Color

**Input:** `.ply`  
**Output:**
- `.npz` — full Gaussian parameters (positions, scales, colors)
- `.txt` — simplified format for direct Unity loading (demo and chunk files)

Key parameters:
- `MAX_GAUSSIANS`: limits point count during development
- `K_NEIGHBORS_ISO / ANISO`: neighborhood size for scale estimation
- `S_MIN / S_MAX`: scale clamping for numerical stability

---

### 3. Spatial Chunking

`chunking_navvis.py` partitions the Gaussian point cloud into fixed-size 3D grid chunks.

For each chunk, it outputs:
- A chunk-specific `.txt` file
- Metadata including bounding box, center, and point count

A global **chunk index JSON** is generated and used by the Unity runtime to manage chunk loading.

**Input:** `.npz`  
**Output:** `navvis_chunk_ix_iy_iz.txt`, `chunk_index.json`

---

### 4. LOD Generation

`lod_builder.py` generates multiple levels of detail (LOD) for each chunk:

- **L0**: full-resolution Gaussian points
- **L1**: moderately downsampled
- **L2**: aggressively downsampled

A unified **LOD index JSON** extends the original chunk index with per-chunk LOD metadata.

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
- Supporting both point-based and quad-based Gaussian rendering
- Managing GPU memory lifecycle (load / unload)

Each loader maintains its **own runtime material instance**, preventing GPU buffer conflicts when rendering multiple chunks simultaneously.

---

### GaussianChunkManager.cs

`GaussianChunkManager` orchestrates all chunks at runtime:

1. Loads the LOD index JSON
2. Creates one GameObject + `GaussianLoader` per chunk
3. Performs **frustum culling** using chunk bounding boxes
4. Performs **distance-based LOD switching** (L0 / L1 / L2)
5. Implements **chunk streaming with hysteresis**:
   - Load chunks within `loadRadius`
   - Unload chunks beyond `unloadRadius`
6. Applies **LOD-aware Gaussian rendering parameters** to maintain perceptual stability across different point densities

This design enables scalable visualization of large point clouds while controlling GPU memory usage.

---

## GPU Rendering Shaders

### 1. GaussianPoints.shader (Point-Based Baseline)

`GaussianPoints.shader` implements a **point-based Gaussian rendering baseline**.

Characteristics:
- Renders Gaussian primitives using `MeshTopology.Points`
- Computes screen-space point size based on:
  - Per-point Gaussian scale
  - Camera distance
- Uses explicit local → world → clip space transformation to avoid double matrix application

This shader serves as:
- A stable baseline for debugging chunking, LOD, and streaming
- A reference implementation for correctness and performance comparisons

---

### 2. GaussianSplatQuads.shader (Gaussian Splatting MVP)

`GaussianSplatQuads.shader` implements the **quad-based Gaussian splatting renderer** used in the current system.

Each Gaussian primitive is expanded into a **camera-facing billboard quad**, and its contribution is evaluated using a Gaussian falloff in the fragment shader.

Key features:
- Quad expansion on the GPU (two triangles per Gaussian)
- Camera-facing billboards using view-space basis vectors
- Gaussian opacity evaluation in quad-local space
- Premultiplied alpha blending
- LOD-aware control of splat size, sharpness, and opacity

This shader represents a **Gaussian splatting MVP**, providing visually continuous surface approximation while remaining stable and debuggable.

> Note: This implementation does not yet include anisotropic covariance projection or depth-aware compositing, which are planned as future extensions.

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
- Point-based Gaussian rendering baseline
- Quad-based Gaussian splatting (MVP)
- LOD-aware splatting parameter control
- Stable NavVis indoor demo scene

### In Progress / Planned
- Depth-aware / weighted Gaussian compositing
- Anisotropic Gaussian splatting
- Compute-shader-based splatting pipeline
- Stereo rendering and VR integration (OpenXR, Vive Pro 2)
- Performance evaluation (FPS, GPU time, memory)
- Scale-up to TUM2Twin datasets

---

## Repository Structure

