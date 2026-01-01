# Geospatial Gaussian Point Cloud Visualization in VR

This repository contains the implementation of a master thesis project on **large-scale geospatial point cloud visualization in Virtual Reality**, based on **Gaussian primitives**, spatial chunking, level-of-detail (LOD) management, and **GPU-based Gaussian splatting**.

The system is designed to support immersive exploration of indoor and urban-scale datasets (e.g. **NavVis**, **TUM2Twin**), with a focus on scalability, rendering correctness, and perceptual visual quality.

---

## Project Title

**Immersive Geospatial Point Cloud Visualization in Virtual Reality Based on Gaussian Splatting**

---

## Overview

This project implements an end-to-end pipeline from raw point cloud data to an interactive real-time visualization system in Unity:

* Offline preprocessing and Gaussian construction (Python)
* Spatial chunking and multi-level LOD generation
* Runtime chunk management with frustum culling and streaming
* GPU-based rendering using Gaussian primitives
* **Covariance-based Gaussian splatting (current MVP)**

The current implementation focuses on a **NavVis indoor scan demo**, used as a minimal and controllable test case before scaling up to the full **TUM2Twin** datasets.

---

## 🚀 Recent Progress: Gaussian Splatting MVP (NEW)

The project has reached a **Gaussian Splatting MVP milestone**, upgrading the system from isotropic point/quad rendering to **anisotropic, covariance-based Gaussian splatting**.

### Summary of New Capabilities

* Each point is represented as a **full 3D Gaussian** with an explicit covariance matrix Σ
* Gaussians are projected into **screen-space ellipses** and rendered as elliptical splats
* Rendering is **SRP-safe** and compatible with **Unity 6 / Metal**
* Chunking, LOD switching, and streaming fully support covariance-based splats

This stage validates the **geometric correctness and engineering feasibility** of Gaussian splatting in Unity.

---

## Pipeline Summary

### 1. Data Preparation (CloudCompare)

Raw point cloud data is preprocessed using CloudCompare:

* Segmentation (extracting a single room / block)
* Statistical Outlier Removal (SOR)
* Spatial Subsampling (uniform density control)

**Output:** cleaned point cloud (`.ply`)

---

### 2. Gaussian Primitive Construction (Python)

`gaussian_builder.py` converts the cleaned point cloud into **anisotropic Gaussian primitives**.

Each point is represented as a Gaussian with:

* 3D position
* RGB color
* **Full 3D covariance matrix Σ**

The covariance is stored explicitly using its **6 independent elements**:

```
xx, xy, xz, yy, yz, zz
```

**Input:** `.ply`

**Output:**

* `.npz` — full Gaussian parameters
* `.txt` — simplified **12-column format** for Unity runtime:

```
x y z r g b xx xy xz yy yz zz
```

Key parameters:

* `MAX_GAUSSIANS`
* `K_NEIGHBORS_ISO / ANISO`
* `S_MIN / S_MAX`

---

### 3. Spatial Chunking

`chunking_navvis.py` partitions the Gaussian cloud into fixed-size 3D grid chunks.

For each chunk:

* A chunk-specific `.txt` file (12-column Gaussian format)
* Metadata including bounding box, center, and point count

A global **chunk index JSON** is generated and consumed by the Unity runtime.

---

### 4. LOD Generation

`lod_builder.py` generates multiple levels of detail (LOD) for each chunk:

* **L0**: full-resolution Gaussians
* **L1**: moderately downsampled
* **L2**: aggressively downsampled

All LOD levels preserve the **full covariance representation**.

---

## Unity Runtime System

### GaussianLoader.cs (Updated)

`GaussianLoader` is the lowest-level runtime rendering component.

Recent updates:

* Loads **covariance-based Gaussian data (Σ)**
* Uploads positions, colors, and covariance to GPU `ComputeBuffer`s
* Uses **SRP-safe rendering** via `RenderPipelineManager` + `CommandBuffer`
* Uses `MaterialPropertyBlock` for explicit per-draw buffer binding (Metal-safe)
* Supports quad-based **elliptical Gaussian splatting**

Each loader maintains its **own runtime material instance** to avoid GPU buffer conflicts.

---

### GaussianChunkManager.cs (Updated)

`GaussianChunkManager` orchestrates all chunks at runtime:

1. Loads the LOD index JSON
2. Creates one GameObject + `GaussianLoader` per chunk
3. Performs **frustum culling** using chunk bounding boxes
4. Performs **distance-based LOD switching** (L0 / L1 / L2)
5. Implements **chunk streaming with hysteresis**
6. Applies **LOD-aware Gaussian splatting parameters**

The chunk system is fully compatible with covariance-based splats.

---

## GPU Rendering Shaders

### 1. GaussianPoints.shader (Baseline)

Point-based Gaussian rendering used as a correctness and debugging baseline.

---

### 2. GaussianSplatQuads.shader (Covariance-Based MVP)

`GaussianSplatQuads.shader` implements **anisotropic Gaussian splatting**:

* Projects **3D Gaussian covariance → 2D screen-space ellipse**
* Computes ellipse axes via projected covariance
* Expands each Gaussian into a camera-facing quad
* Evaluates Gaussian falloff in the fragment shader

This produces **elliptical Gaussian splats**, rather than isotropic point sprites.

> Note: The current implementation uses simple alpha compositing. Depth-aware and order-independent compositing are intentionally deferred.

---

## Current Status

### Completed

* CloudCompare preprocessing pipeline
* Anisotropic Gaussian construction with full covariance
* Spatial chunking and LOD generation (covariance-compatible)
* Unity runtime chunk loading and streaming
* Frustum culling
* Distance-based LOD switching
* SRP-safe Gaussian splat rendering (Unity 6 / Metal)
* Covariance-based elliptical Gaussian splatting MVP
* Stable NavVis indoor demo

### In Progress / Planned

* Depth-aware / weighted Gaussian compositing
* Order-independent transparency (OIT)
* Improved LOD transition stability
* Performance evaluation (FPS, GPU time, memory)
* Stereo VR rendering (OpenXR)
* Scale-up to TUM2Twin datasets

---

## Repository Structure

```
preprocessing/
  gaussian_builder.py      # Build anisotropic Gaussians (full covariance Σ)
  chunking_navvis.py       # Spatial chunking (12-column Gaussian format)
  lod_builder.py           # Multi-level LOD generation

vr-renderer/Assets/
  Scripts/
    GaussianLoader.cs      # SRP-safe covariance-based Gaussian splatting
    GaussianChunkManager.cs
  Shaders/
    GaussianSplatQuads.shader
```

---

## Design Philosophy

* Build a **minimal, correct, and debuggable MVP first**
* Separate preprocessing, runtime management, and rendering logic
* Prefer explicit covariance representations over implicit reconstruction
* Treat rendering quality improvements (compositing, OIT) as controlled extensions

---

This repository now contains a **working covariance-based Gaussian Splatting MVP**, forming a solid foundation for further research and thesis writing.
