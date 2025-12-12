Geospatial-Gaussian-VR

Immersive Geospatial Point Cloud Visualization in Virtual Reality Based on Gaussian Splatting

This repository contains the implementation of a chunked, LOD-enabled, and streamable Gaussian point cloud rendering pipeline, developed as part of a master’s thesis project.
The current version focuses on a NavVis indoor scan demo, providing a scalable runtime foundation for future Gaussian Splatting rendering and VR integration.

⸻

Project Overview

The goal of this project is to build a real-time, immersive visualization system for large-scale geospatial point clouds, targeting VR environments.
The system combines:
	•	Gaussian-based point representations
	•	Spatial chunking
	•	Distance-based LOD selection
	•	Frustum culling
	•	Runtime chunk streaming

The project is implemented using Python (offline preprocessing) and Unity (runtime rendering), and is designed to scale from small indoor scenes to large datasets such as TUM2Twin.

⸻

Current Status (v0.1 – NavVis Demo)

✅ Offline Gaussian construction pipeline
✅ Spatial chunking and LOD generation
✅ Unity runtime loading of chunked Gaussian point clouds
✅ Frustum culling
✅ Distance-based LOD switching (L0 / L1 / L2)
✅ Runtime chunk streaming (GPU buffer load / unload)
✅ Stable point-based rendering with per-chunk material instances

🚧 Gaussian splatting (soft splats) – not yet implemented
🚧 VR integration (OpenXR) – planned
🚧 TUM2Twin large-scale dataset migration – planned

⸻

Pipeline Overview

1. Data Preparation (CloudCompare)

Raw NavVis point clouds are preprocessed using CloudCompare:
	•	Segmentation (extract indoor blocks / rooms)
	•	Statistical Outlier Removal (SOR)
	•	Spatial Subsampling (density control)

Output: Cleaned .ply files

⸻

2. Gaussian Primitive Construction (Python)

gaussian_builder.py converts cleaned point clouds into Gaussian primitives.

Input:
	•	.ply

Output:
	•	.npz – full Gaussian parameters (position, scale, rotation, color)
	•	.txt – simplified Gaussian data for Unity demo rendering

Key parameters:
	•	MAX_GAUSSIANS
	•	K_NEIGHBORS_ISO / K_NEIGHBORS_ANISO
	•	S_MIN / S_MAX (scale clamping)

⸻

3. Spatial Chunking (Python)

chunking_navvis.py splits a Gaussian point cloud into fixed-size 3D grid chunks.

Input:
	•	Gaussian .npz

Output:
	•	navvis_chunk_ix_iy_iz.txt
	•	Chunk metadata
	•	Chunk index JSON (bounding boxes, centers, point counts)

⸻

4. LOD Generation (Python)

lod_builder.py generates multiple Levels of Detail for each chunk.

Output per chunk:
	•	*_L0.txt – full resolution
	•	*_L1.txt – subsampled
	•	*_L2.txt – coarse

A new navvis_chunks_lod_index.json is generated for runtime LOD and streaming.

⸻

5. Runtime Rendering (Unity)

GaussianLoader.cs
	•	Loads Gaussian .txt files
	•	Uploads data to GPU via ComputeBuffer
	•	Renders point clouds using DrawProceduralNow
	•	Supports reload / unload for LOD switching and streaming
	•	Uses per-instance material copies to avoid GPU buffer conflicts

GaussianChunkManager.cs
	•	Loads LOD index JSON
	•	Instantiates chunk GameObjects and GaussianLoaders
	•	Performs per-frame:
	•	Frustum culling
	•	Distance-based LOD selection
	•	Chunk streaming (load / unload)
	•	Provides global point size control

Shader
	•	Point-based rendering with:
	•	Distance-aware screen-space point sizing
	•	Per-point Gaussian scale (sx)
	•	Correct clip-space transformation (no double model matrix application)

⸻

Repository Structure

Geospatial-Gaussian-VR/
├── preprocessing/          # Python offline pipeline
├── vr-renderer/            # Unity project
│   ├── Assets/
│   │   ├── Scripts/
│   │   ├── Shaders/
│   │   └── StreamingAssets/
│   └── ProjectSettings/
├── docs/
├── README.md

Note: Large point cloud data and chunk files are intentionally excluded from version control.

⸻

Requirements

Python
	•	Python 3.8+
	•	numpy
	•	open3d

Unity
	•	Unity 2021 LTS or newer
	•	OpenGL / DirectX 11 compatible GPU

⸻

Roadmap
	•	Gaussian Splatting rendering (billboard / compute shader)
	•	Stereo rendering and VR integration (OpenXR, Vive Pro 2)
	•	Performance evaluation and benchmarking
	•	Migration to TUM2Twin large-scale datasets

⸻

License

This project is developed for academic research purposes.
Dataset licenses (NavVis, TUM2Twin) apply separately.

⸻

Acknowledgements
	•	Kerbl et al., 3D Gaussian Splatting for Real-Time Radiance Field Rendering
	•	CityGaussian / CityGaussianV2
	•	TUM2Twin Project
	•	Open3D
	•	Unity Technologies

git push origin main

如果你愿意，下一步我可以帮你把 README 直接精炼成 thesis Chapter 3（Implementation）英文版，或者写一个 docs/pipeline.md 图文说明版。
