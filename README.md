# Geospatial-Gaussian-VR
Immersive VR visualization of large-scale geospatial point clouds using Gaussian Splatting, Unity OpenXR, and geospatial processing (TUM Geodesy &amp; Geoinformation Master Thesis).

📌 Current Pipeline Overview (NavVis → Gaussian → Unity GPU Renderer)

This repository implements a complete end-to-end pipeline for transforming real-world geospatial point clouds into Gaussian primitives and rendering them in Unity using GPU procedural drawing.
The current stage provides a working demo using NavVis VLX indoor–outdoor scan data (“House 2 / Lichtblick”).

⸻

1. Point Cloud Preprocessing (CloudCompare)

Input:
SampleHouse1ScannedWithNavVisVLX.e57

Processing steps performed in CloudCompare:
	1.	Import raw point cloud
File → Open → .e57
	2.	Noise filtering
Edit → Noise → SOR Filter
Removes outliers and scanning artifacts.
	3.	Spatial downsampling
Edit → Subsample → Spatial
Reduces point density while keeping uniform structure.
	4.	Shift to local origin
Global navigation coordinate → local ENU-like system:
Edit → Apply Transformation → Translate → -GlobalShift
	5.	Export to PLY
File → Save → .ply
Exported file used for Gaussian construction:
data/navvis_house2_centered.ply

✔ Result: Clean, normalized, centered point cloud ready for Python processing.

⸻

2. Gaussian Construction (Python + Open3D)

Script:
preprocessing/gaussian_builder.py

Pipeline Steps
	1.	Load centered PLY using Open3D
	2.	Random sampling of up to 50,000 points for real-time demo
	3.	Compute local Gaussian scales
	•	Build KDTree
	•	For each point:
	•	Query K=8 nearest neighbors
	•	Estimate local density → derive isotropic Gaussian scale
	4.	Initialize rotations as identity (anisotropic Gaussians planned for future stage)
	5.	Normalize colors to [0,1]
	6.	Save Gaussian parameters in two formats:
	•	navvis_house2_gaussians_demo.npz
(full research format — positions, scales, rotations, opacity)
	•	navvis_house2_gaussians_demo.txt
(Unity procedural renderer input)

Unity TXT Format

Each line:

x  y  z   sx  sy  sz   r  g  b

✔ Result: Lightweight Gaussian point representation suitable for fast GPU loading.

⸻

3. Unity GPU Point Renderer (ComputeBuffer + Procedural Draw)

Project:
vr-renderer/

Key components

3.1 Data loading (C#)
Script: Assets/Scripts/GaussianLoader.cs
	•	Reads TXT from Assets/StreamingAssets/
	•	Parses position and color arrays
	•	Uploads to GPU via ComputeBuffer
	•	Sets shader uniforms:
	•	_Positions
	•	_Colors
	•	_PointSize
	•	_LocalToWorld

3.2 Shader (HLSL)
File: Assets/Shaders/GaussianPoints.shader
	•	Uses StructuredBuffer<float3> for positions & colors
	•	Vertex stage computes clip-space location
	•	Fragment pass outputs per-point color
	•	Procedural drawing:

Graphics.DrawProceduralNow(MeshTopology.Points, numPoints);

3.3 Scene Setup
	•	Empty GameObject GaussianRenderer
	•	Position (0,0,0)
	•	Rotation (-90,0,0) to convert Z-up → Unity Y-up
	•	Attached script: GaussianLoader.cs
	•	navvis_house2_gaussians_demo.txt placed in StreamingAssets

3.4 Runtime
	•	Press Play → the NavVis Gaussian cloud is rendered in real-time.

✔ Result: Working real-time Gaussian point renderer with Unity GPU pipeline.

⸻

4. Current Capabilities
	•	End-to-end NavVis point cloud pipeline
	•	Gaussian primitive generation (isotropic v0)
	•	Real-time rendering (50k points)
	•	GPU procedural pipeline (ComputeBuffer + DrawProceduralNow)
	•	Axis-aligned, scaled, colored point cloud
	•	Camera aligned using SceneView → Align With View workflow
	•	Stable reproducible demo scene

⸻

5. Next Steps (Planned in Thesis)
	•	⚡ True Gaussian Splatting (screen-space footprint + falloff)
	•	⚡ VR Integration (OpenXR + Vive Pro 2)
	•	⚡ Chunking + LOD system for large geospatial clouds
	•	⚡ Anisotropic covariance-based Gaussians
	•	⚡ Real-time frustum culling & streaming
	•	⚡ Support for TUM2Twin city-scale datasets

