using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class GaussianLoader : MonoBehaviour
{   
    //Step1: Inspector fields define rendering behavior
    [Header("Data file (relative to StreamingAssets)")]
    public string dataFileName = "TumTLS_v2_gaussians_demo.txt";

    [Header("Rendering")]
    public Material pointMaterial;

    [Tooltip("Use the shared material asset directly so changing its Inspector sliders updates rendering in real-time. If false, a runtime clone is created (Inspector edits to the asset will NOT affect the clone during Play Mode).")]
    public bool useSharedMaterialAsset = true;

    [Tooltip("If true and useSharedMaterialAsset is enabled, a runtime material clone is still created for safe per-chunk buffer binding (avoids Metal ComputeBuffer missing / cross-chunk conflicts). The clone will mirror the asset's numeric parameters each frame.")]
    public bool forceRuntimeMaterialForBuffers = true;

    [Tooltip("Apply this component's per-chunk BASIC shader params (Opacity/SigmaCutoff/MinAxis/MaxAxis/PointSize) every draw. Disable if you want to drive these entirely from the Material asset.")]
    public bool overrideBasicShaderParams = true;

    [Tooltip("Apply this component's per-chunk OIT/near-comp params every draw. Disable if you want to drive these entirely from the Material asset.")]
    public bool overrideOITShaderParams = false;

    [Tooltip("If enabled, render each point as a camera-facing quad with Gaussian alpha (splatting).")]
    public bool renderAsSplatQuads = true;

    [Header("Splatting (shader params)")]
    [Range(0f, 1f)] public float opacity = 0.7f;
    [Tooltip("k-sigma cutoff used to size the ellipse quad (typical 2~4).")]
    public float sigmaCutoff = 3.0f;
    [Tooltip("Clamp ellipse axis in pixel units (min).")]
    public float minAxisPixels = 0.75f;
    [Tooltip("Clamp ellipse axis in pixel units (max).")]
    public float maxAxisPixels = 64.0f;
    public float pointSize = 1.0f;

    [Header("Scale Mixture / Fade (optional)")]
    [Tooltip("Multiply final opacity sent to shader (useful for scale-mixture gating).")]
    public float opacityMultiplier = 1.0f;

    [Tooltip("Multiply final pointSize sent to shader (useful for scale-mixture coarse footprint).")]
    public float pointSizeMultiplier = 1.0f;

    [Tooltip("If > 0, apply a camera view-space Z fade in the shader using _ViewZFadeStart/_ViewZFadeEnd.")]
    public bool enableViewZFade = false;

    [Tooltip("View-space Z where fade begins (meters). For coarse pass you typically set this to mixtureStart.")]
    public float viewZFadeStart = 0.0f;

    [Tooltip("View-space Z where fade reaches 1 (meters). For coarse pass you typically set this to mixtureEnd.")]
    public float viewZFadeEnd = 10.0f;

    [Tooltip("Fade curve exponent. 1=linear, >1 makes the fade more concentrated near the end.")]
    [Range(0.25f, 8f)]
    public float viewZFadeExponent = 1.0f;

    [Tooltip("If true, invert the fade (1->0). Useful for fine pass when you want it to vanish in the far range.")]
    public bool invertViewZFade = false;

    [Header("OIT / Density Compensation (shader params)")]
    [Range(0.1f, 8f)] public float oitWeightExponent = 2.0f;
    [Range(0f, 8f)] public float depthWeight = 1.0f;
    [Range(0.1f, 8f)] public float depthExponent = 2.0f;

    [Range(0.1f, 50f)] public float nearCompRefZ = 6.0f;
    [Range(0f, 8f)] public float nearCompStrength = 0.0f;
    [Range(0f, 20f)] public float nearCompMin = 1.0f;
    [Range(0f, 20f)] public float nearCompMax = 8.0f;

    [Header("Chunk options")]
    public bool treatInputAsWorldSpace = false;
    public Vector3 chunkCenterWorld = Vector3.zero;

    [Header("OIT Integration")]
    [Tooltip("If enabled, this loader will render via GaussianOITFeature passes (Weighted Blended OIT).")]
    public bool preferOIT = true;

    [Tooltip("Render into SceneView as well as GameView (useful for debugging).")]
    public bool renderInSceneView = true;

    [Header("Debug")]
    public bool verboseDebug = false;

    [Header("Performance / LOD")]
    [Tooltip("Cache parsed point data by filename to reduce hitching when switching LOD or reloading chunks.")]
    public bool enableDataCache = true;

    [Tooltip("Allocate GPU buffers with extra capacity (power-of-two growth) to reduce reallocations during LOD switches.")]
    
    //Step 2: Script store data om GPU buffers
    public bool growBuffersByPowerOfTwo = true;

    private ComputeBuffer _positionBuffer;
    private ComputeBuffer _colorBuffer;
    private ComputeBuffer _cov0Buffer;
    private ComputeBuffer _cov1Buffer;

    private int _numPoints;
    private int _bufferCapacity = 0;
    private Material _runtimeMaterial;
    private MaterialPropertyBlock _mpb;
    private Material _sourceMaterialAsset;
    private bool _mirrorAssetParams = false;

    private bool _srpHooked = false;
    private bool _oitHooked = false;
    private bool _directPassHooked = false;

    private const int Float3Stride = sizeof(float) * 3;
    private const int Float4Stride = sizeof(float) * 4;
    private const float FallbackCov = 0.0001f;

    private sealed class CachedData
    {
        public Vector3[] positions;
        public Vector3[] colors;
        public Vector4[] cov0;
        public Vector4[] cov1;
        public int count;
        public DateTime lastWriteUtc;
    }

    // Static cache for parsed data by filename (with write time check for invalidation)
    private static readonly Dictionary<string, CachedData> s_DataCache = new();

    // Utility to round up to next power of two (for buffer growth strategy)    
    private static int NextPow2(int v)
    {
        v = Mathf.Max(1, v);
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }

    // Utility to get full path for a filename in StreamingAssets (handles platform differences)
    private static string FullPathForStreamingAsset(string fileName)
        => Path.Combine(Application.streamingAssetsPath, fileName);

    private bool UsesTriangleQuadRendering()
    {
        if (renderAsSplatQuads)
            return true;

        return _runtimeMaterial != null && _runtimeMaterial.shader != null &&
               _runtimeMaterial.shader.name == "Unlit/RawPointCloud";
    }

    //Step 3: Initialize material instance and hook into rendering pipeline
    private void InitializeMaterialInstance()
    {
        if (pointMaterial == null)
        {
            _sourceMaterialAsset = null;
            _runtimeMaterial = null;
            _mirrorAssetParams = false;
            return;
        }

        _sourceMaterialAsset = pointMaterial;

        if (_runtimeMaterial != null && !ReferenceEquals(_runtimeMaterial, _sourceMaterialAsset))
            Destroy(_runtimeMaterial);

        if (useSharedMaterialAsset && forceRuntimeMaterialForBuffers)
        {
            _runtimeMaterial = new Material(pointMaterial);
            _mirrorAssetParams = true;
        }
        else
        {
            _runtimeMaterial = useSharedMaterialAsset ? pointMaterial : new Material(pointMaterial);
            _mirrorAssetParams = false;
        }
    }

    //Step 4: Render hooks decide when chunks shoud be drawn
    private void RefreshRenderHooks()
    {
        if (_srpHooked)
        {
            RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
            _srpHooked = false;
        }

        if (_oitHooked)
        {
            GaussianOITFeature.OnDrawGaussians -= DrawInOIT_Compat;
            GaussianOITFeature.OnDrawGaussiansRG -= DrawInOIT_RG;
            _oitHooked = false;
        }

        if (_directPassHooked)
        {
            GaussianOITFeature.OnDrawDirect -= DrawDirect_Compat;
            GaussianOITFeature.OnDrawDirectRG -= DrawDirect_RG;
            _directPassHooked = false;
        }

        if (!isActiveAndEnabled)
            return;

        if (preferOIT)
        {
            try
            {
                GaussianOITFeature.OnDrawGaussians += DrawInOIT_Compat;
                GaussianOITFeature.OnDrawGaussiansRG += DrawInOIT_RG;
                _oitHooked = true;
            }
            catch
            {
                _oitHooked = false;
            }
        }
        else
        {
            try
            {
                GaussianOITFeature.OnDrawDirect += DrawDirect_Compat;
                GaussianOITFeature.OnDrawDirectRG += DrawDirect_RG;
                _directPassHooked = true;
            }
            catch
            {
                _directPassHooked = false;
            }
        }

        if (!_oitHooked && !_directPassHooked)
        {
            RenderPipelineManager.endCameraRendering += OnEndCameraRendering;
            _srpHooked = true;
        }
    }

    
    public void ConfigureRendering(Material baseMaterial, bool drawAsSplatQuads, bool useOIT)
    {
        bool sameMaterial = ReferenceEquals(pointMaterial, baseMaterial);
        bool sameMode = renderAsSplatQuads == drawAsSplatQuads;
        bool sameOIT = preferOIT == useOIT;
        if (sameMaterial && sameMode && sameOIT && _runtimeMaterial != null)
            return;

        pointMaterial = baseMaterial;
        renderAsSplatQuads = drawAsSplatQuads;
        preferOIT = useOIT;

        InitializeMaterialInstance();
        RefreshRenderHooks();
    }

    //Step 5: Initialize system rendering
    private void Start()
    {
        if (pointMaterial == null)
        {
            Debug.LogError("[GaussianLoader] pointMaterial is not assigned!");
            return;
        }

        InitializeMaterialInstance();
        _mpb = new MaterialPropertyBlock();

        if (treatInputAsWorldSpace)
            transform.position = chunkCenterWorld;

        LoadData();
    }

    // If using a runtime material instance that mirrors the asset, copy parameters each frame
    private void OnValidate()
    {
        if (pointMaterial == null) return;
        _sourceMaterialAsset = pointMaterial;

        if (!Application.isPlaying)
        {
            _runtimeMaterial = useSharedMaterialAsset ? pointMaterial : null;
            _mirrorAssetParams = false;
        }
    }

    // If mirroring asset parameters to a runtime instance, do it in LateUpdate to ensure it happens after any Inspector changes
    private void LateUpdate()
    {
        if (!_mirrorAssetParams) return;
        if (_sourceMaterialAsset == null || _runtimeMaterial == null) return;
        if (ReferenceEquals(_runtimeMaterial, _sourceMaterialAsset)) return;

        _runtimeMaterial.CopyPropertiesFromMaterial(_sourceMaterialAsset);
    }

    private void OnEnable() => RefreshRenderHooks();

    private void OnDisable()
    {
        if (_srpHooked)
        {
            RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
            _srpHooked = false;
        }

        if (_oitHooked)
        {
            GaussianOITFeature.OnDrawGaussians -= DrawInOIT_Compat;
            GaussianOITFeature.OnDrawGaussiansRG -= DrawInOIT_RG;
            _oitHooked = false;
        }

        if (_directPassHooked)
        {
            GaussianOITFeature.OnDrawDirect -= DrawDirect_Compat;
            GaussianOITFeature.OnDrawDirectRG -= DrawDirect_RG;
            _directPassHooked = false;
        }
    }

    private bool ShouldRenderForCamera(Camera cam)
    {
        if (cam == null) return false;
        if (cam.cameraType == CameraType.Game) return true;
        if (renderInSceneView && cam.cameraType == CameraType.SceneView) return true;
        return false;
    }

    private void OnEndCameraRendering(ScriptableRenderContext context, Camera cam)
    {
        if (!isActiveAndEnabled) return;
        if (!ShouldRenderForCamera(cam)) return;
        if (_oitHooked || _directPassHooked) return;
        RenderForCamera(context, cam);
    }

    private void OnRenderObject()
    {
        if (GraphicsSettings.currentRenderPipeline != null)
            return;

        if (_oitHooked || _directPassHooked) return;

        var cam = Camera.current;
        if (!ShouldRenderForCamera(cam)) return;
        RenderForCamera(default, cam);
    }

    private void EnsureCovBuffersAllocated()
    {
        int cap = Mathf.Max(1, _bufferCapacity > 0 ? _bufferCapacity : _numPoints);
        if (_cov0Buffer == null)
            _cov0Buffer = new ComputeBuffer(cap, Float4Stride, ComputeBufferType.Structured);
        if (_cov1Buffer == null)
            _cov1Buffer = new ComputeBuffer(cap, Float4Stride, ComputeBufferType.Structured);
    }

    private bool MaterialNeedsCovBuffers()
    {
        if (_runtimeMaterial == null) return false;
        if (renderAsSplatQuads) return true;
        return _runtimeMaterial.HasProperty("_Cov0") || _runtimeMaterial.HasProperty("_Cov1");
    }

    // send para to shader
    private void BuildAndBindMPB(MaterialPropertyBlock mpb)
    {
        if (mpb == null) return;

        mpb.Clear();
        mpb.SetMatrix("_LocalToWorld", transform.localToWorldMatrix);

        if (overrideBasicShaderParams)
        {
            float finalOpacity = opacity * Mathf.Max(0.0f, opacityMultiplier);
            mpb.SetFloat("_Opacity", finalOpacity);
            mpb.SetFloat("_SigmaCutoff", sigmaCutoff);

            float finalPointSize = pointSize * Mathf.Max(0.0f, pointSizeMultiplier);
            if (!renderAsSplatQuads)
            {
                mpb.SetFloat("_PointSize", Mathf.Max(0.0f, finalPointSize));
            }
            else
            {
                if (finalPointSize <= 1e-4f)
                {
                    mpb.SetFloat("_PointSize", 0f);
                    mpb.SetFloat("_MinAxisPixels", 0f);
                    mpb.SetFloat("_MaxAxisPixels", 0f);
                }
                else
                {
                    mpb.SetFloat("_PointSize", 1.0f);
                    mpb.SetFloat("_MinAxisPixels", minAxisPixels * finalPointSize);
                    mpb.SetFloat("_MaxAxisPixels", maxAxisPixels * finalPointSize);
                }
            }
        }

        if (overrideOITShaderParams)
        {
            mpb.SetFloat("_WeightExponent", oitWeightExponent);
            mpb.SetFloat("_DepthWeight", depthWeight);
            mpb.SetFloat("_DepthExponent", depthExponent);
            mpb.SetFloat("_NearCompRefZ", nearCompRefZ);
            mpb.SetFloat("_NearCompStrength", nearCompStrength);
            mpb.SetFloat("_NearCompMin", nearCompMin);
            mpb.SetFloat("_NearCompMax", nearCompMax);
        }

        mpb.SetFloat("_ViewZFadeEnabled", enableViewZFade ? 1.0f : 0.0f);
        mpb.SetFloat("_ViewZFadeStart", viewZFadeStart);
        mpb.SetFloat("_ViewZFadeEnd", viewZFadeEnd);
        mpb.SetFloat("_ViewZFadeExponent", viewZFadeExponent);
        mpb.SetFloat("_ViewZFadeInvert", invertViewZFade ? 1.0f : 0.0f);

        mpb.SetBuffer("_Positions", _positionBuffer);
        mpb.SetBuffer("_Colors", _colorBuffer);
        if (!ReferenceEquals(_runtimeMaterial, _sourceMaterialAsset))
        {
            _runtimeMaterial.SetBuffer("_Positions", _positionBuffer);
            _runtimeMaterial.SetBuffer("_Colors", _colorBuffer);
        }

        if (MaterialNeedsCovBuffers())
        {
            EnsureCovBuffersAllocated();
            mpb.SetBuffer("_Cov0", _cov0Buffer);
            mpb.SetBuffer("_Cov1", _cov1Buffer);

            if (!ReferenceEquals(_runtimeMaterial, _sourceMaterialAsset))
            {
                _runtimeMaterial.SetBuffer("_Positions", _positionBuffer);
                _runtimeMaterial.SetBuffer("_Colors", _colorBuffer);
                _runtimeMaterial.SetBuffer("_Cov0", _cov0Buffer);
                _runtimeMaterial.SetBuffer("_Cov1", _cov1Buffer);
            }
        }
    }

    // Separate draw methods for OIT vs direct rendering paths (called from respective hooks)
    private void DrawInOIT_Compat(CommandBuffer cmd, Camera cam)
    {
        if (!isActiveAndEnabled || !ShouldRenderForCamera(cam) || _runtimeMaterial == null || _numPoints <= 0) return;
        if (_positionBuffer == null || _colorBuffer == null) return;
        if (_mpb == null) _mpb = new MaterialPropertyBlock();
        BuildAndBindMPB(_mpb);

    
        bool useQuads = UsesTriangleQuadRendering();
        int vertexCount = useQuads ? (_numPoints * 6) : _numPoints;
        MeshTopology topo = useQuads ? MeshTopology.Triangles : MeshTopology.Points;
        cmd.DrawProcedural(Matrix4x4.identity, _runtimeMaterial, 0, topo, vertexCount, 1, _mpb);
    }

    // RG-compatible draw method for OIT path (called from GaussianOITFeature's RenderPass when using an RG-compatible shader variant)
    private void DrawInOIT_RG(RasterCommandBuffer cmd, Camera cam)
    {
        if (!isActiveAndEnabled || !ShouldRenderForCamera(cam) || _runtimeMaterial == null || _numPoints <= 0) return;
        if (_positionBuffer == null || _colorBuffer == null) return;
        if (_mpb == null) _mpb = new MaterialPropertyBlock();
        BuildAndBindMPB(_mpb);

        bool useQuads = UsesTriangleQuadRendering();
        int vertexCount = useQuads ? (_numPoints * 6) : _numPoints;
        MeshTopology topo = useQuads ? MeshTopology.Triangles : MeshTopology.Points;
        cmd.DrawProcedural(Matrix4x4.identity, _runtimeMaterial, 0, topo, vertexCount, 1, _mpb);
    }

    private void DrawDirect_Compat(CommandBuffer cmd, Camera cam)
    {
        if (!isActiveAndEnabled || !ShouldRenderForCamera(cam) || preferOIT || _runtimeMaterial == null || _numPoints <= 0) return;
        if (_positionBuffer == null || _colorBuffer == null) return;
        if (_mpb == null) _mpb = new MaterialPropertyBlock();
        BuildAndBindMPB(_mpb);

        bool useQuads = UsesTriangleQuadRendering();
        int vertexCount = useQuads ? (_numPoints * 6) : _numPoints;
        MeshTopology topo = useQuads ? MeshTopology.Triangles : MeshTopology.Points;
        cmd.DrawProcedural(Matrix4x4.identity, _runtimeMaterial, 0, topo, vertexCount, 1, _mpb);
    }

    private void DrawDirect_RG(RasterCommandBuffer cmd, Camera cam)
    {
        if (!isActiveAndEnabled || !ShouldRenderForCamera(cam) || preferOIT || _runtimeMaterial == null || _numPoints <= 0) return;
        if (_positionBuffer == null || _colorBuffer == null) return;
        if (_mpb == null) _mpb = new MaterialPropertyBlock();
        BuildAndBindMPB(_mpb);

        bool useQuads = UsesTriangleQuadRendering();
        int vertexCount = useQuads ? (_numPoints * 6) : _numPoints;
        MeshTopology topo = useQuads ? MeshTopology.Triangles : MeshTopology.Points;
        cmd.DrawProcedural(Matrix4x4.identity, _runtimeMaterial, 0, topo, vertexCount, 1, _mpb);
    }

    private void RenderForCamera(ScriptableRenderContext context, Camera cam)
    {
        if (_runtimeMaterial == null || _numPoints <= 0 || _positionBuffer == null || _colorBuffer == null) return;

        bool isSRP = (GraphicsSettings.currentRenderPipeline != null);
        bool useQuads = UsesTriangleQuadRendering();
        int vertexCount = useQuads ? (_numPoints * 6) : _numPoints;
        MeshTopology topo = useQuads ? MeshTopology.Triangles : MeshTopology.Points;

        if (_mpb == null) _mpb = new MaterialPropertyBlock();
        BuildAndBindMPB(_mpb);

        if (isSRP)
        {
            var cmd = CommandBufferPool.Get("GaussianSplatDraw");
            cmd.DrawProcedural(Matrix4x4.identity, _runtimeMaterial, 0, topo, vertexCount, 1, _mpb);
            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }
        else
        {
            _runtimeMaterial.SetMatrix("_LocalToWorld", transform.localToWorldMatrix);

            if (overrideBasicShaderParams)
            {
                float finalOpacity = opacity * Mathf.Max(0.0f, opacityMultiplier);
                _runtimeMaterial.SetFloat("_Opacity", finalOpacity);
                _runtimeMaterial.SetFloat("_SigmaCutoff", sigmaCutoff);

                float finalPointSize = pointSize * Mathf.Max(0.0f, pointSizeMultiplier);
                if (!renderAsSplatQuads)
                {
                    _runtimeMaterial.SetFloat("_PointSize", Mathf.Max(0.0f, finalPointSize));
                }
                else
                {
                    if (finalPointSize <= 1e-4f)
                    {
                        _runtimeMaterial.SetFloat("_PointSize", 0f);
                        _runtimeMaterial.SetFloat("_MinAxisPixels", 0f);
                        _runtimeMaterial.SetFloat("_MaxAxisPixels", 0f);
                    }
                    else
                    {
                        _runtimeMaterial.SetFloat("_PointSize", 1.0f);
                        _runtimeMaterial.SetFloat("_MinAxisPixels", minAxisPixels * finalPointSize);
                        _runtimeMaterial.SetFloat("_MaxAxisPixels", maxAxisPixels * finalPointSize);
                    }
                }
            }

            if (overrideOITShaderParams)
            {
                _runtimeMaterial.SetFloat("_WeightExponent", oitWeightExponent);
                _runtimeMaterial.SetFloat("_DepthWeight", depthWeight);
                _runtimeMaterial.SetFloat("_DepthExponent", depthExponent);
                _runtimeMaterial.SetFloat("_NearCompRefZ", nearCompRefZ);
                _runtimeMaterial.SetFloat("_NearCompStrength", nearCompStrength);
                _runtimeMaterial.SetFloat("_NearCompMin", nearCompMin);
                _runtimeMaterial.SetFloat("_NearCompMax", nearCompMax);
            }

            _runtimeMaterial.SetFloat("_ViewZFadeEnabled", enableViewZFade ? 1.0f : 0.0f);
            _runtimeMaterial.SetFloat("_ViewZFadeStart", viewZFadeStart);
            _runtimeMaterial.SetFloat("_ViewZFadeEnd", viewZFadeEnd);
            _runtimeMaterial.SetFloat("_ViewZFadeExponent", viewZFadeExponent);
            _runtimeMaterial.SetFloat("_ViewZFadeInvert", invertViewZFade ? 1.0f : 0.0f);

            _runtimeMaterial.SetBuffer("_Positions", _positionBuffer);
            _runtimeMaterial.SetBuffer("_Colors", _colorBuffer);

            if (MaterialNeedsCovBuffers())
            {
                EnsureCovBuffersAllocated();
                _runtimeMaterial.SetBuffer("_Cov0", _cov0Buffer);
                _runtimeMaterial.SetBuffer("_Cov1", _cov1Buffer);
            }

            _runtimeMaterial.SetPass(0);
            Graphics.DrawProceduralNow(topo, vertexCount);
        }
    }

    public void ReloadData()
    {
        UnloadData();
        LoadData();
    }

    public void UnloadData()
    {
        _positionBuffer?.Release();
        _colorBuffer?.Release();
        _cov0Buffer?.Release();
        _cov1Buffer?.Release();

        _positionBuffer = null;
        _colorBuffer = null;
        _cov0Buffer = null;
        _cov1Buffer = null;

        if (_runtimeMaterial != null)
        {
            _runtimeMaterial.SetBuffer("_Positions", (ComputeBuffer)null);
            _runtimeMaterial.SetBuffer("_Colors", (ComputeBuffer)null);
            _runtimeMaterial.SetBuffer("_Cov0", (ComputeBuffer)null);
            _runtimeMaterial.SetBuffer("_Cov1", (ComputeBuffer)null);
        }

        _numPoints = 0;
        _bufferCapacity = 0;
        _mpb?.Clear();
    }

    private void LoadData()
    {
        if (string.IsNullOrEmpty(dataFileName))
        {
            Debug.LogError("[GaussianLoader] dataFileName is empty.");
            return;
        }

        string path = FullPathForStreamingAsset(dataFileName);
        if (!File.Exists(path))
        {
            Debug.LogError("[GaussianLoader] File not found: " + path);
            return;
        }

        CachedData cached = null;
        if (enableDataCache)
        {
            var writeUtc = File.GetLastWriteTimeUtc(path);
            if (s_DataCache.TryGetValue(path, out cached))
            {
                if (cached == null || cached.lastWriteUtc != writeUtc)
                {
                    s_DataCache.Remove(path);
                    cached = null;
                }
            }

            if (cached == null)
            {
                cached = new CachedData { lastWriteUtc = writeUtc };
                s_DataCache[path] = cached;
            }
        }

        Vector3[] positionsArr;
        Vector3[] colorsArr;
        Vector4[] cov0Arr;
        Vector4[] cov1Arr;
        int count;

        if (enableDataCache && cached != null && cached.positions != null)
        {
            positionsArr = cached.positions;
            colorsArr = cached.colors;
            cov0Arr = cached.cov0;
            cov1Arr = cached.cov1;
            count = cached.count;
        }
        else
        {
            var posList = new List<Vector3>(1024);
            var colList = new List<Vector3>(1024);
            var c0List  = new List<Vector4>(1024);
            var c1List  = new List<Vector4>(1024);

            foreach (var line in File.ReadLines(path))
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                var t = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                if (t.Length < 6)
                    continue;

                float x = float.Parse(t[0], CultureInfo.InvariantCulture);
                float y = float.Parse(t[1], CultureInfo.InvariantCulture);
                float z = float.Parse(t[2], CultureInfo.InvariantCulture);
                float r = float.Parse(t[3], CultureInfo.InvariantCulture);
                float g = float.Parse(t[4], CultureInfo.InvariantCulture);
                float b = float.Parse(t[5], CultureInfo.InvariantCulture);

                // If treating input as world space, convert to local space by subtracting chunk center. Otherwise treat input as already local.
                Vector3 pWorld = new Vector3(x, y, z);
                Vector3 pLocal = treatInputAsWorldSpace ? (pWorld - chunkCenterWorld) : pWorld;

                posList.Add(pLocal);
                colList.Add(new Vector3(r, g, b));

                if (t.Length >= 12)
                {
                    float xx = float.Parse(t[6], CultureInfo.InvariantCulture);
                    float xy = float.Parse(t[7], CultureInfo.InvariantCulture);
                    float xz = float.Parse(t[8], CultureInfo.InvariantCulture);
                    float yy = float.Parse(t[9], CultureInfo.InvariantCulture);
                    float yz = float.Parse(t[10], CultureInfo.InvariantCulture);
                    float zz = float.Parse(t[11], CultureInfo.InvariantCulture);

                    c0List.Add(new Vector4(xx, xy, xz, yy));
                    c1List.Add(new Vector4(yz, zz, 0f, 0f));
                }
                else
                {
                    c0List.Add(new Vector4(FallbackCov, 0f, 0f, FallbackCov));
                    c1List.Add(new Vector4(0f, FallbackCov, 0f, 0f));
                }
            }

            count = posList.Count;
            if (count == 0)
            {
                Debug.LogError($"[GaussianLoader] No valid points loaded from {dataFileName}");
                return;
            }

            positionsArr = posList.ToArray();
            colorsArr = colList.ToArray();
            cov0Arr = c0List.ToArray();
            cov1Arr = c1List.ToArray();

            if (enableDataCache && cached != null)
            {
                cached.positions = positionsArr;
                cached.colors = colorsArr;
                cached.cov0 = cov0Arr;
                cached.cov1 = cov1Arr;
                cached.count = count;
            }
        }

        _numPoints = count;
        int desiredCapacity = growBuffersByPowerOfTwo ? NextPow2(_numPoints) : _numPoints;

        bool needAlloc = (_bufferCapacity != desiredCapacity)
                         || _positionBuffer == null || _colorBuffer == null
                         || _cov0Buffer == null || _cov1Buffer == null;

        if (needAlloc)
        {
            _positionBuffer?.Release();
            _colorBuffer?.Release();
            _cov0Buffer?.Release();
            _cov1Buffer?.Release();

            _bufferCapacity = desiredCapacity;
            _positionBuffer = new ComputeBuffer(_bufferCapacity, Float3Stride, ComputeBufferType.Structured);
            _colorBuffer    = new ComputeBuffer(_bufferCapacity, Float3Stride, ComputeBufferType.Structured);
            _cov0Buffer     = new ComputeBuffer(_bufferCapacity, Float4Stride, ComputeBufferType.Structured);
            _cov1Buffer     = new ComputeBuffer(_bufferCapacity, Float4Stride, ComputeBufferType.Structured);
        }

        _positionBuffer.SetData(positionsArr, 0, 0, _numPoints);
        _colorBuffer.SetData(colorsArr, 0, 0, _numPoints);
        _cov0Buffer.SetData(cov0Arr, 0, 0, _numPoints);
        _cov1Buffer.SetData(cov1Arr, 0, 0, _numPoints);

        if (verboseDebug)
            Debug.Log($"[GaussianLoader] Loaded {_numPoints} points. (capacity={_bufferCapacity})");
    }

    private void OnDestroy()
    {
        OnDisable();
        UnloadData();

        if (_runtimeMaterial != null && !ReferenceEquals(_runtimeMaterial, _sourceMaterialAsset))
            Destroy(_runtimeMaterial);

        _runtimeMaterial = null;
    }
}
