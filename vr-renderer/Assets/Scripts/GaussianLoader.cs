using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class GaussianLoader : MonoBehaviour
{
    [Header("Data file (relative to StreamingAssets)")]
    public string dataFileName = "TumTLS_v1_gaussians_demo.txt";

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
    public bool growBuffersByPowerOfTwo = true;

    private ComputeBuffer _positionBuffer;
    private ComputeBuffer _colorBuffer;

    // Covariance packed for shader:
    // _Cov0 = (xx, xy, xz, yy)
    // _Cov1 = (yz, zz, 0, 0)
    private ComputeBuffer _cov0Buffer;
    private ComputeBuffer _cov1Buffer;

    private int _numPoints;
    private int _bufferCapacity = 0; // allocated element capacity for current GPU buffers
    private Material _runtimeMaterial;
    private MaterialPropertyBlock _mpb;
    private Material _sourceMaterialAsset; // original asset reference (for live param mirroring)
    private bool _mirrorAssetParams = false;

    private bool _srpHooked = false;
    private bool _oitHooked = false;

    private const int Float3Stride = sizeof(float) * 3;
    private const int Float4Stride = sizeof(float) * 4;

    // A tiny isotropic covariance (meters^2). Used when file has no covariance.
    private const float FallbackCov = 0.0001f;

    // -------------------------
    // Data cache (reduces LOD switch hitches)
    // -------------------------
    private sealed class CachedData
    {
        public Vector3[] positions;
        public Vector3[] colors;
        public Vector4[] cov0;
        public Vector4[] cov1;
        public int count;
        public DateTime lastWriteUtc;
    }

    private static readonly Dictionary<string, CachedData> s_DataCache = new();

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

    private static string FullPathForStreamingAsset(string fileName)
        => Path.Combine(Application.streamingAssetsPath, fileName);

    void Start()
    {
        if (pointMaterial == null)
        {
            Debug.LogError("[GaussianLoader] pointMaterial is not assigned!");
            return;
        }

        _sourceMaterialAsset = pointMaterial;

        // Always prefer a per-chunk runtime material when the shader relies on per-draw ComputeBuffers.
        // On Metal, missing or cross-bound buffers can cause draw calls to be skipped.
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

        _mpb = new MaterialPropertyBlock();

        if (treatInputAsWorldSpace)
            transform.position = chunkCenterWorld;

        LoadData();

        if (verboseDebug)
            Debug.Log($"[GaussianLoader] Start done for {dataFileName}. num={_numPoints}");
    }

    private void OnValidate()
    {
        if (pointMaterial == null) return;
        _sourceMaterialAsset = pointMaterial;

        if (useSharedMaterialAsset)
        {
            _runtimeMaterial = pointMaterial;
        }
        else
        {
            if (!Application.isPlaying)
                _runtimeMaterial = null;
        }
    }

    private void LateUpdate()
    {
        // If we created a runtime clone while still wanting live Inspector control,
        // mirror all numeric/texture properties from the asset each frame.
        // This does NOT copy buffers; buffers are bound per-chunk.
        if (!_mirrorAssetParams) return;
        if (_sourceMaterialAsset == null || _runtimeMaterial == null) return;
        if (ReferenceEquals(_runtimeMaterial, _sourceMaterialAsset)) return;

        _runtimeMaterial.CopyPropertiesFromMaterial(_sourceMaterialAsset);
    }

    private void OnEnable()
    {
        // Prefer rendering through the URP RendererFeature (OIT) when available.
        if (preferOIT)
        {
            try
            {
                GaussianOITFeature.OnDrawGaussians += DrawInOIT_Compat;
                GaussianOITFeature.OnDrawGaussiansRG += DrawInOIT_RG;
                _oitHooked = true;

                if (verboseDebug)
                    Debug.Log($"[GaussianLoader] OIT hooks enabled for {dataFileName}");
            }
            catch
            {
                _oitHooked = false;
            }
        }

        // Fallback: endCameraRendering in SRP
        if (!_oitHooked && !_srpHooked)
        {
            RenderPipelineManager.endCameraRendering += OnEndCameraRendering;
            _srpHooked = true;
        }
    }

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

        // If OIT is enabled for this loader, OITFeature is responsible for drawing.
        if (_oitHooked) return;

        RenderForCamera(context, cam);
    }

    // Built-in pipeline fallback (usually not used in URP)
    private void OnRenderObject()
    {
        if (GraphicsSettings.currentRenderPipeline != null)
            return;

        if (_oitHooked) return;

        var cam = Camera.current;
        if (!ShouldRenderForCamera(cam)) return;

        RenderForCamera(default, cam);
    }

    // --- Core rule: If we draw splat quads, we MUST bind cov buffers every draw (Metal safe) ---
    private void EnsureCovBuffersExist(int count)
    {
        if (count <= 0) return;

        if (_cov0Buffer == null || _cov0Buffer.count != count)
        {
            _cov0Buffer?.Release();
            _cov0Buffer = new ComputeBuffer(count, Float4Stride, ComputeBufferType.Structured);
        }
        if (_cov1Buffer == null || _cov1Buffer.count != count)
        {
            _cov1Buffer?.Release();
            _cov1Buffer = new ComputeBuffer(count, Float4Stride, ComputeBufferType.Structured);
        }

        // Zero is OK for “dummy cov”. But better: small isotropic so you still see something.
        // We’ll fill only if buffer is newly created and no data has been set.
        // (Safe to fill always; cost is minor compared to missing-buffer crash avoidance.)
        var tmp = new Vector4[count];
        for (int i = 0; i < count; i++)
        {
            tmp[i] = new Vector4(FallbackCov, 0f, 0f, FallbackCov); // xx xy xz yy
        }
        _cov0Buffer.SetData(tmp);

        for (int i = 0; i < count; i++)
        {
            tmp[i] = new Vector4(0f, FallbackCov, 0f, 0f); // yz zz 0 0 (approx isotropic in z)
        }
        _cov1Buffer.SetData(tmp);
    }

    // Some materials (e.g., Unlit/GaussianSplatQuads) always declare _Cov0/_Cov1 buffers.
    // On Metal, failing to bind a declared ComputeBuffer can skip the entire draw call.
    private bool MaterialNeedsCovBuffers()
    {
        if (_runtimeMaterial == null) return false;
        // If we are drawing splat quads, we definitely need cov buffers.
        if (renderAsSplatQuads) return true;
        // Even in point mode, the active shader might still declare these buffers.
        return _runtimeMaterial.HasProperty("_Cov0") || _runtimeMaterial.HasProperty("_Cov1");
    }

    private void BuildAndBindMPB(MaterialPropertyBlock mpb)
    {
        if (mpb == null) return;

        mpb.Clear();
        mpb.SetMatrix("_LocalToWorld", transform.localToWorldMatrix);

       
        if (overrideBasicShaderParams)
        {
            mpb.SetFloat("_Opacity", opacity);
            mpb.SetFloat("_SigmaCutoff", sigmaCutoff);

            // IMPORTANT:
            // In the shader, point size is applied BEFORE the min/max pixel clamp.
            // If many splats are already clamped to the same min/max, changing _PointSize may look like it does nothing.
            // To make PointSize always have a visible effect, we scale the clamp together with PointSize and keep shader _PointSize = 1.
            if (pointSize <= 1e-4f)
            {
                // Treat "cleared" / zero as force-minimal splats (almost points)
                mpb.SetFloat("_PointSize", 0f);
                mpb.SetFloat("_MinAxisPixels", 0f);
                mpb.SetFloat("_MaxAxisPixels", 0f);
            }
            else
            {
                mpb.SetFloat("_PointSize", 1.0f);
                mpb.SetFloat("_MinAxisPixels", minAxisPixels * pointSize);
                mpb.SetFloat("_MaxAxisPixels", maxAxisPixels * pointSize);
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

        // Buffers always bound per draw
        mpb.SetBuffer("_Positions", _positionBuffer);
        mpb.SetBuffer("_Colors", _colorBuffer);
        if (!ReferenceEquals(_runtimeMaterial, _sourceMaterialAsset))
        {
            _runtimeMaterial.SetBuffer("_Positions", _positionBuffer);
            _runtimeMaterial.SetBuffer("_Colors", _colorBuffer);
        }

        if (MaterialNeedsCovBuffers())
        {
            if (_cov0Buffer == null || _cov1Buffer == null || _cov0Buffer.count != _numPoints || _cov1Buffer.count != _numPoints)
                EnsureCovBuffersExist(_numPoints);

            mpb.SetBuffer("_Cov0", _cov0Buffer);
            mpb.SetBuffer("_Cov1", _cov1Buffer);

            // Fallback for Metal / some SRP paths: also bind on the material instance.
            // Only do this if we are NOT using the shared asset material.
            if (!ReferenceEquals(_runtimeMaterial, _sourceMaterialAsset))
            {
                _runtimeMaterial.SetBuffer("_Positions", _positionBuffer);
                _runtimeMaterial.SetBuffer("_Colors", _colorBuffer);
                _runtimeMaterial.SetBuffer("_Cov0", _cov0Buffer);
                _runtimeMaterial.SetBuffer("_Cov1", _cov1Buffer);
            }
        }
    }

    // === OIT rendering entrypoints (called by GaussianOITFeature) ===
    private void DrawInOIT_Compat(CommandBuffer cmd, Camera cam)
    {
        if (!isActiveAndEnabled) return;
        if (!ShouldRenderForCamera(cam)) return;
        if (_runtimeMaterial == null) return;
        if (_numPoints <= 0) return;

        // Must have these to draw anything
        if (_positionBuffer == null || _colorBuffer == null) return;

        if (_mpb == null) _mpb = new MaterialPropertyBlock();
        BuildAndBindMPB(_mpb);

        int vertexCount = renderAsSplatQuads ? (_numPoints * 6) : _numPoints;
        MeshTopology topo = renderAsSplatQuads ? MeshTopology.Triangles : MeshTopology.Points;

        cmd.DrawProcedural(Matrix4x4.identity, _runtimeMaterial, 0, topo, vertexCount, 1, _mpb);
    }

    private void DrawInOIT_RG(RasterCommandBuffer cmd, Camera cam)
    {
        if (!isActiveAndEnabled) return;
        if (!ShouldRenderForCamera(cam)) return;
        if (_runtimeMaterial == null) return;
        if (_numPoints <= 0) return;

        if (_positionBuffer == null || _colorBuffer == null) return;

        if (_mpb == null) _mpb = new MaterialPropertyBlock();
        BuildAndBindMPB(_mpb);

        int vertexCount = renderAsSplatQuads ? (_numPoints * 6) : _numPoints;
        MeshTopology topo = renderAsSplatQuads ? MeshTopology.Triangles : MeshTopology.Points;

        cmd.DrawProcedural(Matrix4x4.identity, _runtimeMaterial, 0, topo, vertexCount, 1, _mpb);
    }

    private void RenderForCamera(ScriptableRenderContext context, Camera cam)
    {
        if (_runtimeMaterial == null) return;
        if (_numPoints <= 0) return;
        if (_positionBuffer == null || _colorBuffer == null) return;

        // In SRP, draw with a command buffer.
        bool isSRP = (GraphicsSettings.currentRenderPipeline != null);

        int vertexCount = renderAsSplatQuads ? (_numPoints * 6) : _numPoints;
        MeshTopology topo = renderAsSplatQuads ? MeshTopology.Triangles : MeshTopology.Points;

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
            // Built-in: bind buffers directly to the material (MaterialPropertyBlock is not used by DrawProceduralNow).
            _runtimeMaterial.SetMatrix("_LocalToWorld", transform.localToWorldMatrix);

            if (overrideBasicShaderParams)
            {
                _runtimeMaterial.SetFloat("_Opacity", opacity);
                _runtimeMaterial.SetFloat("_SigmaCutoff", sigmaCutoff);

                if (pointSize <= 1e-4f)
                {
                    _runtimeMaterial.SetFloat("_PointSize", 0f);
                    _runtimeMaterial.SetFloat("_MinAxisPixels", 0f);
                    _runtimeMaterial.SetFloat("_MaxAxisPixels", 0f);
                }
                else
                {
                    _runtimeMaterial.SetFloat("_PointSize", 1.0f);
                    _runtimeMaterial.SetFloat("_MinAxisPixels", minAxisPixels * pointSize);
                    _runtimeMaterial.SetFloat("_MaxAxisPixels", maxAxisPixels * pointSize);
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

            _runtimeMaterial.SetBuffer("_Positions", _positionBuffer);
            _runtimeMaterial.SetBuffer("_Colors", _colorBuffer);

            if (MaterialNeedsCovBuffers())
            {
                if (_cov0Buffer == null || _cov1Buffer == null || _cov0Buffer.count != _numPoints || _cov1Buffer.count != _numPoints)
                    EnsureCovBuffersExist(_numPoints);

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
        // Safety: If buffers are released, the object might still be subscribed.
        // Disabling will stop isActiveAndEnabled from drawing via OIT callbacks.
        // (ChunkManager may handle enabled/disabled as well; this is extra safety.)
        // enabled = false;

        _positionBuffer?.Release();
        _colorBuffer?.Release();
        _cov0Buffer?.Release();
        _cov1Buffer?.Release();

        _positionBuffer = null;
        _colorBuffer = null;
        _cov0Buffer = null;
        _cov1Buffer = null;

        _numPoints = 0;
        _bufferCapacity = 0;
        _mpb?.Clear();
    }

    private void LoadData()
    {
        Debug.Log($"[GaussianLoader] LoadData: {dataFileName}");

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

        // ---- 1) Try cache first (massively reduces hitching when switching LOD back/forth) ----
        CachedData cached = null;
        if (enableDataCache)
        {
            var writeUtc = File.GetLastWriteTimeUtc(path);
            if (s_DataCache.TryGetValue(path, out cached))
            {
                // Invalidate if file changed
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
            // Cache hit
            positionsArr = cached.positions;
            colorsArr = cached.colors;
            cov0Arr = cached.cov0;
            cov1Arr = cached.cov1;
            count = cached.count;
        }
        else
        {
            // ---- 2) Parse file (single pass) ----
            var posList = new List<Vector3>(1024);
            var colList = new List<Vector3>(1024);
            var c0List  = new List<Vector4>(1024);
            var c1List  = new List<Vector4>(1024);

            foreach (var line in File.ReadLines(path))
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                var t = line.Split((char[])null, System.StringSplitOptions.RemoveEmptyEntries);
                if (t.Length < 6)
                    continue;

                float x = float.Parse(t[0], CultureInfo.InvariantCulture);
                float y = float.Parse(t[1], CultureInfo.InvariantCulture);
                float z = float.Parse(t[2], CultureInfo.InvariantCulture);

                float r = float.Parse(t[3], CultureInfo.InvariantCulture);
                float g = float.Parse(t[4], CultureInfo.InvariantCulture);
                float b = float.Parse(t[5], CultureInfo.InvariantCulture);

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
            colorsArr    = colList.ToArray();
            cov0Arr      = c0List.ToArray();
            cov1Arr      = c1List.ToArray();

            if (enableDataCache && cached != null)
            {
                cached.positions = positionsArr;
                cached.colors    = colorsArr;
                cached.cov0      = cov0Arr;
                cached.cov1      = cov1Arr;
                cached.count     = count;
            }
        }

        _numPoints = count;

        // ---- 3) Allocate GPU buffers with extra capacity to reduce realloc hitching ----
        int desiredCapacity = _numPoints;
        if (growBuffersByPowerOfTwo)
            desiredCapacity = NextPow2(desiredCapacity);

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

        // Upload only the active count
        _positionBuffer.SetData(positionsArr, 0, 0, _numPoints);
        _colorBuffer.SetData(colorsArr, 0, 0, _numPoints);
        _cov0Buffer.SetData(cov0Arr, 0, 0, _numPoints);
        _cov1Buffer.SetData(cov1Arr, 0, 0, _numPoints);

        Debug.Log($"[GaussianLoader] Loaded {_numPoints} points. (capacity={_bufferCapacity})");
        if (verboseDebug)
        {
            Debug.Log($"[GaussianLoader] Buffers: pos={(_positionBuffer!=null)} col={(_colorBuffer!=null)} cov0={(_cov0Buffer!=null)} cov1={(_cov1Buffer!=null)}");
        }
    }

    private void OnDestroy()
    {
        OnDisable();
        UnloadData();

        // Destroy runtime clone if we created one.
        if (_runtimeMaterial != null && !ReferenceEquals(_runtimeMaterial, _sourceMaterialAsset))
        {
            Destroy(_runtimeMaterial);
        }
        _runtimeMaterial = null;
    }
}