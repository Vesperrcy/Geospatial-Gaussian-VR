using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class GaussianLoader : MonoBehaviour
{
    [Header("Data file (relative to StreamingAssets)")]
    public string dataFileName = "SampleBlock1_gaussians_demo.txt";

    [Header("Rendering")]
    public Material pointMaterial;

    [Tooltip("Use the shared material asset directly so changing its Inspector sliders updates rendering in real-time. If false, a runtime clone is created (Inspector edits to the asset will NOT affect the clone during Play Mode).")]
    public bool useSharedMaterialAsset = true;

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

    private ComputeBuffer _positionBuffer;
    private ComputeBuffer _colorBuffer;

    // Covariance packed for shader:
    // _Cov0 = (xx, xy, xz, yy)
    // _Cov1 = (yz, zz, 0, 0)
    private ComputeBuffer _cov0Buffer;
    private ComputeBuffer _cov1Buffer;

    private int _numPoints;
    private Material _runtimeMaterial;
    private MaterialPropertyBlock _mpb;

    private bool _srpHooked = false;
    private bool _oitHooked = false;

    private const int Float3Stride = sizeof(float) * 3;
    private const int Float4Stride = sizeof(float) * 4;

    // A tiny isotropic covariance (meters^2). Used when file has no covariance.
    private const float FallbackCov = 0.0001f;

    void Start()
    {
        if (pointMaterial == null)
        {
            Debug.LogError("[GaussianLoader] pointMaterial is not assigned!");
            return;
        }

        _runtimeMaterial = useSharedMaterialAsset ? pointMaterial : new Material(pointMaterial);
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

    private void BuildAndBindMPB(MaterialPropertyBlock mpb)
    {
        if (mpb == null) return;

        mpb.Clear();
        mpb.SetMatrix("_LocalToWorld", transform.localToWorldMatrix);

        // If you want Material Inspector to drive parameters, turn these OFF.
        if (overrideBasicShaderParams)
        {
            mpb.SetFloat("_Opacity", opacity);
            mpb.SetFloat("_SigmaCutoff", sigmaCutoff);
            mpb.SetFloat("_MinAxisPixels", minAxisPixels);
            mpb.SetFloat("_MaxAxisPixels", maxAxisPixels);
            mpb.SetFloat("_PointSize", pointSize);
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

        if (renderAsSplatQuads)
        {
            if (_cov0Buffer == null || _cov1Buffer == null || _cov0Buffer.count != _numPoints || _cov1Buffer.count != _numPoints)
                EnsureCovBuffersExist(_numPoints);

            mpb.SetBuffer("_Cov0", _cov0Buffer);
            mpb.SetBuffer("_Cov1", _cov1Buffer);
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
            // Built-in: best-effort
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
    }

    private void LoadData()
    {
        Debug.Log($"[GaussianLoader] LoadData: {dataFileName}");

        if (string.IsNullOrEmpty(dataFileName))
        {
            Debug.LogError("[GaussianLoader] dataFileName is empty.");
            return;
        }

        string path = Path.Combine(Application.streamingAssetsPath, dataFileName);
        if (!File.Exists(path))
        {
            Debug.LogError("[GaussianLoader] File not found: " + path);
            return;
        }

        List<Vector3> positions = new();
        List<Vector3> colors = new();
        List<Vector4> cov0List = new();
        List<Vector4> cov1List = new();

        var lines = File.ReadAllLines(path);

        foreach (var line in lines)
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

            positions.Add(pLocal);
            colors.Add(new Vector3(r, g, b));

            if (t.Length >= 12)
            {
                float xx = float.Parse(t[6], CultureInfo.InvariantCulture);
                float xy = float.Parse(t[7], CultureInfo.InvariantCulture);
                float xz = float.Parse(t[8], CultureInfo.InvariantCulture);
                float yy = float.Parse(t[9], CultureInfo.InvariantCulture);
                float yz = float.Parse(t[10], CultureInfo.InvariantCulture);
                float zz = float.Parse(t[11], CultureInfo.InvariantCulture);

                cov0List.Add(new Vector4(xx, xy, xz, yy));
                cov1List.Add(new Vector4(yz, zz, 0f, 0f));
            }
            else
            {
                cov0List.Add(new Vector4(FallbackCov, 0f, 0f, FallbackCov));
                cov1List.Add(new Vector4(0f, FallbackCov, 0f, 0f));
            }
        }

        _numPoints = positions.Count;
        if (_numPoints == 0)
        {
            Debug.LogError($"[GaussianLoader] No valid points loaded from {dataFileName}");
            return;
        }

        _positionBuffer = new ComputeBuffer(_numPoints, Float3Stride, ComputeBufferType.Structured);
        _positionBuffer.SetData(positions);

        _colorBuffer = new ComputeBuffer(_numPoints, Float3Stride, ComputeBufferType.Structured);
        _colorBuffer.SetData(colors);

        _cov0Buffer = new ComputeBuffer(_numPoints, Float4Stride, ComputeBufferType.Structured);
        _cov0Buffer.SetData(cov0List);

        _cov1Buffer = new ComputeBuffer(_numPoints, Float4Stride, ComputeBufferType.Structured);
        _cov1Buffer.SetData(cov1List);

        Debug.Log($"[GaussianLoader] Loaded {_numPoints} points.");
        if (verboseDebug)
        {
            Debug.Log($"[GaussianLoader] Buffers: pos={(_positionBuffer!=null)} col={(_colorBuffer!=null)} cov0={(_cov0Buffer!=null)} cov1={(_cov1Buffer!=null)}");
        }
    }

    private void OnDestroy()
    {
        OnDisable();
        UnloadData();

        if (!useSharedMaterialAsset && _runtimeMaterial != null)
        {
            Destroy(_runtimeMaterial);
        }
        _runtimeMaterial = null;
    }
}