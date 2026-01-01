using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

public class GaussianLoader : MonoBehaviour
{
    [Header("Data file (relative to StreamingAssets)")]
    public string dataFileName = "SampleBlock1_gaussians_demo.txt";

    [Header("Rendering")]
    public Material pointMaterial;
    public float pointSize = 1.0f;

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

    [Header("Chunk options")]
    public bool treatInputAsWorldSpace = false;
    public Vector3 chunkCenterWorld = Vector3.zero;

    private ComputeBuffer _positionBuffer;
    private ComputeBuffer _colorBuffer;

    // Covariance packed for shader:
    // _Cov0 = (xx, xy, xz, yy)
    // _Cov1 = (yz, zz, 0, 0)
    private ComputeBuffer _cov0Buffer;
    private ComputeBuffer _cov1Buffer;

    private int _numPoints;

    private Material _runtimeMaterial;

    private bool _materialReady = false;

    // Throttled debug to avoid log spam
    private int _lastMissingLogFrame = -9999;
    private string _lastMissingLogKey = "";

    private bool _srpHooked = false;

    private MaterialPropertyBlock _mpb;

    [Header("Debug")]
    public bool verboseDebug = false;

    void Start()
    {
        if (pointMaterial == null)
        {
            Debug.LogError("[GaussianLoader] pointMaterial is not assigned!");
            return;
        }

        _runtimeMaterial = new Material(pointMaterial);
        _mpb = new MaterialPropertyBlock();

        if (treatInputAsWorldSpace)
        {
            transform.position = chunkCenterWorld;
        }

        LoadData();
        SetupMaterial();

        if (verboseDebug)
        {
            Debug.Log($"[GaussianLoader] Start done for {dataFileName}. materialReady={_materialReady}");
        }
    }

    private void OnEnable()
    {
        // In SRP, OnRenderObject() may not be invoked reliably. Use endCameraRendering instead.
        if (!_srpHooked)
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
    }

    private void OnEndCameraRendering(ScriptableRenderContext context, Camera cam)
    {
        // Draw after the pipeline finished rendering the camera (after clears), so results are visible.
        if (cam == null) return;
        if (!isActiveAndEnabled) return;

        // Render for Game cameras only. If you want SceneView too, comment this out.
        if (cam.cameraType != CameraType.Game) return;

        RenderForCamera(context, cam);

        // In some SRP/Metal cases, executing at end needs an explicit submit.
        context.Submit();
    }

    public void ReloadData()
    {
        UnloadData();
        LoadData();
        SetupMaterial();
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
        _numPoints = 0;
        _materialReady = false;
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
            if (t.Length < 12)
                continue;

            float x = float.Parse(t[0], CultureInfo.InvariantCulture);
            float y = float.Parse(t[1], CultureInfo.InvariantCulture);
            float z = float.Parse(t[2], CultureInfo.InvariantCulture);

            // New format: x y z r g b xx xy xz yy yz zz
            float r = float.Parse(t[3], CultureInfo.InvariantCulture);
            float g = float.Parse(t[4], CultureInfo.InvariantCulture);
            float b = float.Parse(t[5], CultureInfo.InvariantCulture);

            float xx = float.Parse(t[6], CultureInfo.InvariantCulture);
            float xy = float.Parse(t[7], CultureInfo.InvariantCulture);
            float xz = float.Parse(t[8], CultureInfo.InvariantCulture);
            float yy = float.Parse(t[9], CultureInfo.InvariantCulture);
            float yz = float.Parse(t[10], CultureInfo.InvariantCulture);
            float zz = float.Parse(t[11], CultureInfo.InvariantCulture);

            Vector3 pWorld = new Vector3(x, y, z);
            Vector3 pLocal = treatInputAsWorldSpace ? (pWorld - chunkCenterWorld) : pWorld;

            positions.Add(pLocal);
            colors.Add(new Vector3(r, g, b));

            cov0List.Add(new Vector4(xx, xy, xz, yy));
            cov1List.Add(new Vector4(yz, zz, 0f, 0f));
        }

        _numPoints = positions.Count;
        if (_numPoints == 0)
        {
            Debug.LogError($"[GaussianLoader] No valid points loaded from {dataFileName}");
            return;
        }

        _positionBuffer = new ComputeBuffer(_numPoints, sizeof(float) * 3);
        _positionBuffer.SetData(positions);

        _colorBuffer = new ComputeBuffer(_numPoints, sizeof(float) * 3);
        _colorBuffer.SetData(colors);

        _cov0Buffer = new ComputeBuffer(_numPoints, sizeof(float) * 4);
        _cov0Buffer.SetData(cov0List);

        _cov1Buffer = new ComputeBuffer(_numPoints, sizeof(float) * 4);
        _cov1Buffer.SetData(cov1List);

        Debug.Log($"[GaussianLoader] Loaded {_numPoints} points.");
        if (verboseDebug)
        {
            Debug.Log($"[GaussianLoader] Buffers created for {dataFileName}: pos={_positionBuffer != null} col={_colorBuffer != null} cov0={_cov0Buffer != null} cov1={_cov1Buffer != null}");
        }
    }

    private void SetupMaterial()
    {
        _materialReady = false;
        if (_runtimeMaterial == null)
        {
            Debug.LogError("[GaussianLoader] _runtimeMaterial is null in SetupMaterial");
            return;
        }

        if (_positionBuffer == null || _colorBuffer == null || _cov0Buffer == null || _cov1Buffer == null)
        {
            Debug.LogError($"[GaussianLoader] Missing ComputeBuffer(s) for {dataFileName}. " +
                           $"pos={(_positionBuffer != null)} col={(_colorBuffer != null)} cov0={(_cov0Buffer != null)} cov1={(_cov1Buffer != null)}");
            return;
        }

        _runtimeMaterial.SetFloat("_Opacity", opacity);
        _runtimeMaterial.SetFloat("_SigmaCutoff", sigmaCutoff);
        _runtimeMaterial.SetFloat("_MinAxisPixels", minAxisPixels);
        _runtimeMaterial.SetFloat("_MaxAxisPixels", maxAxisPixels);

        _materialReady = true;

        if (verboseDebug)
        {
            Debug.Log($"[GaussianLoader] SetupMaterial OK for {dataFileName} (num={_numPoints})");
        }
    }

    private void OnRenderObject()
    {
        // Built-in pipeline fallback. In SRP (URP/HDRP), this may not be called.
        if (GraphicsSettings.currentRenderPipeline != null)
            return;

        var cam = Camera.current;
        if (cam == null) return;
        RenderForCamera(default, cam);
    }

    private void RenderForCamera(ScriptableRenderContext context, Camera cam)
    {
        if (verboseDebug && Time.frameCount % 120 == 0)
            Debug.Log($"[GaussianLoader] RenderForCamera: {dataFileName}, num={_numPoints}, cam={cam.name}, rp={(GraphicsSettings.currentRenderPipeline != null ? "SRP" : "BuiltIn")}");

        // If running in built-in pipeline, the SRP context will be default and ExecuteCommandBuffer is invalid.
        // In that case, fall back to immediate mode.
        bool isSRP = (GraphicsSettings.currentRenderPipeline != null);

        if (_runtimeMaterial == null)
            return;

        if (_numPoints <= 0)
            return;

        // Validate required buffers for the chosen mode
        bool hasPos = _positionBuffer != null;
        bool hasCol = _colorBuffer != null;
        bool hasCov0 = _cov0Buffer != null;
        bool hasCov1 = _cov1Buffer != null;

        string missingKey = "";
        if (!hasPos) missingKey += "pos;";
        if (!hasCol) missingKey += "col;";
        if (renderAsSplatQuads && !hasCov0) missingKey += "cov0;";
        if (renderAsSplatQuads && !hasCov1) missingKey += "cov1;";

        if (!string.IsNullOrEmpty(missingKey))
        {
            // Throttle logs: once per 30 frames per missing pattern
            if (Time.frameCount - _lastMissingLogFrame > 30 || _lastMissingLogKey != missingKey)
            {
                _lastMissingLogFrame = Time.frameCount;
                _lastMissingLogKey = missingKey;
                Debug.LogWarning($"[GaussianLoader] Skip draw (missing buffers: {missingKey}) file={dataFileName} num={_numPoints} mode={(renderAsSplatQuads ? "SplatQuads" : "Points")}");
            }
            return;
        }

        // Ensure buffers are bound (important after Reload/Unload/LOD switches)
        if (!_materialReady)
        {
            SetupMaterial();
            if (!_materialReady)
                return;
        }

        int vertexCount = renderAsSplatQuads ? (_numPoints * 6) : _numPoints;
        MeshTopology topology = renderAsSplatQuads ? MeshTopology.Triangles : MeshTopology.Points;

        // Bind ALL per-draw resources via MaterialPropertyBlock (SRP/Metal-safe).
        if (_mpb == null) _mpb = new MaterialPropertyBlock();
        _mpb.Clear();

        _mpb.SetMatrix("_LocalToWorld", transform.localToWorldMatrix);
        _mpb.SetFloat("_Opacity", opacity);
        _mpb.SetFloat("_SigmaCutoff", sigmaCutoff);
        _mpb.SetFloat("_MinAxisPixels", minAxisPixels);
        _mpb.SetFloat("_MaxAxisPixels", maxAxisPixels);

        _mpb.SetBuffer("_Positions", _positionBuffer);
        _mpb.SetBuffer("_Colors", _colorBuffer);
        if (renderAsSplatQuads)
        {
            _mpb.SetBuffer("_Cov0", _cov0Buffer);
            _mpb.SetBuffer("_Cov1", _cov1Buffer);
        }

        if (isSRP)
        {
            var cmd = CommandBufferPool.Get("GaussianSplatDraw");
            cmd.DrawProcedural(Matrix4x4.identity, _runtimeMaterial, 0, topology, vertexCount, 1, _mpb);
            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }
        else
        {
            _runtimeMaterial.SetPass(0);
            // Built-in immediate mode doesn't accept MPB; buffers are already validated, so just draw.
            Graphics.DrawProceduralNow(topology, vertexCount);
        }
    }

    private void OnDestroy()
    {
        OnDisable();

        UnloadData();

        if (_runtimeMaterial != null)
        {
            Destroy(_runtimeMaterial);
            _runtimeMaterial = null;
        }
    }
}