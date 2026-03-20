using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// Reads chunk and LOD information from StreamingAssets/{chunksFolderName}/{lodIndexFileName},
/// and creates a GameObject with GaussianLoader(s) per chunk.
/// Supports:
/// - Frustum culling
/// - Chunk streaming (load/unload radius with hysteresis)
/// - Per-frame reload budget (avoid VR hitching)
/// - Runtime display-mode switching between Gaussian splats and raw point cloud
/// </summary>
public class GaussianChunkManager : MonoBehaviour
{
    public enum DisplayMode
    {
        GaussianSplat = 0,
        RawPointCloud = 1
    }

    [Header("Chunk Folder (in StreamingAssets)")]
    public string chunksFolderName = "chunks_TUMv2";

    [Header("LOD index file name")]
    public string lodIndexFileName = "chunks_lod_index.json";

    [Header("Rendering")]
    [Tooltip("Base material for the Gaussian splat view.")]
    public Material pointMaterial;

    [Tooltip("Optional material for the raw point-cloud view. If empty, a runtime material is created from Shader \"Unlit/RawPointCloud\".")]
    public Material rawPointMaterial;

    [Header("Display Mode")]
    public DisplayMode currentDisplayMode = DisplayMode.GaussianSplat;
    public bool allowKeyboardToggle = true;
    public bool showModeSwitcherUI = true;

    [Header("Splatting / Rendering Mode")]
    public bool renderAsSplatQuads = true;

    [Header("Raw Point Cloud")]
    public float rawPointSize = 2.0f;

    [Header("Point size (optional / legacy)")]
    public float pointSizeL0 = 1.0f;
    public float pointSizeL1 = 1.0f;

    [Header("Ellipse Splat Params (two-level)")]
    [Range(0f, 1f)] public float opacityL0 = 0.6f;
    [Range(0f, 1f)] public float opacityL1 = 0.18f;

    [Tooltip("k-sigma cutoff used to size ellipse quads (typical 2~4).")]
    public float sigmaCutoffL0 = 3.0f;
    public float sigmaCutoffL1 = 3.0f;

    [Tooltip("Clamp ellipse axis in pixel units (min).")]
    public float minAxisPixels = 0.75f;

    [Tooltip("Clamp ellipse axis in pixel units (max).")]
    public float maxAxisPixels = 64.0f;

    [Header("Scale Mixture (dual-pass, L0 + L1)")]
    [Tooltip("Render each chunk twice: Fine(L0) for detail, Coarse(L1) for gap filling. Blended by distance.")]
    public bool enableScaleMixture = true;

    [Tooltip("Distance where coarse pass starts to fade in (meters).")]
    public float mixtureStart = 8f;

    [Tooltip("Distance where coarse pass reaches full strength (meters).")]
    public float mixtureEnd = 25f;

    [Tooltip("Extra pointSize multiplier for the coarse pass (fills scanline gaps).")]
    public float coarsePointSizeMultiplier = 2.5f;

    [Tooltip("Opacity multiplier for the coarse pass relative to L1 opacity (avoid over-bright).")]
    public float coarseOpacityMultiplier = 0.35f;

    [Header("Mixture hysteresis (meters)")]
    [Tooltip("Avoid loading/unloading fine/coarse near mixture thresholds.")]
    public float mixtureHysteresis = 1.0f;

    [Header("Per-frame reload budget")]
    [Tooltip("Limits how many loaders may call ReloadData() per frame to avoid VR hitching.")]
    public int maxReloadsPerFrame = 1;

    [Header("Streaming (chunk load/unload)")]
    [Tooltip("Ensure chunks closer than this distance are loaded (LoadData/ReloadData).")]
    public float loadRadius = 30f;

    [Tooltip("Unload chunks farther than this distance (free GPU buffers). Should be slightly larger than loadRadius to avoid oscillation.")]
    public float unloadRadius = 40f;

    [Header("Debug")]
    public bool loadOnStart = true;
    public bool logChunkInfo = true;
    public bool logStreamingChanges = true;
    public bool logCullingChanges = false;

    private readonly Plane[] _frustumPlanes = new Plane[6];

    [Serializable] public class LODLevel { public string filename; public int count; }
    [Serializable] public class LODGroup { public LODLevel L0; public LODLevel L1; public LODLevel L2; }

    [Serializable]
    public class ChunkEntry
    {
        public int[] ijk;
        public string filename;
        public int count;
        public float[] bbox_min;
        public float[] bbox_max;
        public float[] center;
        public LODGroup lod;
    }

    [Serializable]
    public class LODIndex
    {
        public string npz_source;
        public float[] origin;
        public float[] chunk_size;
        public int[] grid_shape;
        public int num_points;
        public int num_chunks;
        public int[] lod_levels;
        public List<ChunkEntry> chunks;
    }

    private LODIndex _lodIndex;
    private Transform _chunkRoot;

    private Material _pointMaterialFineInstance;
    private Material _pointMaterialCoarseInstance;
    private Material _rawPointMaterialInstance;

    private readonly List<GaussianLoader> _chunkLoaders = new();
    private readonly List<GaussianLoader> _chunkLoadersCoarse = new();
    private readonly List<ChunkEntry> _chunkEntries = new();
    private readonly List<bool> _chunkVisibility = new();
    private readonly List<bool> _isChunkLoaded = new();
    private readonly List<bool> _isFineLoaded = new();
    private readonly List<bool> _isCoarseLoaded = new();

    private DisplayMode _lastDisplayMode;
    private bool _lastRenderAsSplatQuads;
    private float _lastPointSizeL0;
    private float _lastPointSizeL1;
    private float _lastOpacityL0;
    private float _lastOpacityL1;
    private float _lastSigmaL0;
    private float _lastSigmaL1;
    private float _lastMinAxisPx;
    private float _lastMaxAxisPx;
    private float _lastRawPointSize;
    private bool _lastScaleMixture;

    private GUIStyle _panelStyle;
    private GUIStyle _buttonStyle;
    private GUIStyle _labelStyle;

    private bool UseGaussianSplatMode => currentDisplayMode == DisplayMode.GaussianSplat;
    private bool UseScaleMixtureForCurrentMode => UseGaussianSplatMode && enableScaleMixture;

    private void Start()
    {
        if (!loadOnStart) return;

        LoadLODIndex();
        InitChunks();
        CacheRenderParams();
    }

    private void OnDestroy()
    {
        DestroyRuntimeMaterial(ref _pointMaterialFineInstance);
        DestroyRuntimeMaterial(ref _pointMaterialCoarseInstance);
        DestroyRuntimeMaterial(ref _rawPointMaterialInstance);
    }

    private void DestroyRuntimeMaterial(ref Material materialInstance)
    {
        if (materialInstance == null)
            return;

        Destroy(materialInstance);
        materialInstance = null;
    }

    private void LoadLODIndex()
    {
        var dir = Path.Combine(Application.streamingAssetsPath, chunksFolderName);
        var path = Path.Combine(dir, lodIndexFileName);

        if (!File.Exists(path))
        {
            Debug.LogError($"[GaussianChunkManager] LOD index file not found: {path}");
            return;
        }

        var json = File.ReadAllText(path);
        _lodIndex = JsonUtility.FromJson<LODIndex>(json);

        if (_lodIndex == null || _lodIndex.chunks == null)
        {
            Debug.LogError("[GaussianChunkManager] Failed to parse LOD index JSON.");
            return;
        }

        Debug.Log($"[GaussianChunkManager] Loaded LOD index, chunks={_lodIndex.chunks.Count}");
    }

    private Material GetGaussianFineMaterial()
    {
        if (pointMaterial == null)
            return null;

        if (_pointMaterialFineInstance == null)
            _pointMaterialFineInstance = new Material(pointMaterial);

        return _pointMaterialFineInstance;
    }

    private Material GetGaussianCoarseMaterial()
    {
        if (pointMaterial == null)
            return null;

        if (_pointMaterialCoarseInstance == null)
            _pointMaterialCoarseInstance = new Material(pointMaterial);

        return _pointMaterialCoarseInstance;
    }

    private Material GetRawPointMaterial()
    {
        if (_rawPointMaterialInstance != null)
            return _rawPointMaterialInstance;

        if (rawPointMaterial != null)
        {
            _rawPointMaterialInstance = new Material(rawPointMaterial);
            return _rawPointMaterialInstance;
        }

        Shader rawShader = Shader.Find("Unlit/RawPointCloud");
        if (rawShader == null)
        {
            Debug.LogError("[GaussianChunkManager] Shader 'Unlit/RawPointCloud' not found.");
            return null;
        }

        _rawPointMaterialInstance = new Material(rawShader);
        return _rawPointMaterialInstance;
    }

    private void InitChunks()
    {
        if (_lodIndex == null || _lodIndex.chunks == null)
        {
            Debug.LogError("[GaussianChunkManager] LOD index not loaded.");
            return;
        }

        if (pointMaterial == null)
        {
            Debug.LogError("[GaussianChunkManager] pointMaterial is not assigned!");
            return;
        }

        if (_chunkRoot != null)
            DestroyImmediate(_chunkRoot.gameObject);

        _chunkLoaders.Clear();
        _chunkLoadersCoarse.Clear();
        _chunkEntries.Clear();
        _chunkVisibility.Clear();
        _isChunkLoaded.Clear();
        _isFineLoaded.Clear();
        _isCoarseLoaded.Clear();

        _chunkRoot = new GameObject("GaussianChunksRoot").transform;
        _chunkRoot.SetParent(null, false);
        _chunkRoot.position = Vector3.zero;
        _chunkRoot.rotation = Quaternion.identity;
        _chunkRoot.localScale = Vector3.one;

        string chunkDir = Path.Combine(Application.streamingAssetsPath, chunksFolderName);

        foreach (var entry in _lodIndex.chunks)
        {
            if (entry == null || entry.count <= 0)
                continue;

            string l0File = GetLODFileName(entry, 0);
            string l1File = GetLODFileName(entry, 1);

            if (string.IsNullOrEmpty(l0File) || string.IsNullOrEmpty(l1File))
                continue;

            string fullL0 = Path.Combine(chunkDir, l0File);
            string fullL1 = Path.Combine(chunkDir, l1File);

            if (!File.Exists(fullL0) || !File.Exists(fullL1))
                continue;

            if (logChunkInfo)
                Debug.Log($"[GaussianChunkManager] {entry.filename ?? "chunk"}: L0={l0File}, L1={l1File}");

            string chunkName = $"Chunk_{entry.ijk[0]}_{entry.ijk[1]}_{entry.ijk[2]}";
            GameObject go = new GameObject(chunkName);
            go.transform.SetParent(_chunkRoot, false);

            var fine = go.AddComponent<GaussianLoader>();
            fine.treatInputAsWorldSpace = true;
            fine.dataFileName = Path.Combine(chunksFolderName, l0File);

            var coarseGO = new GameObject(chunkName + "_Coarse");
            coarseGO.transform.SetParent(go.transform, false);
            var coarse = coarseGO.AddComponent<GaussianLoader>();
            coarse.treatInputAsWorldSpace = true;
            coarse.dataFileName = Path.Combine(chunksFolderName, l1File);

            ApplyFineParams(fine);
            ApplyCoarseParams(coarse);

            fine.enabled = false;
            coarse.enabled = false;

            _chunkLoaders.Add(fine);
            _chunkLoadersCoarse.Add(coarse);
            _chunkEntries.Add(entry);
            _chunkVisibility.Add(true);
            _isChunkLoaded.Add(false);
            _isFineLoaded.Add(false);
            _isCoarseLoaded.Add(false);
        }

        _lastDisplayMode = currentDisplayMode;
    }

    private void ConfigureLoaderForCurrentMode(GaussianLoader loader, bool isCoarsePass)
    {
        if (loader == null)
            return;

        Material baseMaterial;
        bool drawAsSplat;
        bool useOIT;

        if (UseGaussianSplatMode)
        {
            baseMaterial = isCoarsePass ? GetGaussianCoarseMaterial() : GetGaussianFineMaterial();
            drawAsSplat = renderAsSplatQuads;
            useOIT = renderAsSplatQuads;
        }
        else
        {
            baseMaterial = GetRawPointMaterial();
            drawAsSplat = false;
            useOIT = false;
        }

        if (baseMaterial != null)
            loader.ConfigureRendering(baseMaterial, drawAsSplat, useOIT);
    }

    private void ApplyFineParams(GaussianLoader loader)
    {
        if (loader == null) return;

        ConfigureLoaderForCurrentMode(loader, false);

        if (UseGaussianSplatMode)
        {
            loader.pointSize = pointSizeL0;
            loader.opacity = opacityL0;
            loader.sigmaCutoff = sigmaCutoffL0;
            loader.minAxisPixels = minAxisPixels;
            loader.maxAxisPixels = maxAxisPixels;
            loader.pointSizeMultiplier = 1.0f;
            loader.opacityMultiplier = 1.0f;
            loader.enableViewZFade = UseScaleMixtureForCurrentMode;
            loader.viewZFadeStart = mixtureStart;
            loader.viewZFadeEnd = mixtureEnd;
            loader.viewZFadeExponent = 1.0f;
            loader.invertViewZFade = true;
        }
        else
        {
            loader.pointSize = rawPointSize;
            loader.opacity = 1.0f;
            loader.sigmaCutoff = 1.0f;
            loader.minAxisPixels = 0.0f;
            loader.maxAxisPixels = 0.0f;
            loader.pointSizeMultiplier = 1.0f;
            loader.opacityMultiplier = 1.0f;
            loader.enableViewZFade = false;
            loader.invertViewZFade = false;
        }
    }

    private void ApplyCoarseParams(GaussianLoader loader)
    {
        if (loader == null) return;

        ConfigureLoaderForCurrentMode(loader, true);

        if (UseGaussianSplatMode)
        {
            loader.pointSize = pointSizeL1;
            loader.opacity = opacityL1;
            loader.sigmaCutoff = sigmaCutoffL1;
            loader.minAxisPixels = minAxisPixels;
            loader.maxAxisPixels = maxAxisPixels;
            loader.pointSizeMultiplier = coarsePointSizeMultiplier;
            loader.opacityMultiplier = coarseOpacityMultiplier;
            loader.enableViewZFade = UseScaleMixtureForCurrentMode;
            loader.viewZFadeStart = mixtureStart;
            loader.viewZFadeEnd = mixtureEnd;
            loader.viewZFadeExponent = 1.0f;
            loader.invertViewZFade = false;
        }
        else
        {
            loader.pointSize = rawPointSize;
            loader.opacity = 1.0f;
            loader.sigmaCutoff = 1.0f;
            loader.minAxisPixels = 0.0f;
            loader.maxAxisPixels = 0.0f;
            loader.pointSizeMultiplier = 1.0f;
            loader.opacityMultiplier = 1.0f;
            loader.enableViewZFade = false;
            loader.invertViewZFade = false;
        }
    }

    private bool RenderParamsChanged()
    {
        if (_lastDisplayMode != currentDisplayMode) return true;
        if (_lastRenderAsSplatQuads != renderAsSplatQuads) return true;
        if (_lastScaleMixture != enableScaleMixture) return true;
        if (!Mathf.Approximately(_lastPointSizeL0, pointSizeL0) || !Mathf.Approximately(_lastPointSizeL1, pointSizeL1)) return true;
        if (!Mathf.Approximately(_lastOpacityL0, opacityL0) || !Mathf.Approximately(_lastOpacityL1, opacityL1)) return true;
        if (!Mathf.Approximately(_lastSigmaL0, sigmaCutoffL0) || !Mathf.Approximately(_lastSigmaL1, sigmaCutoffL1)) return true;
        if (!Mathf.Approximately(_lastMinAxisPx, minAxisPixels) || !Mathf.Approximately(_lastMaxAxisPx, maxAxisPixels)) return true;
        if (!Mathf.Approximately(_lastRawPointSize, rawPointSize)) return true;

        return false;
    }

    private void CacheRenderParams()
    {
        _lastDisplayMode = currentDisplayMode;
        _lastRenderAsSplatQuads = renderAsSplatQuads;
        _lastScaleMixture = enableScaleMixture;
        _lastPointSizeL0 = pointSizeL0;
        _lastPointSizeL1 = pointSizeL1;
        _lastOpacityL0 = opacityL0;
        _lastOpacityL1 = opacityL1;
        _lastSigmaL0 = sigmaCutoffL0;
        _lastSigmaL1 = sigmaCutoffL1;
        _lastMinAxisPx = minAxisPixels;
        _lastMaxAxisPx = maxAxisPixels;
        _lastRawPointSize = rawPointSize;
    }

    private void ApplyParamsToAllLoaders()
    {
        for (int i = 0; i < _chunkLoaders.Count; i++)
        {
            ApplyFineParams(_chunkLoaders[i]);
            ApplyCoarseParams(_chunkLoadersCoarse[i]);
        }

        CacheRenderParams();
    }

    private void SetDisplayMode(DisplayMode mode)
    {
        if (currentDisplayMode == mode)
            return;

        currentDisplayMode = mode;
        ApplyParamsToAllLoaders();

        for (int i = 0; i < _chunkLoaders.Count; i++)
        {
            if (!UseGaussianSplatMode && _isCoarseLoaded[i])
            {
                _chunkLoadersCoarse[i].UnloadData();
                _chunkLoadersCoarse[i].enabled = false;
                _isCoarseLoaded[i] = false;
            }
        }
    }

    private void HandleModeToggleInput()
    {
        if (!allowKeyboardToggle || Keyboard.current == null)
            return;

        if (Keyboard.current.tabKey.wasPressedThisFrame)
        {
            SetDisplayMode(UseGaussianSplatMode ? DisplayMode.RawPointCloud : DisplayMode.GaussianSplat);
        }
    }

    private bool IsChunkVisible(Plane[] planes, Vector3 bmin, Vector3 bmax)
    {
        foreach (var p in planes)
        {
            Vector3 v = new Vector3(
                p.normal.x >= 0 ? bmax.x : bmin.x,
                p.normal.y >= 0 ? bmax.y : bmin.y,
                p.normal.z >= 0 ? bmax.z : bmin.z
            );

            if (Vector3.Dot(p.normal, v) + p.distance < 0)
                return false;
        }
        return true;
    }

    private static float DistancePointToAABB(Vector3 p, Vector3 bmin, Vector3 bmax)
    {
        float dx = 0f;
        if (p.x < bmin.x) dx = bmin.x - p.x;
        else if (p.x > bmax.x) dx = p.x - bmax.x;

        float dy = 0f;
        if (p.y < bmin.y) dy = bmin.y - p.y;
        else if (p.y > bmax.y) dy = p.y - bmax.y;

        float dz = 0f;
        if (p.z < bmin.z) dz = bmin.z - p.z;
        else if (p.z > bmax.z) dz = p.z - bmax.z;

        return Mathf.Sqrt(dx * dx + dy * dy + dz * dz);
    }

    private string GetLODFileName(ChunkEntry entry, int lodLevel)
    {
        if (entry.lod != null)
        {
            if (lodLevel == 0 && entry.lod.L0 != null && !string.IsNullOrEmpty(entry.lod.L0.filename))
                return entry.lod.L0.filename;
            if (lodLevel == 1 && entry.lod.L1 != null && !string.IsNullOrEmpty(entry.lod.L1.filename))
                return entry.lod.L1.filename;
            if (lodLevel == 2 && entry.lod.L2 != null && !string.IsNullOrEmpty(entry.lod.L2.filename))
                return entry.lod.L2.filename;
        }

        if (entry.lod != null && entry.lod.L0 != null && !string.IsNullOrEmpty(entry.lod.L0.filename))
            return entry.lod.L0.filename;

        return entry.filename;
    }

    private void Update()
    {
        HandleModeToggleInput();

        if (RenderParamsChanged())
            ApplyParamsToAllLoaders();

        if (Camera.main == null || _lodIndex == null)
            return;

        int reloadsThisFrame = 0;

        GeometryUtility.CalculateFrustumPlanes(Camera.main, _frustumPlanes);
        Vector3 camPos = Camera.main.transform.position;

        if (unloadRadius < loadRadius)
            unloadRadius = loadRadius + 1f;

        float hMix = Mathf.Max(0f, mixtureHysteresis);

        for (int i = 0; i < _chunkLoaders.Count; i++)
        {
            var fine = _chunkLoaders[i];
            var coarse = _chunkLoadersCoarse[i];
            var entry = _chunkEntries[i];

            if (fine == null || entry == null || entry.bbox_min == null || entry.bbox_max == null)
                continue;

            Vector3 bmin = new Vector3(entry.bbox_min[0], entry.bbox_min[1], entry.bbox_min[2]);
            Vector3 bmax = new Vector3(entry.bbox_max[0], entry.bbox_max[1], entry.bbox_max[2]);

            bool visible = IsChunkVisible(_frustumPlanes, bmin, bmax);

            if (logCullingChanges && _chunkVisibility[i] != visible)
            {
                _chunkVisibility[i] = visible;
                Debug.Log($"[ChunkCulling] {fine.gameObject.name} visible = {visible}");
            }

            Vector3 center = (entry.center != null && entry.center.Length >= 3)
                ? new Vector3(entry.center[0], entry.center[1], entry.center[2])
                : (bmin + bmax) * 0.5f;

            float distCenter = Vector3.Distance(camPos, center);
            float distAabb = DistancePointToAABB(camPos, bmin, bmax);

            bool shouldLoadChunk = distAabb < loadRadius;
            bool shouldUnloadChunk = distAabb > unloadRadius;

            if (_isChunkLoaded[i] && shouldUnloadChunk)
            {
                if (_isFineLoaded[i]) { fine.UnloadData(); _isFineLoaded[i] = false; }
                if (_isCoarseLoaded[i]) { coarse.UnloadData(); _isCoarseLoaded[i] = false; }

                _isChunkLoaded[i] = false;
                fine.enabled = false;
                coarse.enabled = false;

                if (logStreamingChanges)
                    Debug.Log($"[Streaming] Unload {fine.gameObject.name} (aabb={distAabb:F1}m, center={distCenter:F1}m)");

                continue;
            }

            if (!shouldLoadChunk && !_isChunkLoaded[i])
            {
                fine.enabled = false;
                coarse.enabled = false;
                continue;
            }

            bool wantFine = true;
            bool wantCoarse = false;

            if (UseScaleMixtureForCurrentMode)
            {
                wantFine = _isFineLoaded[i] ? (distCenter <= (mixtureEnd + hMix)) : (distCenter < (mixtureEnd - hMix));
                wantCoarse = _isCoarseLoaded[i] ? (distCenter >= (mixtureStart - hMix)) : (distCenter > (mixtureStart + hMix));
            }

            if (!_isChunkLoaded[i] && shouldLoadChunk)
                _isChunkLoaded[i] = true;

            if (wantFine && !_isFineLoaded[i] && reloadsThisFrame < Mathf.Max(0, maxReloadsPerFrame))
            {
                ApplyFineParams(fine);
                fine.ReloadData();
                _isFineLoaded[i] = true;
                reloadsThisFrame++;

                if (logStreamingChanges)
                    Debug.Log($"[PassLoad] Fine load {fine.gameObject.name} (center={distCenter:F1}m)");
            }
            else if (!wantFine && _isFineLoaded[i])
            {
                fine.UnloadData();
                _isFineLoaded[i] = false;

                if (logStreamingChanges)
                    Debug.Log($"[PassUnload] Fine unload {fine.gameObject.name} (center={distCenter:F1}m)");
            }

            if (wantCoarse && !_isCoarseLoaded[i] && reloadsThisFrame < Mathf.Max(0, maxReloadsPerFrame))
            {
                ApplyCoarseParams(coarse);
                coarse.ReloadData();
                _isCoarseLoaded[i] = true;
                reloadsThisFrame++;

                if (logStreamingChanges)
                    Debug.Log($"[PassLoad] Coarse load {fine.gameObject.name} (center={distCenter:F1}m)");
            }
            else if (!wantCoarse && _isCoarseLoaded[i])
            {
                coarse.UnloadData();
                _isCoarseLoaded[i] = false;

                if (logStreamingChanges)
                    Debug.Log($"[PassUnload] Coarse unload {fine.gameObject.name} (center={distCenter:F1}m)");
            }

            fine.enabled = visible && _isFineLoaded[i];
            coarse.enabled = visible && _isCoarseLoaded[i];

            if (_isFineLoaded[i]) ApplyFineParams(fine);
            if (_isCoarseLoaded[i]) ApplyCoarseParams(coarse);
        }
    }

    private void EnsureGUIStyles()
    {
        if (_panelStyle != null)
            return;

        _panelStyle = new GUIStyle(GUI.skin.box)
        {
            alignment = TextAnchor.UpperLeft,
            fontSize = 16,
            padding = new RectOffset(14, 14, 10, 10)
        };

        _buttonStyle = new GUIStyle(GUI.skin.button)
        {
            fontSize = 15,
            fixedHeight = 34
        };

        _labelStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = 15,
            wordWrap = true
        };
    }

    private void OnGUI()
    {
        if (!showModeSwitcherUI)
            return;

        EnsureGUIStyles();

        Rect panelRect = new Rect(16, 16, 280, 132);
        GUILayout.BeginArea(panelRect, _panelStyle);
        GUILayout.Label("Display Mode", _labelStyle);
        GUILayout.Label($"Current: {(UseGaussianSplatMode ? "Gaussian Splat" : "Raw Point Cloud")}", _labelStyle);

        if (GUILayout.Button("Gaussian Splat View", _buttonStyle))
            SetDisplayMode(DisplayMode.GaussianSplat);

        if (GUILayout.Button("Raw Point Cloud View", _buttonStyle))
            SetDisplayMode(DisplayMode.RawPointCloud);

        if (allowKeyboardToggle)
            GUILayout.Label("Press Tab to switch view.", _labelStyle);

        GUILayout.EndArea();
    }
}
