using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.InputSystem;
using Unity.XR.CoreUtils;

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

    [Serializable]
    public class DatasetDefinition
    {
        public string displayName = "Indoor";
        public string chunksFolderName = "chunks_Indoordata";
        public string lodIndexFileName = "chunks_lod_index.json";
    }

    [Header("Dataset Switching")]
    public DatasetDefinition indoorDataset = new DatasetDefinition
    {
        displayName = "Indoor",
        chunksFolderName = "chunks_Indoordata",
        lodIndexFileName = "chunks_lod_index.json"
    };

    public DatasetDefinition outdoorDataset = new DatasetDefinition
    {
        displayName = "Outdoor",
        chunksFolderName = "chunks_TUMv2",
        lodIndexFileName = "chunks_lod_index.json"
    };

    public bool showDatasetSwitcherUI = true;
    public bool recenterCameraOnDatasetSwitch = true;
    public float cameraDistancePadding = 4f;
    public float cameraHeightPadding = 2f;

    [Header("XR Scene Adaption")]
    [Tooltip("Automatically treat the scene as XR when an XR Origin is present.")]
    public bool autoDetectXRScene = true;

    [Tooltip("Optional override for the transform that should be moved when recentering in VR.")]
    public Transform xrRigRootOverride;

    [Tooltip("Hide the legacy OnGUI panel when running in an XR scene.")]
    public bool hideLegacyOnGUIWhenXRActive = true;

    [Tooltip("Disable the Tab keyboard toggle in XR scenes so desktop input does not interfere with VR interaction.")]
    public bool disableKeyboardToggleWhenXRActive = true;

    [Header("XR Standing Height")]
    [Tooltip("When recentering in XR, place the user relative to the dataset floor instead of above the dataset center.")]
    public bool useDatasetFloorHeightForXRRecenter = true;

    [Tooltip("Desired eye height above the dataset floor in meters for XR standing mode.")]
    public float xrStandingEyeHeight = 1.65f;

    [Tooltip("Extra vertical offset added on top of the standing eye height in XR scenes.")]
    public float xrAdditionalHeightOffset = 0.0f;

    [Tooltip("Keep the XR view mostly level when recentering instead of forcing a downward look toward dataset center.")]
    public bool xrKeepLevelViewOnRecenter = true;

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
    [Tooltip("Wait until the first rendered frame before beginning heavy dataset initialization. Helps SteamVR/OpenXR dismiss the loading compositor.")]
    public bool waitForFirstRenderedFrameBeforeInitialLoad = true;

    [Tooltip("Extra delay before the initial dataset load starts. Useful when VR runtimes need more time to stabilize.")]
    public float initialLoadDelaySeconds = 1.0f;

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
    private XROrigin _xrOrigin;

    private bool UseGaussianSplatMode => currentDisplayMode == DisplayMode.GaussianSplat;
    private bool UseScaleMixtureForCurrentMode => UseGaussianSplatMode && enableScaleMixture;
    private string CurrentDatasetLabel => GetDatasetLabelForCurrentSelection();

    private IEnumerator Start()
    {
        if (!loadOnStart)
            yield break;

        if (waitForFirstRenderedFrameBeforeInitialLoad)
        {
            yield return null;
            yield return new WaitForEndOfFrame();
        }

        if (initialLoadDelaySeconds > 0f)
            yield return new WaitForSeconds(initialLoadDelaySeconds);

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
        if (!TryLoadLODIndex(chunksFolderName, lodIndexFileName, out _lodIndex))
            return;

        Debug.Log($"[GaussianChunkManager] Loaded LOD index, chunks={_lodIndex.chunks.Count}");
    }

    private bool TryLoadLODIndex(string folderName, string indexFileName, out LODIndex lodIndex)
    {
        lodIndex = null;

        var dir = Path.Combine(Application.streamingAssetsPath, folderName);
        var path = Path.Combine(dir, indexFileName);

        if (!File.Exists(path))
        {
            Debug.LogError($"[GaussianChunkManager] LOD index file not found: {path}");
            return false;
        }

        var json = File.ReadAllText(path);
        lodIndex = JsonUtility.FromJson<LODIndex>(json);

        if (lodIndex == null || lodIndex.chunks == null)
        {
            Debug.LogError($"[GaussianChunkManager] Failed to parse LOD index JSON: {path}");
            lodIndex = null;
            return false;
        }

        return true;
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

        ClearActiveChunks();

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

    private void ClearActiveChunks()
    {
        for (int i = 0; i < _chunkLoaders.Count; i++)
        {
            if (_chunkLoaders[i] != null)
            {
                _chunkLoaders[i].UnloadData();
                _chunkLoaders[i].enabled = false;
            }

            if (_chunkLoadersCoarse[i] != null)
            {
                _chunkLoadersCoarse[i].UnloadData();
                _chunkLoadersCoarse[i].enabled = false;
            }
        }

        if (_chunkRoot != null)
        {
            if (Application.isPlaying)
                Destroy(_chunkRoot.gameObject);
            else
                DestroyImmediate(_chunkRoot.gameObject);

            _chunkRoot = null;
        }

        _chunkLoaders.Clear();
        _chunkLoadersCoarse.Clear();
        _chunkEntries.Clear();
        _chunkVisibility.Clear();
        _isChunkLoaded.Clear();
        _isFineLoaded.Clear();
        _isCoarseLoaded.Clear();
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

    public void SwitchToIndoorDataset()
    {
        SwitchDataset(indoorDataset);
    }

    public void SwitchToOutdoorDataset()
    {
        SwitchDataset(outdoorDataset);
    }

    public void ReloadCurrentDataset()
    {
        var currentDataset = new DatasetDefinition
        {
            displayName = CurrentDatasetLabel,
            chunksFolderName = chunksFolderName,
            lodIndexFileName = lodIndexFileName
        };

        SwitchDataset(currentDataset);
    }

    private void SwitchDataset(DatasetDefinition dataset)
    {
        if (dataset == null)
        {
            Debug.LogError("[GaussianChunkManager] Dataset definition is null.");
            return;
        }

        if (string.IsNullOrWhiteSpace(dataset.chunksFolderName) || string.IsNullOrWhiteSpace(dataset.lodIndexFileName))
        {
            Debug.LogError("[GaussianChunkManager] Dataset folder or LOD index file is empty.");
            return;
        }

        if (!TryLoadLODIndex(dataset.chunksFolderName, dataset.lodIndexFileName, out var nextLodIndex))
            return;

        chunksFolderName = dataset.chunksFolderName;
        lodIndexFileName = dataset.lodIndexFileName;
        _lodIndex = nextLodIndex;

        InitChunks();
        ApplyParamsToAllLoaders();

        if (recenterCameraOnDatasetSwitch)
            RecenterMainCameraToDataset(_lodIndex);

        Debug.Log($"[GaussianChunkManager] Switched dataset to {dataset.displayName} ({dataset.chunksFolderName})");
    }

    private void HandleModeToggleInput()
    {
        if (disableKeyboardToggleWhenXRActive && IsXRSceneActive())
            return;

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

    private string GetDatasetLabelForCurrentSelection()
    {
        if (MatchesDataset(indoorDataset))
            return indoorDataset.displayName;

        if (MatchesDataset(outdoorDataset))
            return outdoorDataset.displayName;

        return chunksFolderName;
    }

    private bool MatchesDataset(DatasetDefinition dataset)
    {
        if (dataset == null)
            return false;

        return string.Equals(chunksFolderName, dataset.chunksFolderName, StringComparison.OrdinalIgnoreCase)
               && string.Equals(lodIndexFileName, dataset.lodIndexFileName, StringComparison.OrdinalIgnoreCase);
    }

    private bool TryGetDatasetBounds(LODIndex lodIndex, out Vector3 boundsMin, out Vector3 boundsMax)
    {
        boundsMin = Vector3.zero;
        boundsMax = Vector3.zero;

        if (lodIndex == null || lodIndex.chunks == null || lodIndex.chunks.Count == 0)
            return false;

        bool initialized = false;

        foreach (var chunk in lodIndex.chunks)
        {
            if (chunk == null || chunk.bbox_min == null || chunk.bbox_max == null ||
                chunk.bbox_min.Length < 3 || chunk.bbox_max.Length < 3)
            {
                continue;
            }

            Vector3 bmin = new Vector3(chunk.bbox_min[0], chunk.bbox_min[1], chunk.bbox_min[2]);
            Vector3 bmax = new Vector3(chunk.bbox_max[0], chunk.bbox_max[1], chunk.bbox_max[2]);

            if (!initialized)
            {
                boundsMin = bmin;
                boundsMax = bmax;
                initialized = true;
                continue;
            }

            boundsMin = Vector3.Min(boundsMin, bmin);
            boundsMax = Vector3.Max(boundsMax, bmax);
        }

        return initialized;
    }

    private void RecenterMainCameraToDataset(LODIndex lodIndex)
    {
        var mainCamera = Camera.main;
        if (mainCamera == null)
            return;

        if (!TryGetDatasetBounds(lodIndex, out var boundsMin, out var boundsMax))
            return;

        Vector3 center = (boundsMin + boundsMax) * 0.5f;
        Vector3 size = boundsMax - boundsMin;

        float horizontalExtent = Mathf.Max(size.x, size.z) * 0.5f;
        float cameraDistance = Mathf.Max(5f, horizontalExtent + Mathf.Max(0f, cameraDistancePadding));
        bool hasXRRoot = TryGetXRRecenterRoot(mainCamera, out Transform xrRoot);

        Vector3 targetPosition;
        Quaternion targetRotation;

        if (hasXRRoot && useDatasetFloorHeightForXRRecenter)
        {
            float floorY = boundsMin.y;
            float eyeY = floorY + Mathf.Max(0f, xrStandingEyeHeight) + xrAdditionalHeightOffset;
            targetPosition = new Vector3(center.x, eyeY, center.z - cameraDistance);

            Vector3 lookTarget = xrKeepLevelViewOnRecenter
                ? new Vector3(center.x, eyeY, center.z)
                : center;

            targetRotation = Quaternion.LookRotation(lookTarget - targetPosition, Vector3.up);
        }
        else
        {
            float cameraHeight = Mathf.Max(2f, size.y * 0.35f + Mathf.Max(0f, cameraHeightPadding));
            targetPosition = center + new Vector3(0f, cameraHeight, -cameraDistance);
            targetRotation = Quaternion.LookRotation(center - targetPosition, Vector3.up);
        }

        if (hasXRRoot)
        {
            Vector3 localCameraPosition = xrRoot.InverseTransformPoint(mainCamera.transform.position);
            Quaternion localCameraRotation = Quaternion.Inverse(xrRoot.rotation) * mainCamera.transform.rotation;

            Quaternion desiredRigRotation = targetRotation * Quaternion.Inverse(localCameraRotation);
            Vector3 desiredRigPosition = targetPosition - (desiredRigRotation * localCameraPosition);

            xrRoot.SetPositionAndRotation(desiredRigPosition, desiredRigRotation);
            return;
        }

        mainCamera.transform.SetPositionAndRotation(targetPosition, targetRotation);
    }

    private bool IsXRSceneActive()
    {
        if (!autoDetectXRScene)
            return xrRigRootOverride != null;

        if (xrRigRootOverride != null)
            return true;

        if (_xrOrigin == null)
            _xrOrigin = FindFirstObjectByType<XROrigin>();

        return _xrOrigin != null;
    }

    private bool TryGetXRRecenterRoot(Camera mainCamera, out Transform xrRoot)
    {
        xrRoot = null;

        if (!IsXRSceneActive())
            return false;

        if (xrRigRootOverride != null)
        {
            xrRoot = xrRigRootOverride;
            return true;
        }

        if (_xrOrigin == null)
            _xrOrigin = FindFirstObjectByType<XROrigin>();

        if (_xrOrigin != null)
        {
            xrRoot = _xrOrigin.transform;
            return true;
        }

        if (mainCamera != null)
        {
            _xrOrigin = mainCamera.GetComponentInParent<XROrigin>();
            if (_xrOrigin != null)
            {
                xrRoot = _xrOrigin.transform;
                return true;
            }
        }

        return false;
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
        if (hideLegacyOnGUIWhenXRActive && IsXRSceneActive())
            return;

        if (!showModeSwitcherUI && !showDatasetSwitcherUI)
            return;

        EnsureGUIStyles();

        float panelHeight = 24f;
        if (showDatasetSwitcherUI)
            panelHeight += 128f;
        if (showModeSwitcherUI)
            panelHeight += allowKeyboardToggle ? 150f : 124f;

        Rect panelRect = new Rect(16, 16, 320, panelHeight);
        GUILayout.BeginArea(panelRect, _panelStyle);

        if (showDatasetSwitcherUI)
        {
            GUILayout.Label("Scene Dataset", _labelStyle);
            GUILayout.Label($"Current: {CurrentDatasetLabel}", _labelStyle);

            bool previousEnabled = GUI.enabled;

            GUI.enabled = !MatchesDataset(indoorDataset);
            if (GUILayout.Button(indoorDataset.displayName, _buttonStyle))
                SwitchToIndoorDataset();

            GUI.enabled = !MatchesDataset(outdoorDataset);
            if (GUILayout.Button(outdoorDataset.displayName, _buttonStyle))
                SwitchToOutdoorDataset();

            GUI.enabled = previousEnabled;
            GUILayout.Space(8f);
        }

        if (showModeSwitcherUI)
        {
            GUILayout.Label("Display Mode", _labelStyle);
            GUILayout.Label($"Current: {(UseGaussianSplatMode ? "Gaussian Splat" : "Raw Point Cloud")}", _labelStyle);

            if (GUILayout.Button("Gaussian Splat View", _buttonStyle))
                SetDisplayMode(DisplayMode.GaussianSplat);

            if (GUILayout.Button("Raw Point Cloud View", _buttonStyle))
                SetDisplayMode(DisplayMode.RawPointCloud);

            if (allowKeyboardToggle)
                GUILayout.Label("Press Tab to switch view.", _labelStyle);
        }

        GUILayout.EndArea();
    }
}
