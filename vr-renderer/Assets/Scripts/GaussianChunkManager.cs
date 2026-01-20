using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Reads chunk and LOD information from StreamingAssets/{chunksFolderName}/{lodIndexFileName},
/// and creates a GameObject with GaussianLoader(s) per chunk.
/// Supports:
/// - Frustum culling
/// - Chunk streaming (load/unload radius with hysteresis)
/// - Per-frame reload budget (avoid VR hitching)
///
/// Two-level + scale mixture mode (this version):
/// - Fine pass uses ONLY L0 (detail)
/// - Coarse pass uses ONLY L2 (gap filling / stability)
/// - Opacity is blended by distance between mixtureStart and mixtureEnd (smoothstep)
/// - Fine/Coarse GPU data are loaded/unloaded by distance with hysteresis to save memory/bandwidth
/// </summary>
public class GaussianChunkManager : MonoBehaviour
{
    [Header("Chunk Folder (in StreamingAssets)")]
    public string chunksFolderName = "navvis_chunks_TUMv2";

    [Header("LOD index file name")]
    public string lodIndexFileName = "navvis_chunks_lod_index.json";

    [Header("Rendering")]
    public Material pointMaterial;      // Passed to GaussianLoader

    [Header("Splatting / Rendering Mode")]
    public bool renderAsSplatQuads = true;

    [Header("Point size (optional / legacy)")]
    public float pointSizeL0 = 1.0f;
    public float pointSizeL2 = 1.0f;

    [Header("Ellipse Splat Params (two-level)")]
    [Range(0f, 1f)] public float opacityL0 = 0.6f;
    [Range(0f, 1f)] public float opacityL2 = 0.18f;

    [Tooltip("k-sigma cutoff used to size ellipse quads (typical 2~4).")]
    public float sigmaCutoffL0 = 3.0f;
    public float sigmaCutoffL2 = 3.0f;

    [Tooltip("Clamp ellipse axis in pixel units (min).")]
    public float minAxisPixels = 0.75f;

    [Tooltip("Clamp ellipse axis in pixel units (max).")]
    public float maxAxisPixels = 64.0f;

    [Header("Scale Mixture (dual-pass, L0 + L2)")]
    [Tooltip("Render each chunk twice: Fine(L0) for detail, Coarse(L2) for gap filling. Blended by distance.")]
    public bool enableScaleMixture = true;

    [Tooltip("Distance where coarse pass starts to fade in (meters).")]
    public float mixtureStart = 8f;

    [Tooltip("Distance where coarse pass reaches full strength (meters).")]
    public float mixtureEnd = 25f;

    [Tooltip("Extra pointSize multiplier for the coarse pass (fills scanline gaps).")]
    public float coarsePointSizeMultiplier = 2.5f;

    [Tooltip("Opacity multiplier for the coarse pass relative to L2 opacity (avoid over-bright).")]
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

    // Camera frustum planes for culling
    private readonly Plane[] _frustumPlanes = new Plane[6];

    // ===== JSON mapping =====
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

    // ===== internal state =====
    private LODIndex _lodIndex;
    private Transform _chunkRoot;

    // Fine (L0) loaders
    private readonly List<GaussianLoader> _chunkLoaders = new();
    // Coarse (L2) loaders
    private readonly List<GaussianLoader> _chunkLoadersCoarse = new();

    private readonly List<ChunkEntry> _chunkEntries = new();

    private readonly List<bool> _chunkVisibility = new();

    // Streaming state per chunk & per pass
    private readonly List<bool> _isChunkLoaded = new();     // chunk exists (any pass loaded)
    private readonly List<bool> _isFineLoaded = new();      // L0 pass loaded
    private readonly List<bool> _isCoarseLoaded = new();    // L2 pass loaded

    // Cache last applied Inspector values to avoid per-frame churn
    private bool _lastRenderAsSplatQuads;
    private float _lastPointSizeL0, _lastPointSizeL2;
    private float _lastOpacityL0, _lastOpacityL2;
    private float _lastSigmaL0, _lastSigmaL2;
    private float _lastMinAxisPx, _lastMaxAxisPx;

    private void Start()
    {
        if (!loadOnStart) return;

        LoadLODIndex();
        InitChunks();
        CacheSplatParams();
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
        if (_lodIndex.chunks.Count > 0)
        {
            var e = _lodIndex.chunks[0];
            if (e.center != null && e.center.Length >= 3)
                Debug.Log($"[LODIndex] first chunk center = ({e.center[0]}, {e.center[1]}, {e.center[2]})");
            if (_lodIndex.origin != null && _lodIndex.origin.Length >= 3)
                Debug.Log($"[LODIndex] origin = ({_lodIndex.origin[0]}, {_lodIndex.origin[1]}, {_lodIndex.origin[2]})");
        }
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

        // cleanup old root
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

        int created = 0;
        string chunkDir = Path.Combine(Application.streamingAssetsPath, chunksFolderName);

        foreach (var entry in _lodIndex.chunks)
        {
            if (entry == null) continue;
            if (entry.count <= 0) continue;

            // Need L0 and L1 files (coarse uses L1; L2 is too sparse)
            string l0File = GetLODFileName(entry, 0);
            string l2File = GetLODFileName(entry, 1);

            if (string.IsNullOrEmpty(l0File) || string.IsNullOrEmpty(l2File))
                continue;

            string fullL0 = Path.Combine(chunkDir, l0File);
            string fullL2 = Path.Combine(chunkDir, l2File);

            if (!File.Exists(fullL0))
            {
                Debug.LogWarning($"[GaussianChunkManager] L0 file missing: {fullL0}");
                continue;
            }
            if (!File.Exists(fullL2))
            {
                Debug.LogWarning($"[GaussianChunkManager] L1 (coarse) file missing: {fullL2}");
                continue;
            }

            string chunkName = $"Chunk_{entry.ijk[0]}_{entry.ijk[1]}_{entry.ijk[2]}";
            GameObject go = new GameObject(chunkName);
            go.transform.SetParent(_chunkRoot, false);
            go.transform.localPosition = Vector3.zero;
            go.transform.localRotation = Quaternion.identity;
            go.transform.localScale = Vector3.one;

            // Fine loader on parent
            var fine = go.AddComponent<GaussianLoader>();
            fine.pointMaterial = pointMaterial;
            fine.treatInputAsWorldSpace = true;
            fine.renderAsSplatQuads = renderAsSplatQuads;

            // Coarse loader as child (optional)
            GaussianLoader coarse = null;
            if (enableScaleMixture)
            {
                var goCoarse = new GameObject(chunkName + "_Coarse");
                goCoarse.transform.SetParent(go.transform, false);
                goCoarse.transform.localPosition = Vector3.zero;
                goCoarse.transform.localRotation = Quaternion.identity;
                goCoarse.transform.localScale = Vector3.one;

                coarse = goCoarse.AddComponent<GaussianLoader>();
                coarse.pointMaterial = pointMaterial;
                coarse.treatInputAsWorldSpace = true;
                coarse.renderAsSplatQuads = renderAsSplatQuads;
            }

            // assign default files (we will load/unload per distance in Update)
            fine.dataFileName = Path.Combine(chunksFolderName, l0File);
            if (coarse != null) coarse.dataFileName = Path.Combine(chunksFolderName, l2File);

            // apply params
            ApplyFineParams(fine);
            if (coarse != null) ApplyCoarseParams(coarse);

            // start with NOT loaded; Update() will stream in
            fine.enabled = false;
            if (coarse != null) coarse.enabled = false;

            _chunkLoaders.Add(fine);
            _chunkLoadersCoarse.Add(coarse);
            _chunkEntries.Add(entry);

            _chunkVisibility.Add(true);
            _isChunkLoaded.Add(false);
            _isFineLoaded.Add(false);
            _isCoarseLoaded.Add(false);

            created++;
            //if (logChunkInfo)
                //Debug.Log($"[GaussianChunkManager] Created {chunkName} (L0={l0File}, L2={l2File}, count={entry.count})");
        }

        //Debug.Log($"[GaussianChunkManager] Created {created} Gaussian chunks.");
    }

    private void ApplyFineParams(GaussianLoader loader)
    {
        if (loader == null) return;

        loader.renderAsSplatQuads = renderAsSplatQuads;

        // Base params (do NOT distance-mix in C#)
        loader.pointSize = pointSizeL0;
        loader.opacity = opacityL0;
        loader.sigmaCutoff = sigmaCutoffL0;
        loader.minAxisPixels = minAxisPixels;
        loader.maxAxisPixels = maxAxisPixels;

        // Multipliers (handled inside GaussianLoader when pushing MPB)
        loader.pointSizeMultiplier = 1.0f;
        loader.opacityMultiplier = 1.0f;

        // Per-pixel scale-mixture gating in shader: Fine fades OUT with distance.
        loader.enableViewZFade = enableScaleMixture;
        loader.viewZFadeStart = mixtureStart;
        loader.viewZFadeEnd = mixtureEnd;
        loader.viewZFadeExponent = 1.0f;
        loader.invertViewZFade = true;
    }

    private void ApplyCoarseParams(GaussianLoader loader)
    {
        if (loader == null) return;

        loader.renderAsSplatQuads = renderAsSplatQuads;

        // Base params (do NOT distance-mix in C#)
        loader.pointSize = pointSizeL2;
        loader.opacity = opacityL2;
        loader.sigmaCutoff = sigmaCutoffL2;
        loader.minAxisPixels = minAxisPixels;
        loader.maxAxisPixels = maxAxisPixels;

        // Coarse strength/footprint controlled via multipliers
        loader.pointSizeMultiplier = coarsePointSizeMultiplier;
        loader.opacityMultiplier = coarseOpacityMultiplier;

        // Per-pixel scale-mixture gating in shader: Coarse fades IN with distance.
        loader.enableViewZFade = enableScaleMixture;
        loader.viewZFadeStart = mixtureStart;
        loader.viewZFadeEnd = mixtureEnd;
        loader.viewZFadeExponent = 1.0f;
        loader.invertViewZFade = false;
    }

    private bool SplatParamsChanged()
    {
        if (_lastRenderAsSplatQuads != renderAsSplatQuads) return true;

        if (!Mathf.Approximately(_lastPointSizeL0, pointSizeL0) ||
            !Mathf.Approximately(_lastPointSizeL2, pointSizeL2)) return true;

        if (!Mathf.Approximately(_lastOpacityL0, opacityL0) ||
            !Mathf.Approximately(_lastOpacityL2, opacityL2)) return true;

        if (!Mathf.Approximately(_lastSigmaL0, sigmaCutoffL0) ||
            !Mathf.Approximately(_lastSigmaL2, sigmaCutoffL2)) return true;

        if (!Mathf.Approximately(_lastMinAxisPx, minAxisPixels) ||
            !Mathf.Approximately(_lastMaxAxisPx, maxAxisPixels)) return true;

        return false;
    }

    private void CacheSplatParams()
    {
        _lastRenderAsSplatQuads = renderAsSplatQuads;

        _lastPointSizeL0 = pointSizeL0;
        _lastPointSizeL2 = pointSizeL2;

        _lastOpacityL0 = opacityL0;
        _lastOpacityL2 = opacityL2;

        _lastSigmaL0 = sigmaCutoffL0;
        _lastSigmaL2 = sigmaCutoffL2;

        _lastMinAxisPx = minAxisPixels;
        _lastMaxAxisPx = maxAxisPixels;
    }

    private void ApplyParamsToAllLoaders()
    {
        for (int i = 0; i < _chunkLoaders.Count; i++)
        {
            var fine = _chunkLoaders[i];
            ApplyFineParams(fine);

            var coarse = (i < _chunkLoadersCoarse.Count) ? _chunkLoadersCoarse[i] : null;
            if (coarse != null)
                ApplyCoarseParams(coarse);
        }

        CacheSplatParams();
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

        // Fallback: try L0 first, then entry.filename (not ideal for L2, but keeps robustness)
        if (entry.lod != null && entry.lod.L0 != null && !string.IsNullOrEmpty(entry.lod.L0.filename))
            return entry.lod.L0.filename;

        return entry.filename;
    }

    private static float Smoothstep01(float t)
    {
        t = Mathf.Clamp01(t);
        return t * t * (3f - 2f * t);
    }

    private void Update()
    {
        // Sync render params only when Inspector values change
        if (SplatParamsChanged())
            ApplyParamsToAllLoaders();

        if (Camera.main == null || _lodIndex == null)
            return;

        int reloadsThisFrame = 0;

        GeometryUtility.CalculateFrustumPlanes(Camera.main, _frustumPlanes);
        Vector3 camPos = Camera.main.transform.position;

        // Streaming guard
        if (unloadRadius < loadRadius)
            unloadRadius = loadRadius + 1f;

        float hMix = Mathf.Max(0f, mixtureHysteresis);

        for (int i = 0; i < _chunkLoaders.Count; i++)
        {
            var fine = _chunkLoaders[i];
            var coarse = (i < _chunkLoadersCoarse.Count) ? _chunkLoadersCoarse[i] : null;
            var entry = _chunkEntries[i];

            if (fine == null || entry == null) continue;
            if (entry.bbox_min == null || entry.bbox_max == null) continue;

            Vector3 bmin = new Vector3(entry.bbox_min[0], entry.bbox_min[1], entry.bbox_min[2]);
            Vector3 bmax = new Vector3(entry.bbox_max[0], entry.bbox_max[1], entry.bbox_max[2]);

            bool visible = IsChunkVisible(_frustumPlanes, bmin, bmax);

            if (logCullingChanges && _chunkVisibility[i] != visible)
            {
                _chunkVisibility[i] = visible;
                Debug.Log($"[ChunkCulling] {fine.gameObject.name} visible = {visible}");
            }

            Vector3 center;
            if (entry.center != null && entry.center.Length >= 3)
                center = new Vector3(entry.center[0], entry.center[1], entry.center[2]);
            else
                center = (bmin + bmax) * 0.5f;

            float distCenter = Vector3.Distance(camPos, center);
            float distAabb = DistancePointToAABB(camPos, bmin, bmax);

            bool shouldLoadChunk = distAabb < loadRadius;
            bool shouldUnloadChunk = distAabb > unloadRadius;

            // Unload whole chunk (both passes)
            if (_isChunkLoaded[i] && shouldUnloadChunk)
            {
                if (_isFineLoaded[i]) { fine.UnloadData(); _isFineLoaded[i] = false; }
                if (coarse != null && _isCoarseLoaded[i]) { coarse.UnloadData(); _isCoarseLoaded[i] = false; }
                _isChunkLoaded[i] = false;

                fine.enabled = false;
                if (coarse != null) coarse.enabled = false;

                if (logStreamingChanges)
                    Debug.Log($"[Streaming] Unload {fine.gameObject.name} (aabb={distAabb:F1}m, center={distCenter:F1}m)");

                continue;
            }

            // If chunk is not in load range, keep disabled
            if (!shouldLoadChunk && !_isChunkLoaded[i])
            {
                fine.enabled = false;
                if (coarse != null) coarse.enabled = false;
                continue;
            }

            // Decide which passes are needed by distance (with hysteresis)
            // Fine needed until mixtureEnd; Coarse needed starting at mixtureStart
            bool wantFine;
            bool wantCoarse;

            if (!enableScaleMixture)
            {
                // mixture off: only fine pass
                wantFine = true;
                wantCoarse = false;
            }
            else
            {
                // Hysteresis rules:
                // - Fine loads when dist < mixtureEnd - h, unloads when dist > mixtureEnd + h
                // - Coarse loads when dist > mixtureStart + h, unloads when dist < mixtureStart - h
                wantFine = _isFineLoaded[i] ? (distCenter <= (mixtureEnd + hMix)) : (distCenter < (mixtureEnd - hMix));
                wantCoarse = (coarse != null) && (_isCoarseLoaded[i] ? (distCenter >= (mixtureStart - hMix)) : (distCenter > (mixtureStart + hMix)));
            }

            // Ensure chunk marked loaded if any pass is/will be loaded
            // (we load passes below if budget allows)
            if (!_isChunkLoaded[i] && shouldLoadChunk)
                _isChunkLoaded[i] = true;

            // Load/unload fine pass
            if (wantFine && !_isFineLoaded[i])
            {
                if (reloadsThisFrame < Mathf.Max(0, maxReloadsPerFrame))
                {
                    ApplyFineParams(fine);
                    fine.ReloadData();
                    _isFineLoaded[i] = true;
                    reloadsThisFrame++;

                    if (logStreamingChanges)
                        Debug.Log($"[PassLoad] Fine(L0) load {fine.gameObject.name} (center={distCenter:F1}m)");
                }
            }
            else if (!wantFine && _isFineLoaded[i])
            {
                fine.UnloadData();
                _isFineLoaded[i] = false;

                if (logStreamingChanges)
                    Debug.Log($"[PassUnload] Fine(L0) unload {fine.gameObject.name} (center={distCenter:F1}m)");
            }

            // Load/unload coarse pass
            if (coarse != null)
            {
                if (wantCoarse && !_isCoarseLoaded[i])
                {
                    if (reloadsThisFrame < Mathf.Max(0, maxReloadsPerFrame))
                    {
                        ApplyCoarseParams(coarse);
                        coarse.ReloadData();
                        _isCoarseLoaded[i] = true;
                        reloadsThisFrame++;

                        if (logStreamingChanges)
                            Debug.Log($"[PassLoad] Coarse(L1) load {fine.gameObject.name} (center={distCenter:F1}m)");
                    }
                }
                else if (!wantCoarse && _isCoarseLoaded[i])
                {
                    coarse.UnloadData();
                    _isCoarseLoaded[i] = false;

                    if (logStreamingChanges)
                        Debug.Log($"[PassUnload] Coarse(L1) unload {fine.gameObject.name} (center={distCenter:F1}m)");
                }
            }

            // Enable renderers only if visible & pass loaded
            fine.enabled = visible && _isFineLoaded[i];
            if (coarse != null) coarse.enabled = visible && _isCoarseLoaded[i];

            // Keep loader params up-to-date (viewZ fade + multipliers). Mixture gating is per-pixel in shader.
            if (_isFineLoaded[i]) ApplyFineParams(fine);
            if (coarse != null && _isCoarseLoaded[i]) ApplyCoarseParams(coarse);

            // If mixture is disabled, ensure coarse is not rendered.
            if (!enableScaleMixture && coarse != null)
            {
                coarse.enabled = false;
            }
        }
    }
}