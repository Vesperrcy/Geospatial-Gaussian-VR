using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// 从 StreamingAssets/{chunksFolderName}/{lodIndexFileName} 读取 chunk + LOD 信息，
/// 为每个 chunk 创建一个带 GaussianLoader 的 GameObject（默认加载 L0）。
/// 支持：
/// - Frustum Culling（视锥裁剪）
/// - 距离 LOD 切换（L0/L1/L2）
/// - Chunk Streaming（load/unload 半径滞回）
///
/// 适配新版 GaussianLoader + 椭圆 Gaussian splatting：
/// - loader.opacity / loader.sigmaCutoff / loader.minAxisPixels / loader.maxAxisPixels
/// - cov0/cov1 buffers 由 loader 内部创建并绑定
/// </summary>
public class GaussianChunkManager : MonoBehaviour
{
    [Header("Chunk Folder (in StreamingAssets)")]
    // 对应 vr-renderer/Assets/StreamingAssets/navvis_chunks_Block1
    public string chunksFolderName = "navvis_chunks_Block1";

    [Header("LOD index file name")]
    public string lodIndexFileName = "navvis_chunks_lod_index.json";

    [Header("Rendering")]
    public Material pointMaterial;      // Passed to GaussianLoader

    [Header("Splatting / Rendering Mode")]
    public bool renderAsSplatQuads = true;

    // pointSize 目前对椭圆 shader 不起作用（shader 不用 _PointSize），但保留：
    // 1) 你切回点模式时可用；2) 你未来可能把它作为全局缩放项写进 shader。
    [Header("Point size (optional / legacy)")]
    public float pointSizeL0 = 1.0f;
    public float pointSizeL1 = 1.0f;
    public float pointSizeL2 = 1.0f;

    [Header("Ellipse Splat Params (per LOD)")]
    [Range(0f, 1f)] public float opacityL0 = 0.6f;
    [Range(0f, 1f)] public float opacityL1 = 0.35f;
    [Range(0f, 1f)] public float opacityL2 = 0.18f;

    [Tooltip("k-sigma cutoff used to size ellipse quads (typical 2~4).")]
    public float sigmaCutoffL0 = 3.0f;
    public float sigmaCutoffL1 = 3.0f;
    public float sigmaCutoffL2 = 3.0f;

    [Tooltip("Clamp ellipse axis in pixel units (min).")]
    public float minAxisPixels = 0.75f;

    [Tooltip("Clamp ellipse axis in pixel units (max).")]
    public float maxAxisPixels = 64.0f;

    [Header("LOD distances (meters)")]
    public float lod0To1 = 5f;          // < lod0To1 使用 L0
    public float lod1To2 = 15f;         // > lod1To2 使用 L2，中间用 L1

    [Header("Streaming (chunk load/unload)")]
    [Tooltip("Ensure chunks closer than this distance are loaded (LoadData).")]
    public float loadRadius = 30f;

    [Tooltip("Unload chunks farther than this distance (free GPU buffers). Should be slightly larger than loadRadius to avoid oscillation.")]
    public float unloadRadius = 40f;

    [Header("Debug")]
    public bool loadOnStart = true;
    public bool logChunkInfo = true;        // 创建 chunk 时的 1 条日志
    public bool logLODChanges = true;       // LOD 切换时的日志
    public bool logCullingChanges = false;  // Culling 状态改变时的日志
    [Tooltip("Log when chunks are loaded or unloaded (streaming).")]
    public bool logStreamingChanges = true;

    // Camera frustum planes for culling
    private readonly Plane[] _frustumPlanes = new Plane[6];

    // ===== JSON 映射结构（和 Python 导出的 JSON 对应） =====
    [Serializable]
    public class LODLevel
    {
        public string filename;
        public int count;
    }

    [Serializable]
    public class LODGroup
    {
        public LODLevel L0;
        public LODLevel L1;
        public LODLevel L2;
    }

    [Serializable]
    public class ChunkEntry
    {
        public int[] ijk;        // [ix, iy, iz]
        public string filename;  // 原始 chunk 文件名（无 LOD 时使用）
        public int count;
        public float[] bbox_min;
        public float[] bbox_max;
        public float[] center;   // Python 里写入的 chunk 中心（世界坐标）
        public LODGroup lod;     // 各 LOD 的文件信息
    }

    [Serializable]
    public class LODIndex
    {
        public string npz_source;
        public float[] origin;       // 整个 Block 的原点（世界坐标）
        public float[] chunk_size;
        public int[] grid_shape;
        public int num_points;
        public int num_chunks;
        public int[] lod_levels;
        public List<ChunkEntry> chunks;
    }

    // ===== 内部状态 =====
    private LODIndex _lodIndex;
    private Transform _chunkRoot;

    private readonly List<GaussianLoader> _chunkLoaders = new();
    private readonly List<ChunkEntry>     _chunkEntries = new();

    // 记录每个 chunk 的可见性状态（用于只在变化时打印 log）
    private readonly List<bool> _chunkVisibility = new();

    // 每个 chunk 当前使用的 LOD 层级（0=L0, 1=L1, 2=L2）
    private readonly List<int> _currentLOD = new();

    // 记录每个 chunk 当前是否已加载 GPU 数据（用于 Streaming）
    private readonly List<bool> _isLoaded = new();

    // Cache last applied Inspector values to avoid per-frame churn
    private bool _lastRenderAsSplatQuads;
    private float _lastPointSizeL0, _lastPointSizeL1, _lastPointSizeL2;
    private float _lastOpacityL0, _lastOpacityL1, _lastOpacityL2;
    private float _lastSigmaL0, _lastSigmaL1, _lastSigmaL2;
    private float _lastMinAxisPx, _lastMaxAxisPx;

    private void Start()
    {
        if (loadOnStart)
        {
            LoadLODIndex();
            InitChunks_L0_Only();
            CacheSplatParams();
        }
    }

    private void ApplySplatParamsToLoader(GaussianLoader loader, int lodLevel)
    {
        if (loader == null) return;

        // IMPORTANT: 确保 loader 处于 splat 模式（否则可能不绑定 cov1 -> Metal index3 warning）
        loader.renderAsSplatQuads = renderAsSplatQuads;

        // legacy / optional
        switch (lodLevel)
        {
            case 0: loader.pointSize = pointSizeL0; break;
            case 1: loader.pointSize = pointSizeL1; break;
            default: loader.pointSize = pointSizeL2; break;
        }

        // ellipse shader params
        loader.minAxisPixels = minAxisPixels;
        loader.maxAxisPixels = maxAxisPixels;

        switch (lodLevel)
        {
            case 0:
                loader.opacity = opacityL0;
                loader.sigmaCutoff = sigmaCutoffL0;
                break;
            case 1:
                loader.opacity = opacityL1;
                loader.sigmaCutoff = sigmaCutoffL1;
                break;
            default:
                loader.opacity = opacityL2;
                loader.sigmaCutoff = sigmaCutoffL2;
                break;
        }
    }

    private bool SplatParamsChanged()
    {
        if (_lastRenderAsSplatQuads != renderAsSplatQuads) return true;

        if (!Mathf.Approximately(_lastPointSizeL0, pointSizeL0) ||
            !Mathf.Approximately(_lastPointSizeL1, pointSizeL1) ||
            !Mathf.Approximately(_lastPointSizeL2, pointSizeL2)) return true;

        if (!Mathf.Approximately(_lastOpacityL0, opacityL0) ||
            !Mathf.Approximately(_lastOpacityL1, opacityL1) ||
            !Mathf.Approximately(_lastOpacityL2, opacityL2)) return true;

        if (!Mathf.Approximately(_lastSigmaL0, sigmaCutoffL0) ||
            !Mathf.Approximately(_lastSigmaL1, sigmaCutoffL1) ||
            !Mathf.Approximately(_lastSigmaL2, sigmaCutoffL2)) return true;

        if (!Mathf.Approximately(_lastMinAxisPx, minAxisPixels) ||
            !Mathf.Approximately(_lastMaxAxisPx, maxAxisPixels)) return true;

        return false;
    }

    private void CacheSplatParams()
    {
        _lastRenderAsSplatQuads = renderAsSplatQuads;

        _lastPointSizeL0 = pointSizeL0;
        _lastPointSizeL1 = pointSizeL1;
        _lastPointSizeL2 = pointSizeL2;

        _lastOpacityL0 = opacityL0;
        _lastOpacityL1 = opacityL1;
        _lastOpacityL2 = opacityL2;

        _lastSigmaL0 = sigmaCutoffL0;
        _lastSigmaL1 = sigmaCutoffL1;
        _lastSigmaL2 = sigmaCutoffL2;

        _lastMinAxisPx = minAxisPixels;
        _lastMaxAxisPx = maxAxisPixels;
    }

    private void ApplySplatParamsToAllLoaders()
    {
        if (_chunkLoaders == null || _chunkLoaders.Count == 0) return;

        for (int i = 0; i < _chunkLoaders.Count; i++)
        {
            var loader = _chunkLoaders[i];
            int lod = (i < _currentLOD.Count) ? _currentLOD[i] : 0;
            ApplySplatParamsToLoader(loader, lod);
        }

        CacheSplatParams();
    }

    /// <summary>
    /// 检查 AABB 是否与相机视锥体相交。
    /// </summary>
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

    /// <summary>
    /// 读取 StreamingAssets/{chunksFolderName}/{lodIndexFileName}
    /// </summary>
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

    /// <summary>
    /// 为每个 chunk 创建一个 GameObject + GaussianLoader，仅加载 L0。
    /// </summary>
    public void InitChunks_L0_Only()
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

        // 清理旧的 root
        if (_chunkRoot != null)
        {
            DestroyImmediate(_chunkRoot.gameObject);
        }

        _chunkLoaders.Clear();
        _chunkEntries.Clear();
        _chunkVisibility.Clear();
        _currentLOD.Clear();
        _isLoaded.Clear();

        // 创建新的 root
        _chunkRoot = new GameObject("GaussianChunksRoot_Block1").transform;
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

            // 选择 L0 文件名
            string l0FileName = null;
            if (entry.lod != null && entry.lod.L0 != null && !string.IsNullOrEmpty(entry.lod.L0.filename))
                l0FileName = entry.lod.L0.filename;
            else if (!string.IsNullOrEmpty(entry.filename))
                l0FileName = entry.filename;

            if (string.IsNullOrEmpty(l0FileName))
                continue;

            string fullPath = Path.Combine(chunkDir, l0FileName);
            if (!File.Exists(fullPath))
            {
                Debug.LogWarning($"[GaussianChunkManager] L0 file not found for chunk ijk=({entry.ijk[0]},{entry.ijk[1]},{entry.ijk[2]}): {fullPath}");
                continue;
            }

            string chunkName = $"Chunk_{entry.ijk[0]}_{entry.ijk[1]}_{entry.ijk[2]}";
            GameObject go = new GameObject(chunkName);
            go.transform.SetParent(_chunkRoot, false);
            go.transform.localPosition = Vector3.zero;
            go.transform.localRotation = Quaternion.identity;
            go.transform.localScale = Vector3.one;

            var loader = go.AddComponent<GaussianLoader>();

            // 文件路径（StreamingAssets 相对路径）
            string relativePath = Path.Combine(chunksFolderName, l0FileName);
            loader.dataFileName = relativePath;

            loader.pointMaterial = pointMaterial;
            loader.treatInputAsWorldSpace = true;

            // 强制写入渲染参数（重要：renderAsSplatQuads）
            ApplySplatParamsToLoader(loader, 0);

            _chunkLoaders.Add(loader);
            _chunkEntries.Add(entry);
            _chunkVisibility.Add(true);
            _currentLOD.Add(0);    // 初始使用 L0
            _isLoaded.Add(true);   // loader.Start() 会加载数据

            created++;

            if (logChunkInfo)
            {
                Debug.Log($"[GaussianChunkManager] Created {chunkName} (L0={l0FileName}, count={entry.count})");
            }
        }

        // Apply current Inspector params to all loaders
        ApplySplatParamsToAllLoaders();

        Debug.Log($"[GaussianChunkManager] Created {created} Gaussian chunk loaders.");
    }

    /// <summary>
    /// 根据 lodLevel 和 entry 返回对应的 LOD 文件名（仅文件名，不含路径）。
    /// 如果对应 LOD 不存在，则回退到已有的更高精度 LOD。
    /// </summary>
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

        // 回退策略：优先用 L0，其次 entry.filename
        if (entry.lod != null && entry.lod.L0 != null && !string.IsNullOrEmpty(entry.lod.L0.filename))
            return entry.lod.L0.filename;

        return entry.filename;
    }

    private void Update()
    {
        // 1) Sync params only when Inspector values change
        if (SplatParamsChanged())
        {
            ApplySplatParamsToAllLoaders();
        }

        if (Camera.main == null || _lodIndex == null)
            return;

        // 2) 计算视锥体平面
        GeometryUtility.CalculateFrustumPlanes(Camera.main, _frustumPlanes);

        Vector3 camPos = Camera.main.transform.position;

        for (int i = 0; i < _chunkLoaders.Count; i++)
        {
            var loader = _chunkLoaders[i];
            var entry = _chunkEntries[i];
            if (loader == null || entry == null)
                continue;

            if (entry.bbox_min == null || entry.bbox_max == null)
                continue;

            Vector3 bmin = new Vector3(entry.bbox_min[0], entry.bbox_min[1], entry.bbox_min[2]);
            Vector3 bmax = new Vector3(entry.bbox_max[0], entry.bbox_max[1], entry.bbox_max[2]);

            // 2a) Frustum Culling
            bool visible = IsChunkVisible(_frustumPlanes, bmin, bmax);

            // loader.enabled 会影响 OnRenderObject 是否执行
            loader.enabled = visible && _isLoaded[i];

            if (logCullingChanges && _chunkVisibility[i] != visible)
            {
                _chunkVisibility[i] = visible;
                Debug.Log($"[ChunkCulling] {loader.gameObject.name} visible = {visible}");
            }

            // 3) 距离 LOD 切换 + Streaming
            Vector3 center;
            if (entry.center != null && entry.center.Length >= 3)
                center = new Vector3(entry.center[0], entry.center[1], entry.center[2]);
            else
                center = (bmin + bmax) * 0.5f;

            float dist = Vector3.Distance(camPos, center);

            // --- Streaming hysteresis guard ---
            if (unloadRadius < loadRadius)
                unloadRadius = loadRadius + 1f;

            bool shouldLoad = dist < loadRadius;
            bool shouldUnload = dist > unloadRadius;

            if (_isLoaded[i] && shouldUnload)
            {
                loader.UnloadData();
                _isLoaded[i] = false;
                loader.enabled = false;

                if (logStreamingChanges)
                    Debug.Log($"[Streaming] Unload {loader.gameObject.name} (dist={dist:F1}m)");

                // 已卸载则不再做 LOD 切换
                continue;
            }

            // desired LOD
            int desiredLOD = 0;
            if (dist > lod1To2) desiredLOD = 2;
            else if (dist > lod0To1) desiredLOD = 1;

            // If not loaded and should load -> load desired LOD
            if (!_isLoaded[i] && shouldLoad)
            {
                string lodFileToLoad = GetLODFileName(entry, desiredLOD);
                if (!string.IsNullOrEmpty(lodFileToLoad))
                {
                    loader.dataFileName = Path.Combine(chunksFolderName, lodFileToLoad);
                    ApplySplatParamsToLoader(loader, desiredLOD);

                    loader.ReloadData();
                    _isLoaded[i] = true;
                    _currentLOD[i] = desiredLOD;

                    if (logStreamingChanges)
                        Debug.Log($"[Streaming] Load {loader.gameObject.name} (LOD{desiredLOD}, dist={dist:F1}m)");
                }

                // 刚加载完，跳过一帧避免连续 Reload
                continue;
            }

            // If loaded -> LOD switch
            if (_isLoaded[i])
            {
                if (desiredLOD != _currentLOD[i])
                {
                    _currentLOD[i] = desiredLOD;

                    string lodFile = GetLODFileName(entry, desiredLOD);
                    if (!string.IsNullOrEmpty(lodFile))
                    {
                        loader.dataFileName = Path.Combine(chunksFolderName, lodFile);
                        ApplySplatParamsToLoader(loader, desiredLOD);

                        loader.ReloadData();

                        if (logLODChanges)
                            Debug.Log($"[LOD] {loader.gameObject.name} -> LOD{desiredLOD} (file={lodFile}, dist={dist:F1}m)");
                    }
                }
                else
                {
                    // 同 LOD：仍然确保参数一致（尤其是 renderAsSplatQuads）
                    ApplySplatParamsToLoader(loader, desiredLOD);
                }
            }
        }
    }
}
