using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// 从 StreamingAssets/{chunksFolderName}/{lodIndexFileName} 读取 chunk + LOD 信息，
/// 为每个 chunk 创建一个带 GaussianLoader 的 GameObject（默认加载 L0）。
/// 支持：
/// - pointSize 全局控制
/// - Frustum Culling（视锥裁剪）
/// - 距离 LOD 切换（L0/L1/L2）
/// </summary>
public class GaussianChunkManager : MonoBehaviour
{
    [Header("Chunk Folder (in StreamingAssets)")]
    // 对应 vr-renderer/Assets/StreamingAssets/navvis_chunks_Block1
    public string chunksFolderName = "navvis_chunks_Block1";

    [Header("LOD index file name")]
    public string lodIndexFileName = "navvis_chunks_lod_index.json";

    [Header("Rendering")]
    public Material pointMaterial;      // 传给 GaussianLoader
    public float pointSize = 1.0f;      // 传给 GaussianLoader

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

    // 记录上一次应用到所有 ChunkLoader 的 pointSize，用于检测 Inspector 中的变化
    private float _lastAppliedPointSize = -1f;

    private void Start()
    {
        if (loadOnStart)
        {
            LoadLODIndex();
            InitChunks_L0_Only();
        }
    }

    /// <summary>
    /// 将当前 Manager 的 pointSize 同步到所有已创建的 GaussianLoader。
    /// 仅在 pointSize 有变化时调用即可。
    /// </summary>
    private void ApplyPointSizeToLoaders()
    {
        if (_chunkLoaders == null || _chunkLoaders.Count == 0)
            return;

        foreach (var loader in _chunkLoaders)
        {
            if (loader == null) continue;
            loader.pointSize = pointSize;
        }

        _lastAppliedPointSize = pointSize;
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

        // 创建新的 root，放在世界原点，不受任何父物体影响
        _chunkRoot = new GameObject("GaussianChunksRoot_Block1").transform;
        _chunkRoot.SetParent(null, false);
        _chunkRoot.position = Vector3.zero;
        _chunkRoot.rotation = Quaternion.identity;
        _chunkRoot.localScale = Vector3.one;

        int created = 0;
        string chunkDir = Path.Combine(Application.streamingAssetsPath, chunksFolderName);

        foreach (var entry in _lodIndex.chunks)
        {
            if (entry == null)
                continue;

            if (entry.count <= 0)
                continue;

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

            string relativePath = Path.Combine(chunksFolderName, l0FileName);
            loader.dataFileName = relativePath;
            loader.pointMaterial = pointMaterial;
            loader.pointSize = pointSize;
            loader.treatInputAsWorldSpace = true;

            _chunkLoaders.Add(loader);
            _chunkEntries.Add(entry);
            _chunkVisibility.Add(true);
            _currentLOD.Add(0);   // 初始使用 L0
            _isLoaded.Add(true);   // Start() 中已经加载了 L0 数据

            created++;

            if (logChunkInfo)
            {
                Debug.Log($"[GaussianChunkManager] Created {chunkName} (L0={l0FileName}, count={entry.count})");
            }
        }

        // 初始化完成后，把当前 Manager 的 pointSize 应用到所有 ChunkLoader
        ApplyPointSizeToLoaders();

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
        // 1) pointSize 同步
        if (!Mathf.Approximately(_lastAppliedPointSize, pointSize))
        {
            ApplyPointSizeToLoaders();
        }

        if (Camera.main == null || _lodIndex == null)
            return;

        // 2) 计算视锥体平面
        GeometryUtility.CalculateFrustumPlanes(Camera.main, _frustumPlanes);

        Vector3 camPos = Camera.main.transform.position;

        for (int i = 0; i < _chunkLoaders.Count; i++)
        {
            var loader = _chunkLoaders[i];
            var entry  = _chunkEntries[i];
            if (loader == null || entry == null)
                continue;

            if (entry.bbox_min == null || entry.bbox_max == null)
                continue;

            Vector3 bmin = new Vector3(entry.bbox_min[0], entry.bbox_min[1], entry.bbox_min[2]);
            Vector3 bmax = new Vector3(entry.bbox_max[0], entry.bbox_max[1], entry.bbox_max[2]);

            // 2a) Frustum Culling
            bool visible = IsChunkVisible(_frustumPlanes, bmin, bmax);
            loader.enabled = visible && _isLoaded[i];

            if (logCullingChanges && _chunkVisibility[i] != visible)
            {
                _chunkVisibility[i] = visible;
                Debug.Log($"[ChunkCulling] {loader.gameObject.name} visible = {visible}");
            }

            // (Removed: if (!visible) continue; // 不可见时就不必做 LOD 计算了)

            // 3) 距离 LOD 切换 + Streaming
            // 使用 chunk center，如果没有 center，就用 bbox 中点
            Vector3 center;
            if (entry.center != null && entry.center.Length >= 3)
                center = new Vector3(entry.center[0], entry.center[1], entry.center[2]);
            else
                center = (bmin + bmax) * 0.5f;

            float dist = Vector3.Distance(camPos, center);

            // --- Chunk Streaming（加载/卸载） ---
            // 为了避免频繁抖动，使用 loadRadius / unloadRadius 形成一个滞回区间。
            if (unloadRadius < loadRadius)
            {
                // 确保不会出现逻辑反转
                unloadRadius = loadRadius + 1f;
            }

            bool shouldLoad   = dist < loadRadius;
            bool shouldUnload = dist > unloadRadius;

            // 先根据距离决定是否需要加载/卸载
            if (_isLoaded[i] && shouldUnload)
            {
                loader.UnloadData();
                _isLoaded[i] = false;

                if (logStreamingChanges)
                {
                    Debug.Log($"[Streaming] Unload {loader.gameObject.name} (dist={dist:F1}m)");
                }

                // 已卸载则不再做 LOD 切换
                continue;
            }

            // 根据距离计算“期望 LOD”
            int desiredLOD = 0;
            if (dist > lod1To2)
                desiredLOD = 2;
            else if (dist > lod0To1)
                desiredLOD = 1;

            // 如果当前未加载且应加载，则按 desiredLOD 加载对应文件
            if (!_isLoaded[i] && shouldLoad)
            {
                string lodFileToLoad = GetLODFileName(entry, desiredLOD);
                if (!string.IsNullOrEmpty(lodFileToLoad))
                {
                    loader.dataFileName = Path.Combine(chunksFolderName, lodFileToLoad);
                    loader.ReloadData();

                    _isLoaded[i]  = true;
                    _currentLOD[i] = desiredLOD;

                    if (logStreamingChanges)
                    {
                        Debug.Log($"[Streaming] Load {loader.gameObject.name} (LOD{desiredLOD}, dist={dist:F1}m)");
                    }
                }

                // 刚加载完，后续 LOD 逻辑可以跳过一帧，避免多次 Reload
                continue;
            }

            // 如果已经加载，则只做 LOD 切换
            if (_isLoaded[i])
            {
                if (desiredLOD != _currentLOD[i])
                {
                    _currentLOD[i] = desiredLOD;

                    string lodFile = GetLODFileName(entry, desiredLOD);
                    if (!string.IsNullOrEmpty(lodFile))
                    {
                        loader.dataFileName = Path.Combine(chunksFolderName, lodFile);
                        loader.ReloadData();

                        if (logLODChanges)
                        {
                            Debug.Log($"[LOD] {loader.gameObject.name} -> LOD{desiredLOD} (file={lodFile}, dist={dist:F1}m)");
                        }
                    }
                }
            }
        }
    }
}