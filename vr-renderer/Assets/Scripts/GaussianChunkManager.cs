using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Stage D (Block1): 从 navvis_chunks_Block1/navvis_chunks_lod_index.json 读取 chunk+LOD 信息，
/// 为每个 chunk 创建一个 GaussianLoader（当前只加载 L0）。
/// </summary>
public class GaussianChunkManager : MonoBehaviour
{
    [Header("Chunk Folder (in StreamingAssets)")]
    public string chunksFolderName = "navvis_chunks_Block1";

    [Header("LOD index file name")]
    public string lodIndexFileName = "navvis_chunks_lod_index.json";

    [Header("Rendering")]
    public Material pointMaterial;      // GaussianPointsMat
    public float pointSize = 1.0f;      // 传给 GaussianLoader

    [Header("Debug")]
    public bool loadOnStart = true;

    // === JSON 映射结构（尽量简单） ===

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

    // === 内部状态 ===

    private LODIndex _lodIndex;
    private Transform _chunkRoot;
    private readonly List<GaussianLoader> _chunkLoaders = new();

    private void Start()
    {
        if (loadOnStart)
        {
            LoadLODIndex();
            InitChunks_L0_Only();
        }
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

        Debug.Log($"[GaussianChunkManager] Loaded LOD index: chunks={_lodIndex.chunks.Count}");
    }

    /// <summary>
    /// 为每个 chunk 创建一个 GameObject + GaussianLoader，加载 L0。
    /// </summary>
    public void InitChunks_L0_Only()
    {
        if (_lodIndex == null || _lodIndex.chunks == null)
        {
            Debug.LogError("[GaussianChunkManager] LOD index not loaded.");
            return;
        }

        // 清理旧节点
        if (_chunkRoot != null)
        {
            DestroyImmediate(_chunkRoot.gameObject);
            _chunkLoaders.Clear();
        }

        // 创建根节点，并在这里做一次 Z-up → Y-up 的旋转
        _chunkRoot = new GameObject("GaussianChunksRoot_Block1").transform;
        _chunkRoot.SetParent(transform, false);
        _chunkRoot.localPosition = Vector3.zero;
        _chunkRoot.localRotation = Quaternion.Euler(-90f, 0f, 0f);  // 和你之前 GaussianRenderer 一样
        _chunkRoot.localScale    = Vector3.one;

        int created = 0;

        foreach (var entry in _lodIndex.chunks)
        {
            if (entry == null) continue;

            string lod0File = null;
            if (entry.lod != null && entry.lod.L0 != null && !string.IsNullOrEmpty(entry.lod.L0.filename))
                lod0File = entry.lod.L0.filename;
            else
                lod0File = entry.filename; // fallback

            if (string.IsNullOrEmpty(lod0File))
            {
                Debug.LogWarning("[GaussianChunkManager] Chunk has no L0 filename, skip.");
                continue;
            }

            string chunkName = $"Chunk_{entry.ijk[0]}_{entry.ijk[1]}_{entry.ijk[2]}";
            GameObject go = new GameObject(chunkName);
            go.transform.SetParent(_chunkRoot, false);

            // 不再对 chunk 做额外平移，直接使用点云原始坐标
            go.transform.localPosition = Vector3.zero;
            go.transform.localRotation = Quaternion.identity;
            go.transform.localScale    = Vector3.one;

            var loader = go.AddComponent<GaussianLoader>();
            loader.dataFileName = Path.Combine(chunksFolderName, lod0File);  // 相对于 StreamingAssets
            loader.pointMaterial = pointMaterial;
            loader.pointSize = pointSize;

            _chunkLoaders.Add(loader);
            created++;
        }

        Debug.Log($"[GaussianChunkManager] InitChunks_L0_Only: created {created} chunk loaders.");
    }
}