using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

public class GaussianLoader : MonoBehaviour
{
    [Header("Data file (relative to StreamingAssets)")]
    // 例如： "SampleBlock1_gaussians_demo.txt"
    public string dataFileName = "SampleBlock1_gaussians_demo.txt";

    [Header("Rendering")]
    public Material pointMaterial;
    public float pointSize = 1.0f;

    [Header("Chunk options")]
    // 如果为 true：文件中的 xyz 当作世界坐标，需要减去 chunkCenterWorld 变成本地坐标
    // 如果为 false：xyz 直接当作本地坐标使用
    public bool treatInputAsWorldSpace = false;
    public Vector3 chunkCenterWorld = Vector3.zero;

    // === GPU buffers ===
    private ComputeBuffer _positionBuffer;
    private ComputeBuffer _colorBuffer;
    private ComputeBuffer _scaleBuffer;   // 每个点的 sx

    private int _numPoints;

    // 每个 Loader 自己持有一份材质实例，避免多个 Loader 共用同一个 Material 导致 SetBuffer 互相覆盖
    private Material _runtimeMaterial;

    void Start()
    {
        if (pointMaterial == null)
        {
            Debug.LogError("[GaussianLoader] pointMaterial is not assigned!");
            return;
        }

        // 为每个 Loader 克隆一个独立的材质实例，避免多个 Loader 共用同一个 Material
        _runtimeMaterial = new Material(pointMaterial);

        // 把物体放到 chunkCenterWorld 位置（只在 treatInputAsWorldSpace=true 时生效）
        if (treatInputAsWorldSpace)
        {
            transform.position = chunkCenterWorld;
        }

        LoadData();
        SetupMaterial();
    }

    /// <summary>
    /// 被 ChunkManager 调用，用于切换 LOD 时重新加载不同 txt。
    /// </summary>
    public void ReloadData()
    {
        _positionBuffer?.Release();
        _colorBuffer?.Release();
        _scaleBuffer?.Release();

        _positionBuffer = null;
        _colorBuffer = null;
        _scaleBuffer = null;
        _numPoints = 0;

        LoadData();
        SetupMaterial();
    }

    /// <summary>
    /// 被 ChunkManager 调用，用于 Chunk Streaming：远处卸载 chunk，释放 GPU 资源。
    /// </summary>
    public void UnloadData()
    {
        _positionBuffer?.Release();
        _colorBuffer?.Release();
        _scaleBuffer?.Release();

        _positionBuffer = null;
        _colorBuffer = null;
        _scaleBuffer = null;
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
        List<Vector3> colors    = new();
        List<float>   scales    = new();   // 存每个点的 sx

        var lines = File.ReadAllLines(path);
        int lineIndex = 0;

        foreach (var line in lines)
        {
            lineIndex++;
            if (string.IsNullOrWhiteSpace(line))
                continue;

            var t = line.Split((char[])null, System.StringSplitOptions.RemoveEmptyEntries);
            if (t.Length < 9)
            {
                // 行格式不对就跳过
                continue;
            }

            float x = float.Parse(t[0], CultureInfo.InvariantCulture);
            float y = float.Parse(t[1], CultureInfo.InvariantCulture);
            float z = float.Parse(t[2], CultureInfo.InvariantCulture);

            // sx sy sz：目前我们只用 sx（各向同性半径），sy/sz 暂时忽略
            float sx = float.Parse(t[3], CultureInfo.InvariantCulture);

            Vector3 pWorld = new Vector3(x, y, z);
            Vector3 pLocal = treatInputAsWorldSpace ? (pWorld - chunkCenterWorld) : pWorld;

            positions.Add(pLocal);
            scales.Add(sx);

            float r = float.Parse(t[6], CultureInfo.InvariantCulture);
            float g = float.Parse(t[7], CultureInfo.InvariantCulture);
            float b = float.Parse(t[8], CultureInfo.InvariantCulture);
            colors.Add(new Vector3(r, g, b));
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

        _scaleBuffer = new ComputeBuffer(_numPoints, sizeof(float));
        _scaleBuffer.SetData(scales);

        Debug.Log($"[GaussianLoader] Loaded {_numPoints} points.");
    }

    private void SetupMaterial()
    {
        if (_runtimeMaterial == null)
        {
            Debug.LogError("[GaussianLoader] _runtimeMaterial is null in SetupMaterial");
            return;
        }

        _runtimeMaterial.SetBuffer("_Positions", _positionBuffer);
        _runtimeMaterial.SetBuffer("_Colors",    _colorBuffer);
        _runtimeMaterial.SetFloat("_PointSize",  pointSize);

        if (_scaleBuffer != null)
        {
            _runtimeMaterial.SetBuffer("_Scales", _scaleBuffer);
        }
    }

    private void OnRenderObject()
    {
        if (_positionBuffer == null || _runtimeMaterial == null || _numPoints == 0)
            return;

        // 单文件 demo：GaussianRenderer 的 transform
        // chunk 模式：每个 Chunk_* GameObject 的 transform（position = chunk center 或世界原点）
        _runtimeMaterial.SetMatrix("_LocalToWorld", transform.localToWorldMatrix);
        _runtimeMaterial.SetFloat("_PointSize", pointSize);
        _runtimeMaterial.SetPass(0);

        Graphics.DrawProceduralNow(MeshTopology.Points, _numPoints);
    }

    private void OnDestroy()
    {
        _positionBuffer?.Release();
        _colorBuffer?.Release();
        _scaleBuffer?.Release();

        if (_runtimeMaterial != null)
        {
            Destroy(_runtimeMaterial);
            _runtimeMaterial = null;
        }
    }
}