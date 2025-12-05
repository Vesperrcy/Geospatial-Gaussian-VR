using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

public class GaussianLoader : MonoBehaviour
{
    [Header("Data file (relative to StreamingAssets)")]
    // 例如： "navvis_house2_gaussians_demo.txt"
    public string dataFileName = "SampleBlock1_gaussians_demo.txt";

    [Header("Rendering")]
    public Material pointMaterial;
    public float pointSize = 1.0f;

    [Header("Chunk options")]
    // 如果为 true：文件中的 xyz 当作世界坐标，需要减去 chunkCenterWorld 变成本地坐标
    // 如果为 false：xyz 直接当作本地坐标使用
    public bool treatInputAsWorldSpace = false;
    public Vector3 chunkCenterWorld = Vector3.zero;

    private ComputeBuffer _positionBuffer;
    private ComputeBuffer _colorBuffer;
    private int _numPoints;

    void Start()
    {
        //Application.targetFrameRate = 90;
        // 把物体放到 chunkCenterWorld 位置
        if (treatInputAsWorldSpace)
        {
            transform.position = chunkCenterWorld;
        }
        LoadData();
        SetupMaterial();
    }

    void LoadData()
    {
        Debug.Log($"[GaussianLoader] === LoadData START ===");
        Debug.Log($"[GaussianLoader] dataFileName = {dataFileName}");
        Debug.Log($"[GaussianLoader] treatInputAsWorldSpace = {treatInputAsWorldSpace}");
        Debug.Log($"[GaussianLoader] chunkCenterWorld = {chunkCenterWorld}");

        if (string.IsNullOrEmpty(dataFileName))
        {
            Debug.LogError("[GaussianLoader] dataFileName is empty.");
            return;
        }

        string path = Path.Combine(Application.streamingAssetsPath, dataFileName);
        Debug.Log($"[GaussianLoader] Loading data from: {path}");

        if (!File.Exists(path))
        {
            Debug.LogError("[GaussianLoader] File not found: " + path);
            return;
        }

        List<Vector3> positions = new();
        List<Vector3> colors = new();

        var lines = File.ReadAllLines(path);
        Debug.Log($"[GaussianLoader] File contains {lines.Length} lines.");

        int lineIndex = 0;
        foreach (var line in lines)
        {
            lineIndex++;
            if (string.IsNullOrWhiteSpace(line))
                continue;

            var t = line.Split((char[])null, System.StringSplitOptions.RemoveEmptyEntries);
            if (t.Length < 9)
            {
                Debug.LogWarning($"[GaussianLoader] Line {lineIndex} malformed: {line}");
                continue;
            }

            // ---- ① 输入原始坐标检查 ----
            float x = float.Parse(t[0], CultureInfo.InvariantCulture);
            float y = float.Parse(t[1], CultureInfo.InvariantCulture);
            float z = float.Parse(t[2], CultureInfo.InvariantCulture);

            if (lineIndex <= 3)  // 只打印前三行，避免刷屏
            {
                Debug.Log($"[GaussianLoader] Raw P[{lineIndex}] = ({x:F3}, {y:F3}, {z:F3})");
            }

            Vector3 pWorld = new Vector3(x, y, z);
            Vector3 pLocal;

            // ---- ② 坐标变换检查 ----
            if (treatInputAsWorldSpace)
            {
                pLocal = pWorld - chunkCenterWorld;

                if (lineIndex <= 3)
                {
                    Debug.Log($"[GaussianLoader] pLocal = pWorld - chunkCenterWorld = {pLocal}");
                }
            }
            else
            {
                pLocal = pWorld;

                if (lineIndex <= 3)
                {
                    Debug.Log($"[GaussianLoader] pLocal = pWorld (no transform) = {pLocal}");
                }
            }

            positions.Add(pLocal);

            float r = float.Parse(t[6], CultureInfo.InvariantCulture);
            float g = float.Parse(t[7], CultureInfo.InvariantCulture);
            float b = float.Parse(t[8], CultureInfo.InvariantCulture);
            colors.Add(new Vector3(r, g, b));
        }

        _numPoints = positions.Count;
        Debug.Log($"[GaussianLoader] Loaded {_numPoints} points.");
        Debug.Log($"[GaussianLoader] Sample pLocal[0] = {positions[0]}");

        if (_numPoints == 0)
        {
            Debug.LogError("[GaussianLoader] No valid points loaded.");
            return;
        }

        // ---- ③ 缓冲区创建检查 ----
        Debug.Log($"[GaussianLoader] Creating buffers: {_numPoints} points");

        _positionBuffer = new ComputeBuffer(_numPoints, sizeof(float) * 3);
        _colorBuffer = new ComputeBuffer(_numPoints, sizeof(float) * 3);

        _positionBuffer.SetData(positions);
        _colorBuffer.SetData(colors);

        Debug.Log($"[GaussianLoader] === LoadData DONE ===");
    }

    void SetupMaterial()
    {
        if (pointMaterial == null)
        {
            Debug.LogError("[GaussianLoader] pointMaterial is not assigned!");
            return;
        }

        pointMaterial.SetBuffer("_Positions", _positionBuffer);
        pointMaterial.SetBuffer("_Colors",   _colorBuffer);
        pointMaterial.SetFloat("_PointSize", pointSize);
    }

    void OnRenderObject()
    {
        if (_positionBuffer == null || pointMaterial == null || _numPoints == 0)
            return;

        // 这里的 Transform 就是：
        // 单文件 demo：GaussianRenderer 的 transform
        // chunk 模式：每个 Chunk_* GameObject 的 transform（position = chunk center）
        pointMaterial.SetMatrix("_LocalToWorld", transform.localToWorldMatrix);
        pointMaterial.SetFloat("_PointSize", pointSize);
        pointMaterial.SetPass(0);

        Graphics.DrawProceduralNow(MeshTopology.Points, _numPoints);
    }

    void OnDestroy()
    {
        _positionBuffer?.Release();
        _colorBuffer?.Release();
    }
}