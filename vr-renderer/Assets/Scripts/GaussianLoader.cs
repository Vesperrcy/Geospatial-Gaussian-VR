using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

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
    public float gaussianSharpness = 8.0f;
    [Range(0f, 1f)] public float globalAlpha = 1.0f;

    [Header("Chunk options")]
    public bool treatInputAsWorldSpace = false;
    public Vector3 chunkCenterWorld = Vector3.zero;

    private ComputeBuffer _positionBuffer;
    private ComputeBuffer _colorBuffer;
    private ComputeBuffer _scaleBuffer;

    private int _numPoints;

    private Material _runtimeMaterial;

    void Start()
    {
        if (pointMaterial == null)
        {
            Debug.LogError("[GaussianLoader] pointMaterial is not assigned!");
            return;
        }

        _runtimeMaterial = new Material(pointMaterial);

        if (treatInputAsWorldSpace)
        {
            transform.position = chunkCenterWorld;
        }

        LoadData();
        SetupMaterial();
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
        List<Vector3> colors = new();
        List<float> scales = new();

        var lines = File.ReadAllLines(path);

        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;

            var t = line.Split((char[])null, System.StringSplitOptions.RemoveEmptyEntries);
            if (t.Length < 9)
                continue;

            float x = float.Parse(t[0], CultureInfo.InvariantCulture);
            float y = float.Parse(t[1], CultureInfo.InvariantCulture);
            float z = float.Parse(t[2], CultureInfo.InvariantCulture);

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

        if (_positionBuffer == null || _colorBuffer == null || _scaleBuffer == null)
        {
            Debug.LogError($"[GaussianLoader] Missing ComputeBuffer(s) for {dataFileName}. " +
                           $"pos={(_positionBuffer != null)} col={(_colorBuffer != null)} scale={(_scaleBuffer != null)}");
            return;
        }

        _runtimeMaterial.SetBuffer("_Positions", _positionBuffer);
        _runtimeMaterial.SetBuffer("_Colors", _colorBuffer);
        _runtimeMaterial.SetBuffer("_Scales", _scaleBuffer);

        _runtimeMaterial.SetFloat("_PointSize", pointSize);
        _runtimeMaterial.SetFloat("_Sharpness", gaussianSharpness);
        _runtimeMaterial.SetFloat("_Alpha", globalAlpha);
    }

    private void OnRenderObject()
    {
        if (_runtimeMaterial == null || _numPoints == 0 || _positionBuffer == null || _colorBuffer == null)
            return;

        if (renderAsSplatQuads && _scaleBuffer == null)
            return;

        _runtimeMaterial.SetMatrix("_LocalToWorld", transform.localToWorldMatrix);
        _runtimeMaterial.SetFloat("_PointSize", pointSize);
        _runtimeMaterial.SetFloat("_Sharpness", gaussianSharpness);
        _runtimeMaterial.SetFloat("_Alpha", globalAlpha);

        _runtimeMaterial.SetPass(0);

        if (renderAsSplatQuads)
        {
            int vertexCount = _numPoints * 6;
            Graphics.DrawProceduralNow(MeshTopology.Triangles, vertexCount);
        }
        else
        {
            Graphics.DrawProceduralNow(MeshTopology.Points, _numPoints);
        }
    }

    private void OnDestroy()
    {
        UnloadData();

        if (_runtimeMaterial != null)
        {
            Destroy(_runtimeMaterial);
            _runtimeMaterial = null;
        }
    }
}