using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

public class GaussianLoader : MonoBehaviour
{
    public string dataFileName = "navvis_house2_gaussians_demo.txt";
    public Material pointMaterial;
    public float pointSize = 0.5f;

    private ComputeBuffer _positionBuffer;
    private ComputeBuffer _colorBuffer;
    private int _numPoints;

    void Start()
    {
        Application.targetFrameRate = 90;
        LoadData();
        SetupMaterial();
    }

    void LoadData()
    {
        string path = Path.Combine(Application.streamingAssetsPath, dataFileName);
        Debug.Log("[GaussianLoader] Loading data from: " + path);

        if (!File.Exists(path))
        {
            Debug.LogError("[GaussianLoader] File not found: " + path);
            return;
        }

        List<Vector3> positions = new();
        List<Vector3> colors = new();

        var lines = File.ReadAllLines(path);
        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;

            var t = line.Split((char[])null, System.StringSplitOptions.RemoveEmptyEntries);
            if (t.Length < 9) continue;

            float x = float.Parse(t[0], CultureInfo.InvariantCulture);
            float y = float.Parse(t[1], CultureInfo.InvariantCulture);
            float z = float.Parse(t[2], CultureInfo.InvariantCulture);

            float r = float.Parse(t[6], CultureInfo.InvariantCulture);
            float g = float.Parse(t[7], CultureInfo.InvariantCulture);
            float b = float.Parse(t[8], CultureInfo.InvariantCulture);

            positions.Add(new Vector3(x, y, z));
            colors.Add(new Vector3(r, g, b));
        }

        _numPoints = positions.Count;
        Debug.Log("[GaussianLoader] Loaded " + _numPoints + " points.");

        _positionBuffer = new ComputeBuffer(_numPoints, sizeof(float) * 3);
        _colorBuffer = new ComputeBuffer(_numPoints, sizeof(float) * 3);

        _positionBuffer.SetData(positions);
        _colorBuffer.SetData(colors);
    }

    void SetupMaterial()
    {
        pointMaterial.SetBuffer("_Positions", _positionBuffer);
        pointMaterial.SetBuffer("_Colors", _colorBuffer);
        pointMaterial.SetFloat("_PointSize", pointSize);
    }

    //void OnDrawGizmos()
    //{
    //    Gizmos.color = Color.yellow;
    //    Gizmos.DrawWireCube(transform.position, new Vector3(200, 200, 200));
    //}

    void OnRenderObject()
    {
        if (_positionBuffer == null) return;

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