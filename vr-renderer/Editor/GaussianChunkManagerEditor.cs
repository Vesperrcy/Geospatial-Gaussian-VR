using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(GaussianChunkManager))]
public class GaussianChunkManagerEditor : Editor
{
    public override void OnInspectorGUI()
    {
        serializedObject.Update();

        SerializedProperty property = serializedObject.GetIterator();
        bool enterChildren = true;

        while (property.NextVisible(enterChildren))
        {
            using (new EditorGUI.DisabledScope(property.propertyPath == "m_Script"))
            {
                EditorGUILayout.PropertyField(property, includeChildren: true);
            }

            if (property.name == "chunksFolderName")
                DrawDatasetQuickSwitch();

            if (property.name == "xrKeepLevelViewOnRecenter")
                DrawRecenterTools();

            enterChildren = false;
        }

        serializedObject.ApplyModifiedProperties();
    }

    private void DrawDatasetQuickSwitch()
    {
        var manager = (GaussianChunkManager)target;
        SerializedProperty chunksFolderProp = serializedObject.FindProperty("chunksFolderName");
        SerializedProperty lodIndexFileProp = serializedObject.FindProperty("lodIndexFileName");
        SerializedProperty activeDatasetProp = serializedObject.FindProperty("activeDatasetMode");
        SerializedProperty activeSamplingProp = serializedObject.FindProperty("activeSamplingMode");
        SerializedProperty indoorDatasetProp = serializedObject.FindProperty("indoorDataset");
        SerializedProperty outdoorDatasetProp = serializedObject.FindProperty("outdoorDataset");

        string indoorLabel = indoorDatasetProp.FindPropertyRelative("displayName").stringValue;
        string indoorRandom = indoorDatasetProp.FindPropertyRelative("randomChunksFolderName").stringValue;
        string indoorUniform = indoorDatasetProp.FindPropertyRelative("uniformChunksFolderName").stringValue;
        string indoorLod = indoorDatasetProp.FindPropertyRelative("lodIndexFileName").stringValue;

        string outdoorLabel = outdoorDatasetProp.FindPropertyRelative("displayName").stringValue;
        string outdoorRandom = outdoorDatasetProp.FindPropertyRelative("randomChunksFolderName").stringValue;
        string outdoorUniform = outdoorDatasetProp.FindPropertyRelative("uniformChunksFolderName").stringValue;
        string outdoorLod = outdoorDatasetProp.FindPropertyRelative("lodIndexFileName").stringValue;

        EditorGUILayout.Space(2f);
        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
        EditorGUILayout.LabelField("Dataset Quick Switch", EditorStyles.boldLabel);
        EditorGUILayout.LabelField("Current Folder", chunksFolderProp.stringValue);

        using (new EditorGUILayout.HorizontalScope())
        {
            DrawSwitchButton(manager, chunksFolderProp, lodIndexFileProp, activeDatasetProp, activeSamplingProp,
                $"{indoorLabel} Random", indoorRandom, indoorLod, GaussianChunkManager.DatasetMode.Indoor, GaussianChunkManager.SamplingMode.Random);

            DrawSwitchButton(manager, chunksFolderProp, lodIndexFileProp, activeDatasetProp, activeSamplingProp,
                $"{indoorLabel} Uniform", indoorUniform, indoorLod, GaussianChunkManager.DatasetMode.Indoor, GaussianChunkManager.SamplingMode.Uniform);
        }

        using (new EditorGUILayout.HorizontalScope())
        {
            DrawSwitchButton(manager, chunksFolderProp, lodIndexFileProp, activeDatasetProp, activeSamplingProp,
                $"{outdoorLabel} Random", outdoorRandom, outdoorLod, GaussianChunkManager.DatasetMode.Outdoor, GaussianChunkManager.SamplingMode.Random);

            DrawSwitchButton(manager, chunksFolderProp, lodIndexFileProp, activeDatasetProp, activeSamplingProp,
                $"{outdoorLabel} Uniform", outdoorUniform, outdoorLod, GaussianChunkManager.DatasetMode.Outdoor, GaussianChunkManager.SamplingMode.Uniform);
        }

        if (Application.isPlaying && GUILayout.Button("Reload Current Dataset"))
            manager.ReloadCurrentDataset();

        EditorGUILayout.HelpBox("Edit Indoor/Outdoor random/uniform folder names above, then use these buttons to switch the active data source.", MessageType.Info);
        EditorGUILayout.EndVertical();
        EditorGUILayout.Space(2f);
    }

    private void DrawSwitchButton(
        GaussianChunkManager manager,
        SerializedProperty chunksFolderProp,
        SerializedProperty lodIndexFileProp,
        SerializedProperty activeDatasetProp,
        SerializedProperty activeSamplingProp,
        string label,
        string folder,
        string lodFile,
        GaussianChunkManager.DatasetMode datasetMode,
        GaussianChunkManager.SamplingMode samplingMode)
    {
        bool isCurrent = string.Equals(chunksFolderProp.stringValue, folder, System.StringComparison.OrdinalIgnoreCase)
                         && string.Equals(lodIndexFileProp.stringValue, lodFile, System.StringComparison.OrdinalIgnoreCase);

        using (new EditorGUI.DisabledScope(isCurrent))
        {
            if (GUILayout.Button($"{label}\n{folder}", GUILayout.Height(42f)))
            {
                bool recenterCamera = activeDatasetProp.enumValueIndex != (int)datasetMode;
                chunksFolderProp.stringValue = folder;
                lodIndexFileProp.stringValue = lodFile;
                activeDatasetProp.enumValueIndex = (int)datasetMode;
                activeSamplingProp.enumValueIndex = (int)samplingMode;
                serializedObject.ApplyModifiedProperties();

                if (Application.isPlaying)
                    manager.ReloadCurrentDataset(recenterCamera);
                else
                    EditorUtility.SetDirty(manager);
            }
        }
    }

    private void DrawRecenterTools()
    {
        var manager = (GaussianChunkManager)target;

        EditorGUILayout.Space(2f);
        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
        EditorGUILayout.LabelField("XR Recenter Tools", EditorStyles.boldLabel);

        using (new EditorGUI.DisabledScope(!Application.isPlaying))
        {
            if (GUILayout.Button("Recenter Now"))
                manager.RecenterToCurrentDataset();
        }

        EditorGUILayout.HelpBox("After changing XR standing height values during Play Mode, click Recenter Now to apply them immediately.", MessageType.Info);
        EditorGUILayout.EndVertical();
        EditorGUILayout.Space(2f);
    }
}
