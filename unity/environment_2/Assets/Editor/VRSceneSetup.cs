#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

public static class VRSceneSetup
{
    [MenuItem("Forest Simulator/Setup VR Player Rig")]
    static void SetupVRPlayer()
    {
        // Remove any previous rig
        var existing = GameObject.Find("XRPlayer");
        if (existing != null)
        {
            if (!EditorUtility.DisplayDialog("XRPlayer exists",
                    "An XRPlayer already exists in the scene. Replace it?", "Replace", "Cancel"))
                return;
            Undo.DestroyObjectImmediate(existing);
        }

        // --- Root: XRPlayer ---
        var player = new GameObject("XRPlayer");
        Undo.RegisterCreatedObjectUndo(player, "Setup VR Player Rig");
        player.transform.position = new Vector3(0f, 0f, 0f);

        var cc = player.AddComponent<CharacterController>();
        cc.height = 1.8f;
        cc.center = new Vector3(0f, 0.9f, 0f);
        cc.radius = 0.3f;
        cc.stepOffset = 0.3f;
        cc.skinWidth = 0.02f;

        var ctrl = player.AddComponent<VRPlayerController>();

        // --- Camera: re-parent Main Camera, or create one ---
        Camera mainCam = Camera.main;
        if (mainCam != null)
        {
            Undo.SetTransformParent(mainCam.transform, player.transform, "Parent camera to XRPlayer");
            mainCam.transform.localPosition = Vector3.zero;
            mainCam.transform.localRotation = Quaternion.identity;
            ctrl.headCamera = mainCam;
        }
        else
        {
            var camGO = new GameObject("Main Camera");
            camGO.tag = "MainCamera";
            Undo.RegisterCreatedObjectUndo(camGO, "Create Main Camera");
            camGO.transform.SetParent(player.transform, false);
            var cam = camGO.AddComponent<Camera>();
            camGO.AddComponent<AudioListener>();
            ctrl.headCamera = cam;
        }

        Selection.activeGameObject = player;

        Debug.Log(
            "[VRSceneSetup] XRPlayer rig created.\n\n" +
            "REQUIRED — complete these 3 steps in the Unity Editor before building:\n" +
            "  1. Edit > Project Settings > XR Plug-in Management\n" +
            "     • PC tab  : tick OpenXR  (for testing in editor / PC VR)\n" +
            "     • Android tab : tick OpenXR  (for Meta Quest build)\n" +
            "  2. Under XR Plug-in Management > OpenXR (Android tab):\n" +
            "     • Click the '+' under Interaction Profiles\n" +
            "     • Add 'Oculus Touch Controller Profile'\n" +
            "  3. File > Build Settings > Switch Platform to Android, then Build & Run.");
    }
}
#endif
