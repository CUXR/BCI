using UnityEngine;

namespace BCI
{
    /// Runtime auto-bootstrap for BCICommandClient.
    /// On scene load (Play mode and built players), if no BCICommandClient is
    /// present in the scene, creates one with default Inspector values.
    /// The client's own Start() handles XR Origin and Camera.main lookup.
    public static class BCIBootstrap
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void EnsureClient()
        {
            if (Object.FindObjectOfType<BCICommandClient>() != null) return;

            var go = new GameObject("BCIClient (auto)");
            go.AddComponent<BCICommandClient>();
            Object.DontDestroyOnLoad(go);
            Debug.Log("[BCIBootstrap] Spawned BCIClient (auto). " +
                      "To customize, add a BCICommandClient manually in the scene.");
        }
    }
}
