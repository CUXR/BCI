using UnityEngine;

namespace BCI
{
    /// Runtime auto-bootstrap for BCICommandClient.
    /// If no BCICommandClient exists in the scene, this spawns one.
    public static class BCIBootstrap
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void EnsureClient()
        {
            if (Object.FindObjectOfType<BCICommandClient>() != null)
            {
                return;
            }

            GameObject go = new GameObject("BCIClient (auto)");
            go.AddComponent<BCICommandClient>();
            Object.DontDestroyOnLoad(go);
            Debug.Log("[BCIBootstrap] Spawned BCIClient (auto).");
        }
    }
}
