using System.Collections;
using UnityEngine;
using UnityEngine.Events;
using NativeWebSocket;

namespace BCI
{
    /// Connects to a BCI classifier WebSocket server and dispatches received
    /// LEFT/RIGHT/FORWARD/BACKWARD labels as discrete movement on the XR Origin.
    /// Also fires OnCommand for downstream listeners (HUDs, loggers, audio).
    public class BCICommandClient : MonoBehaviour
    {
        [Header("Connection")]
        [Tooltip("WebSocket URL of the BCI classifier server.")]
        [SerializeField] private string serverUrl = "ws://localhost:5000";

        [Tooltip("Seconds to wait before attempting to reconnect after a disconnect.")]
        [SerializeField] private float reconnectDelaySeconds = 2f;

        [Header("Movement")]
        [Tooltip("Transform to translate/rotate (the XR Origin). " +
                 "If null, this script will try to find an XROrigin in the scene at Start.")]
        [SerializeField] private Transform xrOrigin;

        [Tooltip("Transform whose forward defines head-forward (typically the main camera). " +
                 "If null, Camera.main is used.")]
        [SerializeField] private Transform headTransform;

        [Tooltip("Distance moved per accepted command (meters).")]
        [SerializeField] private float stepSize = 0.5f;

        [Tooltip("Minimum classifier confidence to act on a message. " +
                 "Set to 0 to act on every valid-label message.")]
        [Range(0f, 1f)]
        [SerializeField] private float confidenceThreshold = 0.6f;

        [Header("Events")]
        [Tooltip("Fires with the command label (\"LEFT\", \"RIGHT\", \"FORWARD\", \"BACKWARD\") " +
                 "for every accepted message, after movement has been applied.")]
        public UnityEvent<string> OnCommand;

        private WebSocket webSocket;
        private bool quitting;

        private async void Start()
        {
            ResolveTransforms();
            await Connect();
        }

        private void ResolveTransforms()
        {
            if (xrOrigin == null)
            {
                var origin = FindObjectOfType<Unity.XR.CoreUtils.XROrigin>();
                if (origin != null) xrOrigin = origin.transform;
                else Debug.LogWarning("[BCICommandClient] xrOrigin not assigned and no XROrigin found in scene.");
            }
            if (headTransform == null)
            {
                if (Camera.main != null) headTransform = Camera.main.transform;
                else Debug.LogWarning("[BCICommandClient] headTransform not assigned and Camera.main is null.");
            }
        }

        private async System.Threading.Tasks.Task Connect()
        {
            webSocket = new WebSocket(serverUrl);
            webSocket.OnOpen += () => Debug.Log($"[BCICommandClient] Connected to {serverUrl}");
            webSocket.OnError += err => Debug.LogWarning($"[BCICommandClient] Error: {err}");
            webSocket.OnClose += code =>
            {
                Debug.Log($"[BCICommandClient] Closed (code={code}). Reconnecting in {reconnectDelaySeconds}s.");
                if (!quitting) StartCoroutine(ReconnectAfterDelay());
            };
            webSocket.OnMessage += HandleMessageBytes;

            try { await webSocket.Connect(); }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[BCICommandClient] Connect threw: {e.Message}. Will retry.");
                if (!quitting) StartCoroutine(ReconnectAfterDelay());
            }
        }

        private IEnumerator ReconnectAfterDelay()
        {
            yield return new WaitForSeconds(reconnectDelaySeconds);
            if (!quitting) _ = Connect();
        }

        private void HandleMessageBytes(byte[] bytes)
        {
            string json = System.Text.Encoding.UTF8.GetString(bytes);
            if (!BCICommandLogic.TryParse(json, out BCIMessage msg))
            {
                Debug.LogWarning($"[BCICommandClient] Dropping malformed frame: {json}");
                return;
            }
            if (!BCICommandLogic.IsAccepted(msg, confidenceThreshold))
            {
                return;
            }
            ApplyCommand(msg.label);
        }

        private void ApplyCommand(string label)
        {
            if (xrOrigin != null && headTransform != null)
            {
                Vector3 headForward = BCICommandLogic.ProjectHorizontal(headTransform.forward);
                BCICommandLogic.ComputeMovement(label, headForward, stepSize,
                    out Vector3 dPos, out float dYaw);

                if (Mathf.Abs(dYaw) > 1e-4f)
                    xrOrigin.Rotate(Vector3.up, dYaw, Space.World);
                xrOrigin.position += dPos;
            }

            OnCommand?.Invoke(label);
        }

        private void Update()
        {
#if !UNITY_WEBGL || UNITY_EDITOR
            webSocket?.DispatchMessageQueue();
#endif
        }

        private async void OnApplicationQuit()
        {
            quitting = true;
            if (webSocket != null) await webSocket.Close();
        }

        private async void OnDestroy()
        {
            quitting = true;
            if (webSocket != null) await webSocket.Close();
        }
    }
}
