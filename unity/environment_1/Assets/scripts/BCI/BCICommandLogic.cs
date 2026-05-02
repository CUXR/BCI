using UnityEngine;

namespace BCI
{
    public static class BCICommandLogic
    {
        public static bool IsValidLabel(string label)
        {
            return label == "LEFT"
                || label == "RIGHT"
                || label == "FORWARD"
                || label == "BACKWARD";
        }

        public static bool TryParse(string json, out BCIMessage msg)
        {
            msg = default;
            if (string.IsNullOrEmpty(json)) return false;
            try
            {
                msg = JsonUtility.FromJson<BCIMessage>(json);
                return msg.label != null;
            }
            catch
            {
                return false;
            }
        }

        public static bool IsAccepted(BCIMessage msg, float threshold)
        {
            return IsValidLabel(msg.label) && msg.confidence >= threshold;
        }

        public static Vector3 ProjectHorizontal(Vector3 v)
        {
            v.y = 0f;
            float m = v.magnitude;
            if (m < 1e-5f) return Vector3.forward;
            return v / m;
        }

        /// Computes the yaw delta (degrees, around world up) and position delta
        /// (world space) to apply for a given command.
        ///
        /// headForward must already be horizontal+unit length (use ProjectHorizontal).
        /// Behavior:
        ///   FORWARD  -> 0 yaw, step +headForward * stepSize
        ///   BACKWARD -> 0 yaw, step -headForward * stepSize
        ///   LEFT     -> -90 yaw, step along NEW forward * stepSize
        ///   RIGHT    -> +90 yaw, step along NEW forward * stepSize
        /// Unknown labels yield zero deltas; callers should filter via IsValidLabel.
        public static void ComputeMovement(
            string label,
            Vector3 headForward,
            float stepSize,
            out Vector3 deltaPosition,
            out float deltaYawDegrees)
        {
            deltaYawDegrees = 0f;
            deltaPosition = Vector3.zero;

            switch (label)
            {
                case "FORWARD":
                    deltaPosition = headForward * stepSize;
                    break;
                case "BACKWARD":
                    deltaPosition = -headForward * stepSize;
                    break;
                case "LEFT":
                    deltaYawDegrees = -90f;
                    deltaPosition = Quaternion.Euler(0f, -90f, 0f) * headForward * stepSize;
                    break;
                case "RIGHT":
                    deltaYawDegrees = 90f;
                    deltaPosition = Quaternion.Euler(0f, 90f, 0f) * headForward * stepSize;
                    break;
            }
        }
    }
}
