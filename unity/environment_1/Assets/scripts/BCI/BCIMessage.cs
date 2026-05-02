using System;

namespace BCI
{
    /// One message frame from the BCI classifier server.
    /// Wire format: {"label": "LEFT", "confidence": 0.87, "ts": 1714000000.123}
    [Serializable]
    public struct BCIMessage
    {
        public string label;
        public float confidence;
        public double ts;
    }
}
