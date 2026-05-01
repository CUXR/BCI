using UnityEngine;

// Attach to XRPlayer. Runs in Awake so the player is above ground before any
// other script (VRPlayerController, XR tracking) reads or modifies the position.
[DisallowMultipleComponent]
public class SpawnAboveGround : MonoBehaviour
{
    [Tooltip("Extra clearance above the detected surface (metres). 0 = feet exactly on ground.")]
    public float heightOffset = 0f;

    void Awake()
    {
        // Cast from well above the current XZ position down through the whole scene.
        Vector3 origin = new Vector3(transform.position.x, transform.position.y + 500f, transform.position.z);
        if (!Physics.Raycast(origin, Vector3.down, out RaycastHit hit, 1000f))
        {
            Debug.LogWarning("[SpawnAboveGround] No ground found below player — position unchanged.");
            return;
        }

        var cc = GetComponent<CharacterController>();

        // Place feet (transform pivot) at the surface plus any extra offset.
        // Add 2× skinWidth so the CC collision hull starts clearly above the collider surface —
        // without this the CC resolves the initial overlap by pushing the player downward.
        float skinClearance = cc ? cc.skinWidth * 2f : 0.05f;
        float targetY = hit.point.y + heightOffset + skinClearance;

        if (cc) cc.enabled = false;
        transform.position = new Vector3(transform.position.x, targetY, transform.position.z);
        if (cc) cc.enabled = true;
    }
}
