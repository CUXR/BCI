using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

[RequireComponent(typeof(CharacterController))]
public class VRPlayerController : MonoBehaviour
{
    [Header("Movement")]
    public float moveSpeed = 3f;
    public float turnSpeed = 90f;

    [Header("VR Rig")]
    [Tooltip("Child camera that represents the player head. Auto-found if left empty.")]
    public Camera headCamera;

    [Header("Physics")]
    public float gravity = -15f;

    private CharacterController cc;
    private float verticalVelocity;

    private InputDevice leftController;

    void Awake()
    {
        cc = GetComponent<CharacterController>();
        // Set before SpawnAboveGround.Awake() uses cc.skinWidth for clearance calculation.
        // 0.08 is ~25% of the capsule radius and prevents tunnelling on procedural MeshColliders.
        cc.skinWidth = 0.08f;
    }

    void Start()
    {

        if (!headCamera)
            headCamera = GetComponentInChildren<Camera>();

        // Position camera at eye height when running without a headset
        if (headCamera && !XRSettings.isDeviceActive)
            headCamera.transform.localPosition = new Vector3(0f, 1.7f, 0f);

        // Request floor-level tracking so HMD height is real-world height above floor
        var xrSubsystems = new List<XRInputSubsystem>();
        SubsystemManager.GetSubsystems(xrSubsystems);
        foreach (var s in xrSubsystems)
            s.TrySetTrackingOriginMode(TrackingOriginModeFlags.Floor);

        RefreshControllers();
        InputDevices.deviceConnected += _ => RefreshControllers();
    }

    void RefreshControllers()
    {
        var buf = new List<InputDevice>();

        InputDevices.GetDevicesWithCharacteristics(
            InputDeviceCharacteristics.Left | InputDeviceCharacteristics.Controller, buf);
        if (buf.Count > 0) leftController = buf[0];
    }

    void Update()
    {
        // Retry controller lookup each frame until found — XR devices can connect after Start().
        if (!leftController.isValid)
            RefreshControllers();

        HandleTurn();
        HandleMove();
    }

    void HandleTurn()
    {
        // In VR the HMD drives camera rotation directly — nothing to do here.
        if (XRSettings.isDeviceActive) return;

        // Editor: mouse X rotates the player rig for conventional first-person preview.
        float yaw = Input.GetAxis("Mouse X") * turnSpeed * Time.deltaTime;
        transform.Rotate(0f, yaw, 0f);
    }

    void HandleMove()
    {
        Vector2 input = Vector2.zero;

        if (leftController.isValid &&
            leftController.TryGetFeatureValue(CommonUsages.primary2DAxis, out Vector2 axis) &&
            axis.magnitude > 0.15f)
        {
            input = axis;
        }
        else
        {
            // WASD / arrow key fallback for editor testing
            input.x = Input.GetAxis("Horizontal");
            input.y = Input.GetAxis("Vertical");
        }

        // Project head direction onto horizontal plane so looking up/down doesn't affect speed
        Vector3 forward = headCamera.transform.forward;
        Vector3 right   = headCamera.transform.right;
        forward.y = 0f; forward.Normalize();
        right.y   = 0f; right.Normalize();

        Vector3 move = (forward * input.y + right * input.x) * moveSpeed;

        // Gravity
        if (cc.isGrounded && verticalVelocity < 0f)
            verticalVelocity = -2f;
        verticalVelocity += gravity * Time.deltaTime;
        move.y = verticalVelocity;

        cc.Move(move * Time.deltaTime);
    }
}
