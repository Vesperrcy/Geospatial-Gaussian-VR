using UnityEngine;
using UnityEngine.InputSystem; // 新输入系统

public class FreeFlyCamera : MonoBehaviour
{
    public float moveSpeed = 5f;

    [Tooltip("Mouse look sensitivity. Mouse.delta is already per-frame, so do NOT multiply by Time.deltaTime.")]
    public float lookSpeed = 0.15f;

    [Tooltip("If true, camera rotation only updates while holding RMB.")]
    public bool lookOnlyWhileRMB = true;

    [Tooltip("Clamp vertical look to avoid flipping.")]
    public float pitchMin = -89f;
    public float pitchMax = 89f;

    [Tooltip("Lock and hide cursor while holding RMB for consistent mouse delta.")]
    public bool lockCursorOnRightMouse = true;

    [Tooltip("If the camera has a parent (XR rig / player), rotate localRotation instead of world rotation.")]
    public bool useLocalRotation = true;

    private float yaw = 0f;
    private float pitch = 0f;
    private bool _wasRightPressed = false;
    private int _ignoreMouseDeltaFrames = 0;

    private Quaternion _lastObservedRot;
    private const float ExternalRotEpsilonDeg = 0.05f;

    private void Start()
    {
        var e = (useLocalRotation ? transform.localEulerAngles : transform.eulerAngles);
        yaw = e.y;
        pitch = e.x;
        // Unity stores eulerAngles in [0,360). Normalize so clamping works as expected.
        if (pitch > 180f) pitch -= 360f;

        _lastObservedRot = useLocalRotation ? transform.localRotation : transform.rotation;
    }

    void Update()
    {
        var keyboard = Keyboard.current;
        var mouse    = Mouse.current;
        if (keyboard == null || mouse == null) return;

        // 鼠标右键拖动视角
        bool rightPressed = mouse.rightButton.isPressed;

        // Decide whether we should rotate this frame
        bool allowLook = lookOnlyWhileRMB ? rightPressed : true;

        // If we require RMB but it is not held, swallow any delta so we don't get a jump
        // when RMB is pressed later (accumulated delta on some platforms).
        if (!allowLook)
        {
            _ignoreMouseDeltaFrames = 0;
            _ = mouse.delta.ReadValue();

            // Keep yaw/pitch synced while not in look mode.
            // This prevents mismatch when some other script / parent / editor gizmo changes rotation.
            var e = (useLocalRotation ? transform.localEulerAngles : transform.eulerAngles);
            yaw = e.y;
            pitch = e.x;
            if (pitch > 180f) pitch -= 360f;

            _lastObservedRot = useLocalRotation ? transform.localRotation : transform.rotation;
        }

        // Optional cursor lock for consistent deltas
        if (lockCursorOnRightMouse)
        {
            if (allowLook && !_wasRightPressed)
            {
                // Enter mouse-look mode
                Cursor.lockState = CursorLockMode.Locked;
                Cursor.visible = false;

                // Re-sync yaw/pitch from current rotation (in case something else changed it)
                var e = (useLocalRotation ? transform.localEulerAngles : transform.eulerAngles);
                yaw = e.y;
                pitch = e.x;
                if (pitch > 180f) pitch -= 360f;

                _lastObservedRot = useLocalRotation ? transform.localRotation : transform.rotation;

                // Swallow the first deltas after locking to avoid a sudden jump
                _ignoreMouseDeltaFrames = 2;

                // Read once to clear any accumulated delta on this frame
                _ = mouse.delta.ReadValue();
            }
            else if (!allowLook && _wasRightPressed)
            {
                Cursor.lockState = CursorLockMode.None;
                Cursor.visible = true;
            }
        }
        _wasRightPressed = allowLook;

        if (allowLook)
        {
            // Swallow the first deltas after entering RMB look to prevent a jump.
            if (_ignoreMouseDeltaFrames > 0)
            {
                _ignoreMouseDeltaFrames--;
                _ = mouse.delta.ReadValue();
            }
            else
            {
                // If something else changed rotation while we're in look mode (e.g., parent rig,
                // another script, editor tools), resync yaw/pitch to what we actually see.
                var currentRot = useLocalRotation ? transform.localRotation : transform.rotation;
                if (Quaternion.Angle(currentRot, _lastObservedRot) > ExternalRotEpsilonDeg)
                {
                    var eNow = (useLocalRotation ? transform.localEulerAngles : transform.eulerAngles);
                    yaw = eNow.y;
                    pitch = eNow.x;
                    if (pitch > 180f) pitch -= 360f;

                    _lastObservedRot = currentRot;
                    _ignoreMouseDeltaFrames = 1;
                    _ = mouse.delta.ReadValue();
                    return;
                }

                Vector2 delta = mouse.delta.ReadValue();

                // IMPORTANT: Mouse.delta is already a per-frame delta in pixels.
                // Multiplying by Time.deltaTime makes it almost zero.
                yaw   += delta.x * lookSpeed;
                pitch -= delta.y * lookSpeed;

                pitch = Mathf.Clamp(pitch, pitchMin, pitchMax);
                var q = Quaternion.Euler(pitch, yaw, 0f);
                if (useLocalRotation)
                    transform.localRotation = q;
                else
                    transform.rotation = q;

                _lastObservedRot = useLocalRotation ? transform.localRotation : transform.rotation;
            }
        }

        // WASD 平面移动
        float h = 0f;
        float v = 0f;
        if (keyboard.aKey.isPressed) h -= 1f;
        if (keyboard.dKey.isPressed) h += 1f;
        if (keyboard.wKey.isPressed) v += 1f;
        if (keyboard.sKey.isPressed) v -= 1f;

        Vector3 move = transform.right * h + transform.forward * v;

        // Q/E 上下
        if (keyboard.qKey.isPressed) move -= transform.up;
        if (keyboard.eKey.isPressed) move += transform.up;

        transform.position += move * moveSpeed * Time.deltaTime;
    }
}