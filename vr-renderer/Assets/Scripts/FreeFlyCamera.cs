using UnityEngine;
using UnityEngine.InputSystem; // 新输入系统

public class FreeFlyCamera : MonoBehaviour
{
    public float moveSpeed = 5f;

    [Tooltip("Mouse look sensitivity. Mouse.delta is already per-frame, so do NOT multiply by Time.deltaTime.")]
    public float lookSpeed = 0.15f;

    [Tooltip("Clamp vertical look to avoid flipping.")]
    public float pitchMin = -89f;
    public float pitchMax = 89f;

    [Tooltip("Lock and hide cursor while holding RMB for consistent mouse delta.")]
    public bool lockCursorOnRightMouse = true;

    private float yaw = 0f;
    private float pitch = 0f;
    private bool _wasRightPressed = false;

    private void Start()
    {
        var e = transform.eulerAngles;
        yaw = e.y;
        pitch = e.x;
    }

    void Update()
    {
        var keyboard = Keyboard.current;
        var mouse    = Mouse.current;
        if (keyboard == null || mouse == null) return;

        // 鼠标右键拖动视角
        bool rightPressed = mouse.rightButton.isPressed;

        // Optional cursor lock for consistent deltas
        if (lockCursorOnRightMouse)
        {
            if (rightPressed && !_wasRightPressed)
            {
                Cursor.lockState = CursorLockMode.Locked;
                Cursor.visible = false;
            }
            else if (!rightPressed && _wasRightPressed)
            {
                Cursor.lockState = CursorLockMode.None;
                Cursor.visible = true;
            }
        }
        _wasRightPressed = rightPressed;

        if (rightPressed)
        {
            Vector2 delta = mouse.delta.ReadValue();

            // IMPORTANT: Mouse.delta is already a per-frame delta in pixels.
            // Multiplying by Time.deltaTime makes it almost zero.
            yaw   += delta.x * lookSpeed;
            pitch -= delta.y * lookSpeed;

            pitch = Mathf.Clamp(pitch, pitchMin, pitchMax);
            transform.rotation = Quaternion.Euler(pitch, yaw, 0f);
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