using UnityEngine;
using UnityEngine.InputSystem; // 新输入系统

public class FreeFlyCamera : MonoBehaviour
{
    public float moveSpeed = 5f;
    public float lookSpeed = 2f;

    private float yaw = 0f;
    private float pitch = 0f;

    void Update()
    {
        var keyboard = Keyboard.current;
        var mouse    = Mouse.current;
        if (keyboard == null || mouse == null) return;

        // 鼠标右键拖动视角
        if (mouse.rightButton.isPressed)
        {
            Vector2 delta = mouse.delta.ReadValue();
            yaw   += delta.x * lookSpeed * Time.deltaTime;
            pitch -= delta.y * lookSpeed * Time.deltaTime;
            transform.eulerAngles = new Vector3(pitch, yaw, 0f);
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