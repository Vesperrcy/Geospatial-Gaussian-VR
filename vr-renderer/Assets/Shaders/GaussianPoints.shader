Shader "Unlit/GaussianPoints"
{
    Properties
    {
        // 全局控制因子：乘在每个点自己的 sx 上
        _PointSize ("Point Size", Float) = 0.1
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }

        Pass
        {
            ZWrite On
            ZTest LEqual
            Cull Off

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            // ===== GPU buffers =====
            StructuredBuffer<float3> _Positions;
            StructuredBuffer<float3> _Colors;
            StructuredBuffer<float>  _Scales;      // ★ 每个点的 sx（世界单位）

            float4x4 _LocalToWorld;
            float    _PointSize;                  // 全局缩放系数

            struct appdata
            {
                uint vertexID : SV_VertexID;
            };

            struct v2f
            {
                float4 pos   : SV_POSITION;
                float4 col   : COLOR;
                float  psize : PSIZE;             // 屏幕空间点半径
            };

            v2f vert (appdata v)
            {
                v2f o;

                // 读取局部坐标
                float3 localPos = _Positions[v.vertexID];

                // Local -> World (provided by C# as transform.localToWorldMatrix)
                float4 worldPos4 = mul(_LocalToWorld, float4(localPos, 1.0));

                // World -> Clip (avoid applying object-to-world twice)
                float4 clipPos = mul(UNITY_MATRIX_VP, worldPos4);
                o.pos = clipPos;

                // 颜色
                float3 c = _Colors[v.vertexID];
                o.col = float4(c, 1.0);

                // ===== 屏幕空间半径：结合 sx 和 距离 =====
                float3 worldPos = worldPos4.xyz;

                // 世界空间距离（相机到点）
                float dist = distance(_WorldSpaceCameraPos.xyz, worldPos);
                dist = max(dist, 0.01);          // 防止除以 0

                // 每个点自己的世界尺度 sx（来自 Python 估计）
                float sx = _Scales[v.vertexID];   // 单位：米（或你在 preprocessing 中的单位）

                // k：把“世界单位 * sx”映射到“大致像素”的比例
                // 可以先设成 20 左右，再配合 _PointSize 在 Inspector 里调
                float k = 100.0;

                // 近大远小：屏幕半径 ~ sx / dist
                float size = _PointSize * k * sx / dist;

                // 限制范围，避免点太大/太小
                size = clamp(size, 1.0, 40.0);

                // 写入 PSIZE，交给 rasterizer
                o.psize = size;
                // =====================================

                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // 暂时只返回颜色（还没做真正的高斯 alpha）
                return i.col;
            }

            ENDCG
        }
    }
}