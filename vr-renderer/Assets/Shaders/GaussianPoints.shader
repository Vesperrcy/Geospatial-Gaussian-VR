Shader "Unlit/GaussianPoints"
{
    Properties
    {
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

            StructuredBuffer<float3> _Positions;
            StructuredBuffer<float3> _Colors;
            float _PointSize;
            float4x4 _LocalToWorld;

            struct appdata { uint vertexID : SV_VertexID; };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float4 col : COLOR;
                float  psize : PSIZE;
            };

            v2f vert (appdata v)
            {
                v2f o;

                float3 localPos = _Positions[v.vertexID];
                float4 worldPos4 = mul(_LocalToWorld, float4(localPos, 1.0));

                float4 clipPos = UnityObjectToClipPos(worldPos4);
                o.pos = clipPos;

                float3 c = _Colors[v.vertexID];
                o.col = float4(c, 1.0);

                // ===== 这里开始改：用世界空间距离 =====
                float3 worldPos = worldPos4.xyz;
                float dist = distance(_WorldSpaceCameraPos.xyz, worldPos);
                dist = max(dist, 0.01);         // 防止除以 0

                float k = 50.0;
                float size = _PointSize * k / dist;

                size = clamp(size, 1.0, 40.0);
                o.psize = size;
                // =====================================

                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                return i.col;
            }
            ENDCG
        }
    }
}