Shader "Unlit/GaussianSplatQuads"
{
    Properties
    {
        _PointSize ("Point Size (world scale)", Float) = 1.0
        _Sharpness ("Gaussian Sharpness", Float) = 8.0
        _Alpha ("Global Alpha", Range(0,1)) = 1.0
    }

    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }

        Pass
        {
            ZWrite Off
            ZTest LEqual
            Cull Off

            // Premultiplied alpha blending:
            // out.rgb is already multiplied by out.a
            Blend One OneMinusSrcAlpha

            CGPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            StructuredBuffer<float3> _Positions;
            StructuredBuffer<float3> _Colors;
            StructuredBuffer<float>  _Scales;     // per-point sx (world units)

            float4x4 _LocalToWorld;

            float _PointSize;   // global multiplier for sx
            float _Sharpness;   // controls falloff
            float _Alpha;       // global alpha multiplier

            struct appdata
            {
                uint vertexID : SV_VertexID;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float4 col : COLOR;
                float2 uv  : TEXCOORD0; // quad local coords in [-1,1]
            };

            // Map 6 vertices to a quad (two triangles)
            // 0: (-1,-1)  1:(+1,-1)  2:(+1,+1)
            // 3: (-1,-1)  4:(+1,+1)  5:(-1,+1)
            float2 CornerUV(uint corner)
            {
                if (corner == 0) return float2(-1, -1);
                if (corner == 1) return float2( 1, -1);
                if (corner == 2) return float2( 1,  1);
                if (corner == 3) return float2(-1, -1);
                if (corner == 4) return float2( 1,  1);
                return float2(-1,  1);
            }

            v2f vert(appdata v)
            {
                v2f o;

                uint pointIndex = v.vertexID / 6;
                uint corner     = v.vertexID - pointIndex * 6;

                float3 localPos = _Positions[pointIndex];
                float4 worldPos4 = mul(_LocalToWorld, float4(localPos, 1.0));

                // Camera basis in world space (from inverse view matrix)
                float3 camRight = normalize(UNITY_MATRIX_I_V[0].xyz);
                float3 camUp    = normalize(UNITY_MATRIX_I_V[1].xyz);

                float sx = _Scales[pointIndex];
                float radius = max(1e-6, sx * _PointSize);

                float2 uv = CornerUV(corner);
                o.uv = uv;

                float3 worldOffset = (camRight * uv.x + camUp * uv.y) * radius;
                float4 worldPosQuad = worldPos4 + float4(worldOffset, 0.0);

                // World -> Clip
                o.pos = mul(UNITY_MATRIX_VP, worldPosQuad);

                float3 c = _Colors[pointIndex];
                o.col = float4(c, 1.0);

                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                // Gaussian in quad local space: uv in [-1,1]
                float r2 = dot(i.uv, i.uv);

                // Optional hard cut to reduce overdraw
                if (r2 > 1.0) discard;

                // Gaussian falloff
                float a = exp(-r2 * _Sharpness) * _Alpha;

                // Premultiply RGB
                float3 rgb = i.col.rgb * a;

                return float4(rgb, a);
            }

            ENDCG
        }
    }
}