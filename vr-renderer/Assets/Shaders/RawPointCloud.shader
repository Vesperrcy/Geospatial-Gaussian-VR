Shader "Unlit/RawPointCloud"
{
    Properties
    {
        _PointSize ("Point Size", Float) = 2.0
        _Brightness ("Brightness", Float) = 1.0
        _Tint ("Tint", Color) = (1,1,1,1)
    }

    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" "RenderPipeline"="UniversalPipeline" }

        Pass
        {
            Name "RawPointCloud"
            ZWrite Off
            ZTest LEqual
            Cull Off
            Blend SrcAlpha OneMinusSrcAlpha

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            StructuredBuffer<float3> _Positions;
            StructuredBuffer<float3> _Colors;

            float4x4 _LocalToWorld;

            CBUFFER_START(UnityPerMaterial)
                float _PointSize;
                float _Brightness;
                float4 _Tint;
            CBUFFER_END

            struct Attributes
            {
                uint vertexID : SV_VertexID;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float4 color : COLOR;
            };

            float2 CornerUV(uint corner)
            {
                if (corner == 0) return float2(-1, -1);
                if (corner == 1) return float2( 1, -1);
                if (corner == 2) return float2( 1,  1);
                if (corner == 3) return float2(-1, -1);
                if (corner == 4) return float2( 1,  1);
                return float2(-1,  1);
            }

            Varyings vert(Attributes input)
            {
                Varyings output = (Varyings)0;

                uint pointIndex = input.vertexID / 6;
                uint corner = input.vertexID - pointIndex * 6;

                float3 localPos = _Positions[pointIndex];
                float4 worldPos = mul(_LocalToWorld, float4(localPos, 1.0));
                float4 clipPos = TransformWorldToHClip(worldPos.xyz);

                float2 cornerUV = CornerUV(corner);
                float2 pixelOffset = cornerUV * max(_PointSize, 1.0);
                float2 ndcOffset = float2(
                    pixelOffset.x / max(_ScreenParams.x * 0.5, 1.0),
                    pixelOffset.y / max(_ScreenParams.y * 0.5, 1.0)
                );

                clipPos.xy += ndcOffset * clipPos.w;

                output.positionCS = clipPos;
                float3 rawColor = _Colors[pointIndex] * _Brightness;
                output.color = float4(rawColor, 1.0) * _Tint;
                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                return input.color;
            }
            ENDHLSL
        }
    }
}
