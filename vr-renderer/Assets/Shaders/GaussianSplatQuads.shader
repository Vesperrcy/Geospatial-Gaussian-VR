Shader "Unlit/GaussianSplatQuads"
{
    Properties
    {
        _Opacity ("Global Opacity", Range(0,1)) = 0.7
        _SigmaCutoff ("Sigma Cutoff (k)", Float) = 3.0
        _MinAxisPixels ("Min Axis (px)", Float) = 0.75
        _MaxAxisPixels ("Max Axis (px)", Float) = 64.0
    }

    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" "RenderPipeline"="UniversalPipeline" }

        Pass
        {
            Name "GaussianSplat"
            Tags { "LightMode"="UniversalForward" }

            ZWrite Off
            ZTest LEqual
            Cull Off
            Blend One OneMinusSrcAlpha   // premultiplied alpha

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag

            // URP core include: provides UNITY_MATRIX_M / unity_MatrixVP / _ScreenParams / transform helpers
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            StructuredBuffer<float3> _Positions;
            StructuredBuffer<float3> _Colors;

            // cov0 = (xx, xy, xz, yy)
            // cov1 = (yz, zz, 0, 0)
            StructuredBuffer<float4> _Cov0;
            StructuredBuffer<float4> _Cov1;

            float4x4 _LocalToWorld;

            float _Opacity;
            float _SigmaCutoff;
            float _MinAxisPixels;
            float _MaxAxisPixels;

            struct Attributes
            {
                uint vertexID : SV_VertexID;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float4 color      : COLOR;
                float2 uv         : TEXCOORD0; // in [-1,1]
                float  k          : TEXCOORD1; // sigma cutoff
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

            Varyings vert(Attributes v)
            {
                Varyings o;

                uint pointIndex = v.vertexID / 6;
                uint corner     = v.vertexID - pointIndex * 6;

                float3 localPos = _Positions[pointIndex];
                float4 worldPos = mul(_LocalToWorld, float4(localPos, 1.0));

                // view space
                float3 viewPos = mul(GetWorldToViewMatrix(), worldPos).xyz;

                float z = -viewPos.z;
                if (z < 1e-4)
                {
                    o.positionCS = float4(0,0,0,0);
                    o.color = float4(0,0,0,0);
                    o.uv = 0;
                    o.k = _SigmaCutoff;
                    return o;
                }

                // covariance local (symmetric)
                float4 c0 = _Cov0[pointIndex];
                float4 c1 = _Cov1[pointIndex];
                float xx = c0.x;
                float xy = c0.y;
                float xz = c0.z;
                float yy = c0.w;
                float yz = c1.x;
                float zz = c1.y;

                float3x3 SigmaLocal;
                SigmaLocal[0] = float3(xx, xy, xz);
                SigmaLocal[1] = float3(xy, yy, yz);
                SigmaLocal[2] = float3(xz, yz, zz);

                // SigmaWorld = A * SigmaLocal * A^T
                float3x3 A = (float3x3)_LocalToWorld;
                float3x3 SigmaWorld = mul(A, mul(SigmaLocal, transpose(A)));

                // SigmaView = V * SigmaWorld * V^T
                float3x3 V3 = (float3x3)GetWorldToViewMatrix();
                float3x3 SigmaView = mul(V3, mul(SigmaWorld, transpose(V3)));

                // Projection Jacobian -> NDC
                float4x4 P = GetViewToHClipMatrix();
                float fx = P[0][0];
                float fy = P[1][1];

                float invZ = 1.0 / z;
                float invZ2 = invZ * invZ;

                float x = viewPos.x;
                float y = viewPos.y;

                float3 Ju = float3(fx * invZ, 0.0, -fx * x * invZ2);
                float3 Jv = float3(0.0, fy * invZ, -fy * y * invZ2);

                // Sigma2D = J * SigmaView * J^T (2x2)
                float a = dot(Ju, mul(SigmaView, Ju));
                float b = dot(Ju, mul(SigmaView, Jv));
                float c = dot(Jv, mul(SigmaView, Jv));

                a = max(a, 1e-12);
                c = max(c, 1e-12);

                // Eigen of [a b; b c]
                float trace = a + c;
                float detTerm = (a - c) * (a - c) + 4.0 * b * b;
                float s = sqrt(max(detTerm, 0.0));
                float l1 = 0.5 * (trace + s);
                float l2 = 0.5 * (trace - s);
                l1 = max(l1, 1e-12);
                l2 = max(l2, 1e-12);

                float2 e1;
                if (abs(b) > 1e-12) e1 = normalize(float2(b, l1 - a));
                else e1 = float2(1.0, 0.0);
                float2 e2 = float2(-e1.y, e1.x);

                float k = max(0.5, _SigmaCutoff);
                float axis1 = sqrt(l1) * k; // NDC
                float axis2 = sqrt(l2) * k;

                // clamp axes in pixel units
                float2 screen = _ScreenParams.xy;
                float px1 = axis1 * 0.5 * screen.x;
                float px2 = axis2 * 0.5 * screen.y;
                px1 = clamp(px1, _MinAxisPixels, _MaxAxisPixels);
                px2 = clamp(px2, _MinAxisPixels, _MaxAxisPixels);
                axis1 = (px1 * 2.0) / screen.x;
                axis2 = (px2 * 2.0) / screen.y;

                float2 uv = CornerUV(corner);
                o.uv = uv;
                o.k  = k;

                float2 ndcOffset = e1 * (uv.x * axis1) + e2 * (uv.y * axis2);

                float4 clipCenter = mul(GetWorldToHClipMatrix(), worldPos);
                float4 clipPos = clipCenter;
                clipPos.xy += ndcOffset * clipCenter.w;
                o.positionCS = clipPos;

                o.color = float4(_Colors[pointIndex], 1.0);
                return o;
            }

            float4 frag(Varyings i) : SV_Target
            {
                float r2 = dot(i.uv, i.uv);
                if (r2 > 1.0) discard;

                float q = (i.k * i.k) * r2;
                float a = exp(-0.5 * q) * _Opacity;

                // premultiplied
                return float4(i.color.rgb * a, a);
            }

            ENDHLSL
        }
    }
}