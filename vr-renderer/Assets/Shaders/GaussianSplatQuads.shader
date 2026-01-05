Shader "Unlit/GaussianSplatQuads"
{
    Properties
    {
        _Opacity ("Global Opacity", Range(0,1)) = 0.7
        _SigmaCutoff ("Sigma Cutoff (k)", Float) = 3.0
        _MinAxisPixels ("Min Axis (px)", Float) = 0.25
        _MaxAxisPixels ("Max Axis (px)", Float) = 64.0

        _PointSize ("Point Size Multiplier", Float) = 1.0

        _WeightExponent ("OIT Weight Exponent", Range(0,4)) = 1.0
        _DepthWeight ("Depth Weight", Range(0,2)) = 0.35
        _DepthExponent ("Depth Exponent", Range(0,4)) = 2.0

        _NearCompRefZ ("Near Comp Ref Z (m)", Float) = 6.0
        _NearCompStrength ("Near Comp Strength", Range(0,4)) = 1.0
        _NearCompMin ("Near Comp Min", Range(1,8)) = 1.0
        _NearCompMax ("Near Comp Max", Range(1,8)) = 3.0
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

            // OIT accumulation: RT0 add, RT1 add
            Blend 0 One One
            Blend 1 One One

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            StructuredBuffer<float3> _Positions;
            StructuredBuffer<float3> _Colors;

            StructuredBuffer<float4> _Cov0;
            StructuredBuffer<float4> _Cov1;

            float4x4 _LocalToWorld;

            // ✅ 关键：所有 Properties 对应的变量必须在 UnityPerMaterial 里
            CBUFFER_START(UnityPerMaterial)
                float _Opacity;
                float _SigmaCutoff;
                float _MinAxisPixels;
                float _MaxAxisPixels;

                float _PointSize;

                float _WeightExponent;
                float _DepthWeight;
                float _DepthExponent;

                float _NearCompRefZ;
                float _NearCompStrength;
                float _NearCompMin;
                float _NearCompMax;
            CBUFFER_END

            struct Attributes { uint vertexID : SV_VertexID; };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float4 color      : COLOR;
                float2 uv         : TEXCOORD0; // in [-1,1]
                float  k          : TEXCOORD1;
                float  viewZ      : TEXCOORD2;
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

                float3 viewPos = mul(GetWorldToViewMatrix(), worldPos).xyz;

                float z = -viewPos.z;
                o.viewZ = z;

                if (z < 1e-4)
                {
                    o.positionCS = float4(0,0,0,0);
                    o.color = float4(0,0,0,0);
                    o.uv = 0;
                    o.k = _SigmaCutoff;
                    return o;
                }

                float4 c0 = _Cov0[pointIndex];
                float4 c1 = _Cov1[pointIndex];
                float xx = c0.x, xy = c0.y, xz = c0.z, yy = c0.w;
                float yz = c1.x, zz = c1.y;

                float3x3 SigmaLocal;
                SigmaLocal[0] = float3(xx, xy, xz);
                SigmaLocal[1] = float3(xy, yy, yz);
                SigmaLocal[2] = float3(xz, yz, zz);

                float3x3 A = (float3x3)_LocalToWorld;
                float3x3 SigmaWorld = mul(A, mul(SigmaLocal, transpose(A)));

                float3x3 V3 = (float3x3)GetWorldToViewMatrix();
                float3x3 SigmaView = mul(V3, mul(SigmaWorld, transpose(V3)));

                float4x4 P = GetViewToHClipMatrix();
                float fx = P[0][0];
                float fy = P[1][1];

                float invZ  = 1.0 / z;
                float invZ2 = invZ * invZ;

                float x = viewPos.x;
                float y = viewPos.y;

                float3 Ju = float3(fx * invZ, 0.0, -fx * x * invZ2);
                float3 Jv = float3(0.0, fy * invZ, -fy * y * invZ2);

                float a = dot(Ju, mul(SigmaView, Ju));
                float b = dot(Ju, mul(SigmaView, Jv));
                float c = dot(Jv, mul(SigmaView, Jv));

                a = max(a, 1e-12);
                c = max(c, 1e-12);

                float trace   = a + c;
                float detTerm = (a - c) * (a - c) + 4.0 * b * b;
                float s = sqrt(max(detTerm, 0.0));
                float l1 = max(0.5 * (trace + s), 1e-12);
                float l2 = max(0.5 * (trace - s), 1e-12);

                float2 e1 = (abs(b) > 1e-12) ? normalize(float2(b, l1 - a)) : float2(1.0, 0.0);
                float2 e2 = float2(-e1.y, e1.x);

                float k = max(0.5, _SigmaCutoff);

                float axis1 = sqrt(l1) * k;
                float axis2 = sqrt(l2) * k;

                float2 screen = _ScreenParams.xy;
                float  minDim = min(screen.x, screen.y);

                float px1 = axis1 * 0.5 * minDim;
                float px2 = axis2 * 0.5 * minDim;

                px1 *= _PointSize;
                px2 *= _PointSize;

                // ✅ Near-field density compensation
                float zSafe = max(z, 1e-3);
                float boost = _NearCompRefZ / zSafe;
                boost = pow(max(boost, 1e-3), _NearCompStrength);
                boost = clamp(boost, _NearCompMin, _NearCompMax);

                px1 *= boost;
                px2 *= boost;

                px1 = clamp(px1, _MinAxisPixels, _MaxAxisPixels);
                px2 = clamp(px2, _MinAxisPixels, _MaxAxisPixels);

                axis1 = (px1 * 2.0) / minDim;
                axis2 = (px2 * 2.0) / minDim;

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

            struct FragmentOutput
            {
                float4 accumColor  : SV_Target0;
                float  accumWeight : SV_Target1;
            };

            FragmentOutput frag(Varyings i)
            {
                FragmentOutput o;

                float r2 = dot(i.uv, i.uv);
                if (r2 > 1.0) discard;

                float q = (i.k * i.k) * r2;
                float alpha = exp(-0.5 * q) * _Opacity;

                float depthW = 1.0 / (1.0 + _DepthWeight * pow(max(i.viewZ, 1e-3), _DepthExponent));
                float w = pow(saturate(alpha), _WeightExponent) * depthW;

                float3 premul = i.color.rgb * alpha;

                o.accumColor  = float4(premul * w, alpha * w);
                o.accumWeight = alpha * w;
                return o;
            }

            ENDHLSL
        }
    }
}