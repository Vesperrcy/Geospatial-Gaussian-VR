Shader "Unlit/GaussianSplatQuads"
{
    Properties
    {
        _Opacity ("Global Opacity", Range(0,1)) = 0.7
        _SigmaCutoff ("Sigma Cutoff (k)", Float) = 3.0
        _MinAxisPixels ("Min Footprint (px)", Float) = 0.75
        _MaxAxisPixels ("Max Footprint (px)", Float) = 64.0

        _PointSize ("Point Size Multiplier", Float) = 1.0

        _WeightExponent ("OIT Weight Exponent", Range(0,4)) = 1.0
        _DepthWeight ("Depth Weight", Range(0,2)) = 0.35
        _DepthExponent ("Depth Exponent", Range(0,4)) = 2.0

        _NearCompRefZ ("Near Comp Ref Z (m)", Float) = 6.0
        _NearCompStrength ("Near Comp Strength", Range(0,4)) = 1.0
        _NearCompMin ("Near Comp Min", Range(1,8)) = 1.0
        _NearCompMax ("Near Comp Max", Range(1,8)) = 3.0

        _DepthAwareScale ("Depth-aware Scale (1/m)", Float) = 20.0
        _DepthAwareBias ("Depth-aware Bias (m)", Float) = 0.03
        _DepthAwareStrength ("Depth-aware Strength", Range(0,8)) = 0.0
        _SurfaceThickness ("Surface Thickness (m)", Float) = 0.08
        _SurfaceMinAlpha  ("Surface Min Alpha", Range(0,0.5)) = 0.05

        // --- per-loader view-space Z fade (set by GaussianLoader MPB) ---
        _ViewZFadeEnabled ("ViewZ Fade Enabled", Float) = 0
        _ViewZFadeStart   ("ViewZ Fade Start (m)", Float) = 0
        _ViewZFadeEnd     ("ViewZ Fade End (m)", Float) = 10
        _ViewZFadeExponent("ViewZ Fade Exponent", Float) = 1
        _ViewZFadeInvert  ("ViewZ Fade Invert", Float) = 0

        // --- coarse-only anti-fog controls (coarse = _ViewZFadeInvert < 0.5) ---
        _CoarseNearKillStart ("Coarse Near Kill Start (m)", Float) = 0.0
        _CoarseNearKillEnd   ("Coarse Near Kill End (m)", Float) = 10.0
        _CoarseNearFootprintScale ("Coarse Near Footprint Scale", Range(0.05,1.0)) = 0.35
        _CoarseAlphaCap ("Coarse Alpha Cap", Range(0,1)) = 0.18

        // --- optional grazing-angle compensation (multiplies footprint only; keep small) ---
        _GrazeBoostMax ("Grazing Boost Max", Range(1,4)) = 1.0
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

            // OIT accumulation
            Blend 0 One One
            Blend 1 One One

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"

            StructuredBuffer<float3> _Positions;
            StructuredBuffer<float3> _Colors;
            StructuredBuffer<float4> _Cov0;
            StructuredBuffer<float4> _Cov1;

            float4x4 _LocalToWorld;

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

                float _DepthAwareScale;
                float _DepthAwareBias;
                float _DepthAwareStrength;
                float _SurfaceThickness;
                float _SurfaceMinAlpha;

                float _ViewZFadeEnabled;
                float _ViewZFadeStart;
                float _ViewZFadeEnd;
                float _ViewZFadeExponent;
                float _ViewZFadeInvert;

                float _CoarseNearKillStart;
                float _CoarseNearKillEnd;
                float _CoarseNearFootprintScale;
                float _CoarseAlphaCap;

                float _GrazeBoostMax;
            CBUFFER_END

            struct Attributes { uint vertexID : SV_VertexID; };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float4 color      : COLOR;
                float2 dpix       : TEXCOORD0; // pixel-space offset from splat center (interpolated)
                float4 qAndCut    : TEXCOORD1; // (q00, q01, q11, cutoff2)
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

            // Coarse pass is identified by invert==0 (fine uses invert==1)
            float CoarseFlag()
            {
                return (_ViewZFadeInvert < 0.5) ? 1.0 : 0.0;
            }

            float Smooth01(float z, float a, float b)
            {
                float denom = max(b - a, 1e-6);
                float t = saturate((z - a) / denom);
                return t * t * (3.0 - 2.0 * t);
            }

            // Stable inverse for 2x2 symmetric matrix [[a,b],[b,c]]
            void InverseSym2x2(float a, float b, float c, out float q00, out float q01, out float q11)
            {
                float det = a * c - b * b;
                det = max(det, 1e-20);
                float invDet = rcp(det);
                q00 =  c * invDet;
                q01 = -b * invDet;
                q11 =  a * invDet;
            }

            Varyings vert(Attributes v)
            {
                Varyings o = (Varyings)0;

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
                    o.dpix = 0;
                    o.qAndCut = 0;
                    return o;
                }

                // --- unpack covariance (local/object space) ---
                float4 c0 = _Cov0[pointIndex];
                float4 c1 = _Cov1[pointIndex];
                float xx = c0.x, xy = c0.y, xz = c0.z, yy = c0.w;
                float yz = c1.x, zz = c1.y;

                float3x3 SigmaLocal;
                SigmaLocal[0] = float3(xx, xy, xz);
                SigmaLocal[1] = float3(xy, yy, yz);
                SigmaLocal[2] = float3(xz, yz, zz);

                // Σ_world = A Σ_local A^T
                float3x3 A = (float3x3)_LocalToWorld;
                float3x3 SigmaWorld = mul(A, mul(SigmaLocal, transpose(A)));

                // Σ_view = Rcw Σ_world Rcw^T
                float3x3 V3 = (float3x3)GetWorldToViewMatrix();
                float3x3 SigmaView = mul(V3, mul(SigmaWorld, transpose(V3)));

                // --- Strict EWA: project 3D covariance to 2D using Jacobian of perspective ---
                // NDC projection: x_ndc = (P00 * x / z) + ... ; for small offsets we use Jacobian J (2x3)
                float4x4 P = GetViewToHClipMatrix();
                float fx = P[0][0];
                float fy = P[1][1];

                float invZ  = rcp(z);
                float invZ2 = invZ * invZ;

                float x = viewPos.x;
                float y = viewPos.y;

                // J_ndc rows (d ndc / d view)
                float3 Ju = float3(fx * invZ, 0.0, -fx * x * invZ2);
                float3 Jv = float3(0.0, fy * invZ, -fy * y * invZ2);

                // Σ_ndc = J Σ_view J^T => [[aN,bN],[bN,cN]] in ndc^2
                float aN = dot(Ju, mul(SigmaView, Ju));
                float bN = dot(Ju, mul(SigmaView, Jv));
                float cN = dot(Jv, mul(SigmaView, Jv));

                // Convert ndc covariance to pixel covariance: p = 0.5*Screen*(ndc+1)
                float2 screenPx = _ScreenParams.xy;
                float sx = 0.5 * screenPx.x;
                float sy = 0.5 * screenPx.y;

                // Σ_px = S Σ_ndc S, S=diag(sx,sy)
                float a = aN * (sx * sx);
                float b = bN * (sx * sy);
                float c = cN * (sy * sy);

                // --- Stabilize and enforce SPD ---
                a = max(a, 1e-12);
                c = max(c, 1e-12);
                float maxB = 0.999 * sqrt(a * c);
                b = clamp(b, -maxB, maxB);

                // --- Screen-space minimum footprint (pixel-domain) ---
                float minPx = max(_MinAxisPixels, 0.0);
                float minVar = minPx * minPx;
                a += minVar;
                c += minVar;

                // --- Optional grazing-angle compensation (kept conservative) ---
                // Use anisotropy ratio in screen space to decide a mild expansion.
                float trace = a + c;
                float detTerm = (a - c) * (a - c) + 4.0 * b * b;
                float s = sqrt(max(detTerm, 0.0));
                float l1 = max(0.5 * (trace + s), 1e-12);
                float l2 = max(0.5 * (trace - s), 1e-12);
                float ratio = sqrt(l1 / max(l2, 1e-12));
                float g = saturate((ratio - 2.0) / 6.0);
                float grazeBoost = lerp(1.0, max(_GrazeBoostMax, 1.0), g);

                // --- Global footprint scaling (must scale covariance, not just extent) ---
                float zSafe = max(z, 1e-3);
                float boost = _NearCompRefZ / zSafe;
                boost = pow(max(boost, 1e-3), _NearCompStrength);
                boost = clamp(boost, _NearCompMin, _NearCompMax);

                float scaleMul = max(_PointSize, 1e-3) * boost * grazeBoost;

                // Coarse-only near footprint suppression (prevents fog wall)
                float isCoarse = CoarseFlag();
                if (isCoarse > 0.5)
                {
                    float coarseZ = Smooth01(z, _CoarseNearKillStart, _CoarseNearKillEnd);
                    float footprintMul = lerp(_CoarseNearFootprintScale, 1.0, coarseZ);
                    scaleMul *= footprintMul;
                }

                // Σ_px *= scaleMul^2
                float s2 = scaleMul * scaleMul;
                a *= s2; b *= s2; c *= s2;

                // Clamp maximum footprint by clamping eigenvalues (in pixel units)
                // Desired semi-axis pixels: axis = k * sqrt(lambda)
                float k = max(0.5, _SigmaCutoff);
                float maxAxisPx = max(_MaxAxisPixels, 1.0);
                float maxLambda = (maxAxisPx / k);
                maxLambda = maxLambda * maxLambda;

                // Recompute eigenvalues after scaling
                trace = a + c;
                detTerm = (a - c) * (a - c) + 4.0 * b * b;
                s = sqrt(max(detTerm, 0.0));
                l1 = max(0.5 * (trace + s), 1e-12);
                l2 = max(0.5 * (trace - s), 1e-12);

                float cl = min(1.0, maxLambda / max(l1, 1e-12));
                // Apply same clamp to both eigenvalues to preserve orientation (cheap + stable)
                // (If you need exact eigen clamp, you'd rotate->scale->rotate back; this is a good engineering compromise.)
                a *= cl; b *= cl; c *= cl;

                // Final eigen for extent
                trace = a + c;
                detTerm = (a - c) * (a - c) + 4.0 * b * b;
                s = sqrt(max(detTerm, 0.0));
                l1 = max(0.5 * (trace + s), 1e-12);
                l2 = max(0.5 * (trace - s), 1e-12);

                // Eigenvectors (screen pixel domain)
                float2 e1 = (abs(b) > 1e-12) ? normalize(float2(b, l1 - a)) : float2(1.0, 0.0);
                float2 e2 = float2(-e1.y, e1.x);

                float axis1Px = k * sqrt(l1);
                float axis2Px = k * sqrt(l2);

                // Enforce min/max semi-axis in pixel units
                axis1Px = clamp(axis1Px, minPx, maxAxisPx);
                axis2Px = clamp(axis2Px, minPx, maxAxisPx);

                // Build quad in pixel space. CornerUV in [-1,1]
                float2 uv = CornerUV(corner);
                float2 dpix = e1 * (uv.x * axis1Px) + e2 * (uv.y * axis2Px);
                o.dpix = dpix;

                // Clip-space center + pixel offset -> NDC offset
                float4 clipCenter = mul(GetWorldToHClipMatrix(), worldPos);
                float2 ndcOffset = float2(dpix.x / sx, dpix.y / sy);
                clipCenter.xy += ndcOffset * clipCenter.w;
                o.positionCS = clipCenter;

                // Inverse covariance for EWA weight in pixel domain
                float q00, q01, q11;
                InverseSym2x2(a, b, c, q00, q01, q11);

                // Cutoff in Mahalanobis distance^2
                float cutoff2 = k * k;
                o.qAndCut = float4(q00, q01, q11, cutoff2);

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

                // EWA quadratic form in pixel domain: s = d^T Q d
                float2 d = i.dpix;
                float q00 = i.qAndCut.x;
                float q01 = i.qAndCut.y;
                float q11 = i.qAndCut.z;
                float cutoff2 = i.qAndCut.w;

                float s = (d.x * (q00 * d.x + q01 * d.y)) + (d.y * (q01 * d.x + q11 * d.y));

                // Hard cutoff at k-sigma
                if (s > cutoff2) discard;

                // EWA weight
                float alpha = exp(-0.5 * s) * _Opacity;

                // View-space Z fade (per-loader gating for scale mixture)
                if (_ViewZFadeEnabled > 0.5)
                {
                    float a = _ViewZFadeStart;
                    float b = _ViewZFadeEnd;
                    float denom = max(abs(b - a), 1e-6);

                    float t = saturate((i.viewZ - a) / denom);
                    float f = t * t * (3.0 - 2.0 * t);

                    float e = max(_ViewZFadeExponent, 0.25);
                    f = pow(max(f, 0.0), e);

                    if (_ViewZFadeInvert > 0.5)
                        f = 1.0 - f;

                    alpha *= saturate(f);
                }

                // Coarse-only near suppression + alpha cap (kills fog)
                float isCoarse = (_ViewZFadeInvert < 0.5) ? 1.0 : 0.0;
                if (isCoarse > 0.5)
                {
                    float coarseZ = Smooth01(i.viewZ, _CoarseNearKillStart, _CoarseNearKillEnd);
                    alpha *= coarseZ;
                    alpha = min(alpha, _CoarseAlphaCap);
                }

                // Optional surface-aware depth constraint (default strength=0 -> off)
                if (_DepthAwareStrength > 0.001)
                {
                    float2 screenUV = GetNormalizedScreenSpaceUV(i.positionCS);
                    float rawDepth = SampleSceneDepth(screenUV);

                    if (rawDepth < 0.999999)
                    {
                        float sceneZ = LinearEyeDepth(rawDepth, _ZBufferParams);
                        float dz = i.viewZ - sceneZ;

                        float behind = max(dz - _DepthAwareBias, 0.0);

                        float thickness = max(_SurfaceThickness, 1e-3);
                        float denom = max(thickness * thickness, 1e-6);

                        float surf = exp(-(behind * behind) / denom);
                        surf = max(surf, _SurfaceMinAlpha);

                        float sStrength = saturate(_DepthAwareStrength / 8.0);
                        alpha *= lerp(1.0, surf, sStrength);
                    }
                }

                if (alpha < 1e-4) discard;

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