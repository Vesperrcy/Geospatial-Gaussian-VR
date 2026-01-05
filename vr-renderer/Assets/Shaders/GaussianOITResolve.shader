Shader "Hidden/GaussianOITResolve"
{
    SubShader
    {
        Tags { "RenderPipeline"="UniversalPipeline" "Queue"="Transparent" }
        Pass
        {
            Name "GaussianOITResolve"
            ZWrite Off
            ZTest Always
            Cull Off
            Blend One Zero

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex Vert
            #pragma fragment Frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            // Accumulation textures (created & filled in Accumulate pass)
            TEXTURE2D_X(_AccumColorTex);
            SAMPLER(sampler_AccumColorTex);

            TEXTURE2D_X(_AccumWeightTex);
            SAMPLER(sampler_AccumWeightTex);

            // 0=Final, 1=Show AccumColor, 2=Show Weight
            int _DebugMode;

            struct Attributes
            {
                uint vertexID : SV_VertexID;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv         : TEXCOORD0;
            };

            Varyings Vert(Attributes v)
            {
                Varyings o;
                o.positionCS = GetFullScreenTriangleVertexPosition(v.vertexID);
                o.uv = GetFullScreenTriangleTexCoord(v.vertexID);
                return o;
            }

            float4 Frag(Varyings i) : SV_Target
            {
                float4 accum = SAMPLE_TEXTURE2D_X(_AccumColorTex, sampler_AccumColorTex, i.uv);
                float  w     = SAMPLE_TEXTURE2D_X(_AccumWeightTex, sampler_AccumWeightTex, i.uv).r;

                // Debug views
                if (_DebugMode == 1)
                {
                    // show raw accumulation (will look “milky/overbright” — that's expected)
                    return float4(accum.rgb, 1.0);
                }
                if (_DebugMode == 2)
                {
                    // show weight heatmap (brighter = denser overlap)
                    return float4(w.xxx, 1.0);
                }

                // Resolve
                float eps = 1e-6;
                float inv = rcp(max(w, eps));
                float3 rgb = accum.rgb * inv;

                // A reasonable alpha from weight (prevents “glow washout”)
                // You can also try: alpha = saturate(w);
                float alpha = 1.0 - exp(-w);

                return float4(rgb, alpha);
            }
            ENDHLSL
        }
    }
}