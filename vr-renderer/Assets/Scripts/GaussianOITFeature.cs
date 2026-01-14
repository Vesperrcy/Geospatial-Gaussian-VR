using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering.RenderGraphModule;

public class GaussianOITFeature : ScriptableRendererFeature
{
    // 让“Gaussian 绘制”发生在 OIT pass 内：下一步我们会让 Loader/Manager 订阅这个事件
    public static event Action<CommandBuffer, Camera> OnDrawGaussians;
    // RenderGraph path uses RasterCommandBuffer (Unity 6.2)
    public static event Action<RasterCommandBuffer, Camera> OnDrawGaussiansRG;

    // Depth prepass: draw Gaussians into camera depth (self-depth for depth-aware blending)
    public static event Action<CommandBuffer, Camera> OnDrawGaussiansDepth;
    // RenderGraph path uses RasterCommandBuffer (Unity 6.2)
    public static event Action<RasterCommandBuffer, Camera> OnDrawGaussiansDepthRG;

    public enum DebugView { Final, AccumColor, AccumWeight }

    [Serializable]
    public class Settings
    {
        [Header("RT Allocation")]
        [Range(0.25f, 1.0f)] public float rtScale = 1.0f;

        [Header("Pass Events")]
        public RenderPassEvent accumulateEvent = RenderPassEvent.BeforeRenderingTransparents;
        public RenderPassEvent resolveEvent    = RenderPassEvent.AfterRenderingTransparents;

        [Header("Debug")]
        public DebugView debugView = DebugView.Final;
        public bool logOnce = false;
    }

    public Settings settings = new Settings();

    // RTHandles (URP 推荐)
    internal RTHandle _accumColor;
    internal RTHandle _accumWeight;

    [Header("Resolve")]
    [SerializeField] private Shader resolveShader;
    private Material _resolveMat;

    private static readonly int _AccumColorTexID  = Shader.PropertyToID("_AccumColorTex");
    private static readonly int _AccumWeightTexID = Shader.PropertyToID("_AccumWeightTex");
    private static readonly int _DebugModeID      = Shader.PropertyToID("_DebugMode");

    // RenderGraph resources (Unity 6 / URP RenderGraph path)
    internal TextureHandle _rgAccumColor;
    internal TextureHandle _rgAccumWeight;
    internal bool _rgValid;

    private DepthPrePass _depthPass;
    private AccumulatePass _accumPass;
    private ResolvePass _resolvePass;

    private bool _loggedOnce;

    public override void Create()
    {
        // Resolve material
        if (resolveShader == null)
            resolveShader = Shader.Find("Hidden/GaussianOITResolve");
        if (resolveShader != null)
            _resolveMat = CoreUtils.CreateEngineMaterial(resolveShader);

        _depthPass  = new DepthPrePass(this, settings);
        _accumPass  = new AccumulatePass(this, settings);
        _resolvePass = new ResolvePass(this, settings);

        // Depth prepass should happen before accumulate so SampleSceneDepth becomes meaningful
        _depthPass.renderPassEvent  = settings.accumulateEvent;
        _accumPass.renderPassEvent  = settings.accumulateEvent;
        _resolvePass.renderPassEvent = settings.resolveEvent;

        _loggedOnce = false;
        _rgValid = false;
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        // 调试/开发阶段：对 Game + SceneView 生效（Frame Debugger 在 Editor 里常用 SceneView 相机）
        var ct = renderingData.cameraData.cameraType;
        if (ct != CameraType.Game && ct != CameraType.SceneView)
            return;

        // 如果你愿意也可以限制只对主相机
        // if (renderingData.cameraData.camera != Camera.main) return;

        if (settings.logOnce && !_loggedOnce)
        {
            _loggedOnce = true;
            Debug.Log("[GaussianOITFeature] Passes enqueued (Accumulate + Resolve)");
        }

        // Set debug mode on resolve material per frame (for both RG and compatibility paths)
        if (_resolveMat != null)
            _resolveMat.SetInt(_DebugModeID, (int)settings.debugView);

        renderer.EnqueuePass(_depthPass);
        renderer.EnqueuePass(_accumPass);
        renderer.EnqueuePass(_resolvePass);
    }

    // ---------------- Pass 0: Depth Prepass ----------------
    class DepthPrePass : ScriptableRenderPass
    {
        private readonly GaussianOITFeature _feature;
        private readonly Settings _s;
        private static readonly ProfilingSampler _ps = new ProfilingSampler("Gaussian OIT DepthPrePass");

        public DepthPrePass(GaussianOITFeature feature, Settings s)
        {
            _feature = feature;
            _s = s;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var cameraData = frameData.Get<UniversalCameraData>();
            var resourceData = frameData.Get<UniversalResourceData>();

            using var builder = renderGraph.AddRasterRenderPass<PassData>("Gaussian OIT DepthPrePass", out var passData, _ps);
            passData.camera = cameraData.camera;

            // Write to active depth (self-depth). Do NOT touch color.
            builder.SetRenderAttachmentDepth(resourceData.activeDepthTexture, AccessFlags.Write);

            builder.AllowPassCulling(false);

            builder.SetRenderFunc((PassData data, RasterGraphContext ctx) =>
            {
                // Ensure depth starts as far for this pass when there are no opaques.
                // If there ARE opaques, this clear would wipe them; so only clear for camera types where
                // we assume Gaussian-only scenes. For now: clear always (simple, predictable).
                ctx.cmd.ClearRenderTarget(true, false, Color.clear);

                OnDrawGaussiansDepthRG?.Invoke(ctx.cmd, data.camera);
            });
        }

        private class PassData
        {
            public Camera camera;
        }

#pragma warning disable CS0672
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            var cam = renderingData.cameraData.camera;
            var cmd = CommandBufferPool.Get("GaussianOIT_DepthPrePass");

            using (new ProfilingScope(cmd, _ps))
            {
#pragma warning disable CS0618
                var depthHandle = renderingData.cameraData.renderer.cameraDepthTargetHandle;
#pragma warning restore CS0618

                // Bind depth only
                CoreUtils.SetRenderTarget(cmd, depthHandle, ClearFlag.Depth);

                // Clear depth so SampleSceneDepth isn't just "far" garbage when no opaque exists.
                // (This makes self-depth deterministic in Gaussian-only scenes.)
                cmd.ClearRenderTarget(true, false, Color.clear);

                OnDrawGaussiansDepth?.Invoke(cmd, cam);
            }

            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }
#pragma warning restore CS0672
    }

    // ---------------- Pass A: Accumulate ----------------
    class AccumulatePass : ScriptableRenderPass
    {
        private readonly GaussianOITFeature _feature;
        private readonly Settings _s;
        private static readonly ProfilingSampler _ps = new ProfilingSampler("Gaussian OIT Accumulate");

        public AccumulatePass(GaussianOITFeature feature, Settings s)
        {
            _feature = feature;
            _s = s;
        }

#pragma warning disable CS0672
        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            var camDesc = renderingData.cameraData.cameraTargetDescriptor;

            int w = Mathf.Max(1, Mathf.RoundToInt(camDesc.width * _s.rtScale));
            int h = Mathf.Max(1, Mathf.RoundToInt(camDesc.height * _s.rtScale));

            // Common descriptor setup (no depth, no MSAA)
            var desc = camDesc;
            desc.width = w;
            desc.height = h;
            desc.depthBufferBits = 0;
            desc.msaaSamples = 1;

            // AccumColor: HDR color (RGBA16F)
            desc.graphicsFormat = GraphicsFormat.R16G16B16A16_SFloat;
            RenderingUtils.ReAllocateHandleIfNeeded(
                ref _feature._accumColor,
                desc,
                FilterMode.Bilinear,
                TextureWrapMode.Clamp,
                name: "_Gaussian_AccumColor"
            );

            // AccumWeight: single channel weight (R16F)
            desc.graphicsFormat = GraphicsFormat.R16_SFloat;
            RenderingUtils.ReAllocateHandleIfNeeded(
                ref _feature._accumWeight,
                desc,
                FilterMode.Bilinear,
                TextureWrapMode.Clamp,
                name: "_Gaussian_AccumWeight"
            );
        }
#pragma warning restore CS0672

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            // Create / (re)allocate RG textures each frame. These are per-frame graph resources.
            var cameraData = frameData.Get<UniversalCameraData>();
            var camDesc = cameraData.cameraTargetDescriptor;

            int w = Mathf.Max(1, Mathf.RoundToInt(camDesc.width * _s.rtScale));
            int h = Mathf.Max(1, Mathf.RoundToInt(camDesc.height * _s.rtScale));

            // AccumColor (RGBA16F)
            var colorDesc = new TextureDesc(w, h)
            {
                name = "_Gaussian_AccumColor_RG",
                colorFormat = GraphicsFormat.R16G16B16A16_SFloat,
                depthBufferBits = DepthBits.None,
                msaaSamples = MSAASamples.None,
                clearBuffer = true,
                clearColor = Color.clear,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                enableRandomWrite = false
            };

            // AccumWeight (R16F)
            var weightDesc = new TextureDesc(w, h)
            {
                name = "_Gaussian_AccumWeight_RG",
                colorFormat = GraphicsFormat.R16_SFloat,
                depthBufferBits = DepthBits.None,
                msaaSamples = MSAASamples.None,
                clearBuffer = true,
                clearColor = Color.clear,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                enableRandomWrite = false
            };

            _feature._rgAccumColor = renderGraph.CreateTexture(colorDesc);
            _feature._rgAccumWeight = renderGraph.CreateTexture(weightDesc);
            _feature._rgValid = true;

            using var builder = renderGraph.AddRasterRenderPass<PassData>("Gaussian OIT Accumulate", out var passData, _ps);

            passData.feature = _feature;
            passData.camera = cameraData.camera;

            // MRT attachments
            builder.SetRenderAttachment(_feature._rgAccumColor, 0);
            builder.SetRenderAttachment(_feature._rgAccumWeight, 1);

            // Expose as global textures for later passes (Resolve shader samples these)
            builder.SetGlobalTextureAfterPass(_feature._rgAccumColor, _AccumColorTexID);
            builder.SetGlobalTextureAfterPass(_feature._rgAccumWeight, _AccumWeightTexID);

            // We will issue draws via CommandBuffer inside the render func.
            builder.AllowPassCulling(false);

            builder.SetRenderFunc((PassData data, RasterGraphContext ctx) =>
            {
                // RenderGraph attachments are already bound (MRT) and cleared via TextureDesc.clearBuffer.
                // NOTE(Unity 6.2): RasterGraphContext has no `resources` accessor here, so we skip exposing
                // the RG textures as global textures for now. We'll wire resolve using proper RG reads later.

                // Let Gaussians draw into MRT (RenderGraph / RasterCommandBuffer)
                OnDrawGaussiansRG?.Invoke(ctx.cmd, data.camera);
            });
        }

        private class PassData
        {
            public GaussianOITFeature feature;
            public Camera camera;
        }

#pragma warning disable CS0672
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            var cam = renderingData.cameraData.camera;
            var cmd = CommandBufferPool.Get("GaussianOIT_Accumulate");

            using (new ProfilingScope(cmd, _ps))
            {
                // MRT: color + weight
                var accumC = _feature._accumColor;
                var accumW = _feature._accumWeight;

                // 绑定 MRT
                cmd.SetRenderTarget(
                    new RenderTargetIdentifier[] { accumC.nameID, accumW.nameID },
                    accumC.nameID
                );

                // clear：color=0, weight=0
                cmd.ClearRenderTarget(clearDepth: false, clearColor: true, backgroundColor: Color.clear);

                // Expose to shaders (Resolve shader samples these)
                cmd.SetGlobalTexture(_AccumColorTexID, accumC);
                cmd.SetGlobalTexture(_AccumWeightTexID, accumW);

                // 让 Gaussian 在这里画（下一步把 Loader 的 draw 迁移进来）
                OnDrawGaussians?.Invoke(cmd, cam);
            }

            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }
#pragma warning restore CS0672
    }

    // ---------------- Pass B: Resolve ----------------
    class ResolvePass : ScriptableRenderPass
    {
        private readonly GaussianOITFeature _feature;
        private readonly Settings _s;
        private static readonly ProfilingSampler _ps = new ProfilingSampler("Gaussian OIT Resolve");

        public ResolvePass(GaussianOITFeature feature, Settings s)
        {
            _feature = feature;
            _s = s;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            // If resolve material isn't ready or we didn't allocate accum textures this frame, skip.
            if (_feature._resolveMat == null || !_feature._rgValid)
                return;

            var cameraData = frameData.Get<UniversalCameraData>();
            var resourceData = frameData.Get<UniversalResourceData>();

            using var builder = renderGraph.AddRasterRenderPass<PassData>("Gaussian OIT Resolve", out var passData, _ps);

            passData.feature = _feature;
            passData.debugMode = (int)_s.debugView;

            // Output to camera active color
            builder.SetRenderAttachment(resourceData.activeColorTexture, 0);

            // Declare we will read the global textures set after Accumulate pass
            builder.UseGlobalTexture(_AccumColorTexID, AccessFlags.Read);
            builder.UseGlobalTexture(_AccumWeightTexID, AccessFlags.Read);

            builder.AllowPassCulling(false);

            builder.SetRenderFunc((PassData data, RasterGraphContext ctx) =>
            {
                // IMPORTANT (Unity 6.2 RenderGraph): do NOT modify global state via ctx.cmd.
                // Set per-pass parameters on the material instance instead.
                data.feature._resolveMat.SetInt(_DebugModeID, data.debugMode);

                // Destination is already set via SetRenderAttachment.
                // Resolve shader samples _AccumColorTex/_AccumWeightTex globals.
                Blitter.BlitTexture(ctx.cmd, new Vector4(1, 1, 0, 0), data.feature._resolveMat, 0);
            });
        }

        private class PassData
        {
            public GaussianOITFeature feature;
            public int debugMode;
        }

#pragma warning disable CS0672
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            if (_feature._resolveMat == null)
                return;

            var cmd = CommandBufferPool.Get("GaussianOIT_Resolve");

            using (new ProfilingScope(cmd, _ps))
            {
                // Bind accum textures for the resolve shader
                cmd.SetGlobalTexture(_AccumColorTexID, _feature._accumColor);
                cmd.SetGlobalTexture(_AccumWeightTexID, _feature._accumWeight);
                cmd.SetGlobalInt(_DebugModeID, (int)_s.debugView);

                // Output to camera color
#pragma warning disable CS0618
                var camColor = renderingData.cameraData.renderer.cameraColorTargetHandle;
#pragma warning restore CS0618
                CoreUtils.SetRenderTarget(cmd, camColor, ClearFlag.None);

                // Fullscreen resolve. Shader samples globals.
                CoreUtils.DrawFullScreen(cmd, _feature._resolveMat);
            }

            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }
#pragma warning restore CS0672
    }
}