using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering.RenderGraphModule;

public class GaussianOITFeature : ScriptableRendererFeature
{
    public static event Action<CommandBuffer, Camera> OnDrawGaussians;
    public static event Action<RasterCommandBuffer, Camera> OnDrawGaussiansRG;
    public static event Action<CommandBuffer, Camera> OnDrawGaussiansDepth;
    public static event Action<RasterCommandBuffer, Camera> OnDrawGaussiansDepthRG;
    public static event Action<CommandBuffer, Camera> OnDrawDirect;
    public static event Action<RasterCommandBuffer, Camera> OnDrawDirectRG;

    public enum DebugView { Final, AccumColor, AccumWeight }

    [Serializable]
    public class Settings
    {
        [Header("RT Allocation")]
        [Range(0.25f, 1.0f)] public float rtScale = 1.0f;

        [Header("Pass Events")]
        public RenderPassEvent accumulateEvent = RenderPassEvent.BeforeRenderingTransparents;
        public RenderPassEvent resolveEvent = RenderPassEvent.AfterRenderingTransparents;

        [Header("Debug")]
        public DebugView debugView = DebugView.Final;
        public bool logOnce = false;
    }

    public Settings settings = new Settings();

    internal RTHandle _accumColor;
    internal RTHandle _accumWeight;

    [Header("Resolve")]
    [SerializeField] private Shader resolveShader;
    private Material _resolveMat;

    private static readonly int _AccumColorTexID = Shader.PropertyToID("_AccumColorTex");
    private static readonly int _AccumWeightTexID = Shader.PropertyToID("_AccumWeightTex");
    private static readonly int _DebugModeID = Shader.PropertyToID("_DebugMode");

    internal TextureHandle _rgAccumColor;
    internal TextureHandle _rgAccumWeight;
    internal bool _rgValid;

    private DepthPrePass _depthPass;
    private DirectDrawPass _directPass;
    private AccumulatePass _accumPass;
    private ResolvePass _resolvePass;

    private bool _loggedOnce;

    public override void Create()
    {
        if (resolveShader == null)
            resolveShader = Shader.Find("Hidden/GaussianOITResolve");
        if (resolveShader != null)
            _resolveMat = CoreUtils.CreateEngineMaterial(resolveShader);

        _depthPass = new DepthPrePass(this, settings);
        _directPass = new DirectDrawPass(this, settings);
        _accumPass = new AccumulatePass(this, settings);
        _resolvePass = new ResolvePass(this, settings);

        _depthPass.renderPassEvent = settings.accumulateEvent;
        _accumPass.renderPassEvent = settings.accumulateEvent;
        _resolvePass.renderPassEvent = settings.resolveEvent;
        _directPass.renderPassEvent = settings.resolveEvent;

        _loggedOnce = false;
        _rgValid = false;
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        var ct = renderingData.cameraData.cameraType;
        if (ct != CameraType.Game && ct != CameraType.SceneView)
            return;

        if (settings.logOnce && !_loggedOnce)
        {
            _loggedOnce = true;
            Debug.Log("[GaussianOITFeature] Passes enqueued (Accumulate + Resolve + Direct)");
        }

        if (_resolveMat != null)
            _resolveMat.SetInt(_DebugModeID, (int)settings.debugView);

        renderer.EnqueuePass(_depthPass);
        renderer.EnqueuePass(_accumPass);
        renderer.EnqueuePass(_resolvePass);
        renderer.EnqueuePass(_directPass);
    }

    class DirectDrawPass : ScriptableRenderPass
    {
        private readonly GaussianOITFeature _feature;
        private static readonly ProfilingSampler _ps = new ProfilingSampler("Gaussian Direct Draw");

        public DirectDrawPass(GaussianOITFeature feature, Settings s)
        {
            _feature = feature;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var cameraData = frameData.Get<UniversalCameraData>();
            var resourceData = frameData.Get<UniversalResourceData>();

            using var builder = renderGraph.AddRasterRenderPass<PassData>("Gaussian Direct Draw", out var passData, _ps);
            passData.camera = cameraData.camera;
            builder.SetRenderAttachment(resourceData.activeColorTexture, 0);
            builder.SetRenderAttachmentDepth(resourceData.activeDepthTexture, AccessFlags.ReadWrite);
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, RasterGraphContext ctx) =>
            {
                OnDrawDirectRG?.Invoke(ctx.cmd, data.camera);
            });
        }

        private class PassData { public Camera camera; }

#pragma warning disable CS0672
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            var cam = renderingData.cameraData.camera;
            var cmd = CommandBufferPool.Get("Gaussian_DirectDraw");
            using (new ProfilingScope(cmd, _ps))
            {
#pragma warning disable CS0618
                var camColor = renderingData.cameraData.renderer.cameraColorTargetHandle;
                var camDepth = renderingData.cameraData.renderer.cameraDepthTargetHandle;
#pragma warning restore CS0618
                CoreUtils.SetRenderTarget(cmd, camColor, camDepth, ClearFlag.None);
                OnDrawDirect?.Invoke(cmd, cam);
            }
            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }
#pragma warning restore CS0672
    }

    class DepthPrePass : ScriptableRenderPass
    {
        private static readonly ProfilingSampler _ps = new ProfilingSampler("Gaussian OIT DepthPrePass");
        public DepthPrePass(GaussianOITFeature feature, Settings s) {}

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var cameraData = frameData.Get<UniversalCameraData>();
            var resourceData = frameData.Get<UniversalResourceData>();

            using var builder = renderGraph.AddRasterRenderPass<PassData>("Gaussian OIT DepthPrePass", out var passData, _ps);
            passData.camera = cameraData.camera;
            builder.SetRenderAttachmentDepth(resourceData.activeDepthTexture, AccessFlags.Write);
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, RasterGraphContext ctx) =>
            {
                ctx.cmd.ClearRenderTarget(true, false, Color.clear);
                OnDrawGaussiansDepthRG?.Invoke(ctx.cmd, data.camera);
            });
        }

        private class PassData { public Camera camera; }

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
                CoreUtils.SetRenderTarget(cmd, depthHandle, ClearFlag.Depth);
                cmd.ClearRenderTarget(true, false, Color.clear);
                OnDrawGaussiansDepth?.Invoke(cmd, cam);
            }
            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }
#pragma warning restore CS0672
    }

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

            var desc = camDesc;
            desc.width = w;
            desc.height = h;
            desc.depthBufferBits = 0;
            desc.msaaSamples = 1;

            desc.graphicsFormat = GraphicsFormat.R16G16B16A16_SFloat;
            RenderingUtils.ReAllocateHandleIfNeeded(ref _feature._accumColor, desc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_Gaussian_AccumColor");

            desc.graphicsFormat = GraphicsFormat.R16_SFloat;
            RenderingUtils.ReAllocateHandleIfNeeded(ref _feature._accumWeight, desc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_Gaussian_AccumWeight");
        }
#pragma warning restore CS0672

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var cameraData = frameData.Get<UniversalCameraData>();
            var camDesc = cameraData.cameraTargetDescriptor;
            int w = Mathf.Max(1, Mathf.RoundToInt(camDesc.width * _s.rtScale));
            int h = Mathf.Max(1, Mathf.RoundToInt(camDesc.height * _s.rtScale));

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
            passData.camera = cameraData.camera;
            builder.SetRenderAttachment(_feature._rgAccumColor, 0);
            builder.SetRenderAttachment(_feature._rgAccumWeight, 1);
            builder.SetGlobalTextureAfterPass(_feature._rgAccumColor, _AccumColorTexID);
            builder.SetGlobalTextureAfterPass(_feature._rgAccumWeight, _AccumWeightTexID);
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, RasterGraphContext ctx) =>
            {
                OnDrawGaussiansRG?.Invoke(ctx.cmd, data.camera);
            });
        }

        private class PassData { public Camera camera; }

#pragma warning disable CS0672
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            var cam = renderingData.cameraData.camera;
            var cmd = CommandBufferPool.Get("GaussianOIT_Accumulate");
            using (new ProfilingScope(cmd, _ps))
            {
                var accumC = _feature._accumColor;
                var accumW = _feature._accumWeight;
                cmd.SetRenderTarget(new RenderTargetIdentifier[] { accumC.nameID, accumW.nameID }, accumC.nameID);
                cmd.ClearRenderTarget(false, true, Color.clear);
                cmd.SetGlobalTexture(_AccumColorTexID, accumC);
                cmd.SetGlobalTexture(_AccumWeightTexID, accumW);
                OnDrawGaussians?.Invoke(cmd, cam);
            }
            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }
#pragma warning restore CS0672
    }

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
            if (_feature._resolveMat == null || !_feature._rgValid)
                return;

            var resourceData = frameData.Get<UniversalResourceData>();
            using var builder = renderGraph.AddRasterRenderPass<PassData>("Gaussian OIT Resolve", out var passData, _ps);
            passData.feature = _feature;
            passData.debugMode = (int)_s.debugView;
            builder.SetRenderAttachment(resourceData.activeColorTexture, 0);
            builder.UseGlobalTexture(_AccumColorTexID, AccessFlags.Read);
            builder.UseGlobalTexture(_AccumWeightTexID, AccessFlags.Read);
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, RasterGraphContext ctx) =>
            {
                data.feature._resolveMat.SetInt(_DebugModeID, data.debugMode);
                Blitter.BlitTexture(ctx.cmd, new Vector4(1, 1, 0, 0), data.feature._resolveMat, 0);
            });
        }

        private class PassData { public GaussianOITFeature feature; public int debugMode; }

#pragma warning disable CS0672
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            if (_feature._resolveMat == null)
                return;

            var cmd = CommandBufferPool.Get("GaussianOIT_Resolve");
            using (new ProfilingScope(cmd, _ps))
            {
                cmd.SetGlobalTexture(_AccumColorTexID, _feature._accumColor);
                cmd.SetGlobalTexture(_AccumWeightTexID, _feature._accumWeight);
                cmd.SetGlobalInt(_DebugModeID, (int)_s.debugView);
#pragma warning disable CS0618
                var camColor = renderingData.cameraData.renderer.cameraColorTargetHandle;
#pragma warning restore CS0618
                CoreUtils.SetRenderTarget(cmd, camColor, ClearFlag.None);
                CoreUtils.DrawFullScreen(cmd, _feature._resolveMat);
            }
            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }
#pragma warning restore CS0672
    }
}
