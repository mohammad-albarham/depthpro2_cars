//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

import MetalKit

let renderShader = """

    struct Varyings { float4 position [[position]]; float2 uv; };

    vertex Varyings fullscreen(uint vid [[vertex_id]]) {
        float2 p = float2((vid << 1) & 2, vid & 2);
        return { float4(p * 2.0 - 1.0, 0.0, 1.0), float2(p.x, 1.0 - p.y) };
    }

    // Turbo colormap, polynomial approximation by Ruofei Du of Anton Mikhailov's design:
    // https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
    // Copyright 2019 Google LLC. SPDX-License-Identifier: Apache-2.0
    float3 turbo(float x) {
        const float4 r4 = float4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
        const float4 g4 = float4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
        const float4 b4 = float4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
        const float2 r2 = float2(-152.94239396, 59.28637943);
        const float2 g2 = float2(4.27729857, 2.82956604);
        const float2 b2 = float2(-89.90310912, 27.34824973);
        float4 v4 = float4(1.0, x, x * x, x * x * x);
        float2 v2 = v4.zw * v4.z;
        return float3(dot(v4, r4) + dot(v2, r2), dot(v4, g4) + dot(v2, g2), dot(v4, b4) + dot(v2, b2));
    }

    fragment half4 side_by_side(Varyings in [[stage_in]],
                                texture2d<half> camera [[texture(0)]],
                                texture2d<float> inverseDepth [[texture(1)]],
                                constant float3& scaleNearFar [[buffer(0)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 uv = float2(fract(in.uv.x * 2.0), in.uv.y);
        if (abs(in.uv.x - 0.5) < fwidth(in.uv.x)) return half4(0.0h, 0.0h, 0.0h, 1.0h);
        half4 color;
        if (in.uv.x < 0.5) {
            color = camera.sample(s, uv);
        } else {
            float meters = scaleNearFar.x / max(inverseDepth.sample(s, uv).r, 1e-6);
            float t = 1.0 - saturate(log(meters / scaleNearFar.y) / log(scaleNearFar.z / scaleNearFar.y));
            color = half4(half3(turbo(t)), 1.0h);
        }
        return color;
    }

    """

@MainActor
final class Renderer: NSObject, MTKViewDelegate {
    private let gpu: GPU
    private let pipeline: MTLRenderPipelineState
    private let worker: Worker
    private let metrics: Metrics
    private var lastSeq = -1

    init(gpu: GPU, view: MTKView, worker: Worker, metrics: Metrics) throws {
        self.gpu = gpu
        self.worker = worker
        self.metrics = metrics
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = gpu.library.makeFunction(name: "fullscreen")
        descriptor.fragmentFunction = gpu.library.makeFunction(name: "side_by_side")
        descriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        pipeline = try gpu.device.makeRenderPipelineState(descriptor: descriptor)
        super.init()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let pool = worker.pool, let slot = pool.acquireLatest() else { return }
        guard slot.times.seq != lastSeq, let pass = view.currentRenderPassDescriptor,
            let drawable = view.currentDrawable, let commands = gpu.queue.makeCommandBuffer(),
            let encoder = commands.makeRenderCommandEncoder(descriptor: pass)
        else {
            pool.release(slot)
            return
        }
        lastSeq = slot.times.seq
        var scaleNearFar = SIMD3(slot.metricScale, slot.depthRange.x, slot.depthRange.y)
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentTexture(slot.camera, index: 0)
        encoder.setFragmentTexture(slot.inverseDepthTexture, index: 1)
        encoder.setFragmentBytes(&scaleNearFar, length: MemoryLayout<SIMD3<Float>>.size, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()

        let (times, metrics) = (slot.times, metrics)
        drawable.addPresentedHandler { drawable in
            guard drawable.presentedTime > 0 else { return }
            metrics.displayed(times, at: drawable.presentedTime)
        }
        commands.present(drawable)
        commands.addCompletedHandler { _ in pool.release(slot) }
        commands.commit()
    }
}
