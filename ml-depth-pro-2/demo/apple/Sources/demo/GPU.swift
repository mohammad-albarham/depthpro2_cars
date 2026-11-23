//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

import CoreVideo
import Metal

struct GPU {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue()
        else { throw DemoError("no Metal device") }
        self.device = device
        self.queue = queue
        library = try device.makeLibrary(
            source: "#include <metal_stdlib>\nusing namespace metal;\n" + preprocessShader + renderShader
                + fieldOfViewShader,
            options: nil)
    }

    func privateTexture(_ format: MTLPixelFormat, width: Int, height: Int) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .private
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw DemoError("cannot allocate a \(width)x\(height) texture")
        }
        return texture
    }

    func texture(
        _ buffer: CVPixelBuffer, format: MTLPixelFormat, usage: MTLTextureUsage
    ) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: CVPixelBufferGetWidth(buffer),
            height: CVPixelBufferGetHeight(buffer), mipmapped: false)
        descriptor.usage = usage
        guard let surface = CVPixelBufferGetIOSurface(buffer)?.takeUnretainedValue(),
            let texture = device.makeTexture(descriptor: descriptor, iosurface: surface, plane: 0)
        else { throw DemoError("pixel buffer is not IOSurface-backed") }
        return texture
    }
}

func makeHalfSurface(width: Int, height: Int) throws -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    let attributes: [CFString: Any] = [
        kCVPixelBufferIOSurfacePropertiesKey: [String: Any](),
        kCVPixelBufferMetalCompatibilityKey: true,
    ]
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent16Half,
        attributes as CFDictionary, &buffer)
    guard status == kCVReturnSuccess, let buffer else {
        throw DemoError("CVPixelBufferCreate \(width)x\(height) failed (\(status))")
    }
    return buffer
}
