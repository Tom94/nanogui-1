#include <nanogui/texture.h>
#include <nanogui/metal.h>
#import <Metal/Metal.h>

NAMESPACE_BEGIN(nanogui)

// Command queue for asynchronous cleanup operations
extern dispatch_queue_t cleanup_queue;

void Texture::init() {
    Vector2i size = m_size;
    m_size = 0;
    resize(size);

    MTLSamplerAddressMode wrap_mode_mtl;
    switch (m_wrap_mode) {
        case WrapMode::Repeat:       wrap_mode_mtl = MTLSamplerAddressModeRepeat; break;
        case WrapMode::ClampToEdge:  wrap_mode_mtl = MTLSamplerAddressModeClampToEdge; break;
        case WrapMode::MirrorRepeat: wrap_mode_mtl = MTLSamplerAddressModeMirrorRepeat; break;
        default: throw std::runtime_error("Texture::Texture(): invalid wrap mode!");
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>) metal_device();
    MTLSamplerDescriptor *sampler_desc = [MTLSamplerDescriptor new];

    sampler_desc.minFilter =
        m_min_interpolation_mode == InterpolationMode::Nearest
            ? MTLSamplerMinMagFilterNearest
            : MTLSamplerMinMagFilterLinear;

    sampler_desc.magFilter =
        m_mag_interpolation_mode == InterpolationMode::Nearest
            ? MTLSamplerMinMagFilterNearest
            : MTLSamplerMinMagFilterLinear;

    sampler_desc.mipFilter =
        (m_min_interpolation_mode == InterpolationMode::Trilinear ||
         m_mag_interpolation_mode == InterpolationMode::Trilinear)
            ? MTLSamplerMipFilterLinear
            : MTLSamplerMipFilterNotMipmapped;

    sampler_desc.sAddressMode = wrap_mode_mtl;
    sampler_desc.tAddressMode = wrap_mode_mtl;
    id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:sampler_desc];

    m_sampler_state_handle = (__bridge_retained void *) sampler;
}

Texture::~Texture() {
    (void) (__bridge_transfer id<MTLTexture>) m_handle;
    (void) (__bridge_transfer id<MTLSamplerState>) m_sampler_state_handle;
}

static const char *expand_kernel_source = R"(
#include <metal_stdlib>
using namespace metal;

constant bool swap_rb [[function_constant(0)]];

#define EXPAND_NORM(NAME, T, SCALE, LO)                                        \
kernel void NAME(device const T* src [[buffer(0)]],                            \
        texture2d<float, access::write> dst [[texture(0)]],                    \
        uint2 gid [[thread_position_in_grid]]) {                               \
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {               \
        return;                                                                \
    }                                                                          \
                                                                               \
    uint i = 3 * (gid.y * dst.get_width() + gid.x);                            \
    float3 rgb = float3(src[i], src[i + 1], src[i + 2]) * (SCALE);             \
    rgb = clamp(rgb, LO, 1.0f);                                                \
    if (swap_rb) {                                                             \
        rgb = rgb.bgr;                                                         \
    }                                                                          \
                                                                               \
    dst.write(float4(rgb, 1.0f), gid);                                         \
}

#define EXPAND_RAW(NAME, T)                                                    \
kernel void NAME(device const T* src [[buffer(0)]],                            \
        texture2d<float, access::write> dst [[texture(0)]],                    \
        uint2 gid [[thread_position_in_grid]]) {                               \
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {               \
        return;                                                                \
    }                                                                          \
                                                                               \
    uint i = 3 * (gid.y * dst.get_width() + gid.x);                            \
    float3 rgb = float3(src[i], src[i + 1], src[i + 2]);                       \
    if (swap_rb) {                                                             \
        rgb = rgb.bgr;                                                         \
    }                                                                          \
                                                                               \
    dst.write(float4(rgb, 1.0f), gid);                                         \
}

EXPAND_NORM(expand_rgb_u8,  uchar,  1.0f / 255.0f,    0.0f)
EXPAND_NORM(expand_rgb_i8,  char,   1.0f / 127.0f,   -1.0f)
EXPAND_NORM(expand_rgb_u16, ushort, 1.0f / 65535.0f,  0.0f)
EXPAND_NORM(expand_rgb_i16, short,  1.0f / 32767.0f, -1.0f)
EXPAND_RAW(expand_rgb_f16, half)
EXPAND_RAW(expand_rgb_f32, float)
)";

static id<MTLComputePipelineState> expand_pipeline(Texture::ComponentFormat comp_fmt,
                                                   bool swap_rb) {
    // [format][swap_rb]
    static id<MTLComputePipelineState> pipelines[6][2] = {};
    static id<MTLLibrary> library = nil;
    static std::mutex mutex;

    size_t index;
    const char *name;

    switch (comp_fmt) {
        case Texture::ComponentFormat::UInt8:
            index = 0; name = "expand_rgb_u8";  break;
        case Texture::ComponentFormat::Int8:
            index = 1; name = "expand_rgb_i8";  break;
        case Texture::ComponentFormat::UInt16:
            index = 2; name = "expand_rgb_u16"; break;
        case Texture::ComponentFormat::Int16:
            index = 3; name = "expand_rgb_i16"; break;
        case Texture::ComponentFormat::Float16:
            index = 4; name = "expand_rgb_f16"; break;
        case Texture::ComponentFormat::Float32:
            index = 5; name = "expand_rgb_f32"; break;
        default:
            // UInt32/Int32 are demoted in resize(); depth formats never get here.
            throw std::runtime_error(
                "Texture::upload_interleaved_async(): invalid component format!");
    }

    size_t swap_index = swap_rb ? 1 : 0;

    std::lock_guard<std::mutex> guard(mutex);

    if (pipelines[index][swap_index]) {
        return pipelines[index][swap_index];
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>) metal_device();
    NSError *error = nil;

    if (!library) {
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.fastMathEnabled = NO;

        library = [device newLibraryWithSource: @(expand_kernel_source)
                                       options: options
                                         error: &error];

        if (!library) {
            throw std::runtime_error(
                std::string("Texture::upload_interleaved_async(): could not compile "
                            "expansion library: ") + [[error description] UTF8String]);
        }
    }

    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    [constants setConstantValue: &swap_rb type: MTLDataTypeBool atIndex: 0];

    id<MTLFunction> function = [library newFunctionWithName: @(name)
                                             constantValues: constants
                                                      error: &error];

    if (!function) {
        throw std::runtime_error(
            std::string("Texture::upload_interleaved_async(): could not specialize "
                        "kernel: ") + [[error description] UTF8String]);
    }

    pipelines[index][swap_index] =
        [device newComputePipelineStateWithFunction: function error: &error];

    if (!pipelines[index][swap_index]) {
        throw std::runtime_error(
            std::string("Texture::upload_interleaved_async(): could not create "
                        "pipeline state: ") + [[error description] UTF8String]);
    }

    return pipelines[index][swap_index];
}

void Texture::upload_async(const uint8_t *data, void (*callback)(void*), void *payload) {
    upload_async(data, channels(), callback, payload);
}

void Texture::upload_async(const uint8_t *data, size_t src_channels,
                           void (*callback)(void *), void *payload) {
    if (!data) {
        return;
    }

    if (src_channels != channels()) {
        if (src_channels != 3 || channels() != 4 || (m_pixel_format != PixelFormat::RGBA &&
            m_pixel_format != PixelFormat::BGRA)) {
            throw std::runtime_error(
                    "Texture::upload_async(): only 3 -> 4 channel expansion is supported!");
        }

        if (!(m_flags & TextureFlags::ShaderWrite)) {
            throw std::runtime_error(
                    "Texture::upload_async(): texture must be created with "
                    "TextureFlags::ShaderWrite for expansion!");
        }
    }

    id<MTLTexture> texture = (__bridge id<MTLTexture>) m_handle;
    id<MTLDevice> device = (__bridge id<MTLDevice>) metal_device();
    id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>) metal_command_queue();

    size_t bytes_per_component = bytes_per_pixel() / channels();
    size_t data_size = bytes_per_component * src_channels * m_size.x() * m_size.y();

    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy: (void*)data
                                                     length: data_size
                                                    options: MTLResourceStorageModeShared
                                                deallocator: nil];

    if (buffer == nil) {
        // Likely, ``data`` is not page-aligned
        buffer = [device newBufferWithBytes: data
                                     length: data_size
                                    options: MTLResourceStorageModeShared];
    }

    if (buffer == nil) {
        throw std::runtime_error(
            "Texture::upload_and_expand_async(): could not allocate staging buffer!");
    }

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];

    if (src_channels != channels()) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        bool swap_rb = m_pixel_format == PixelFormat::BGRA;
        id<MTLComputePipelineState> pipeline = expand_pipeline(m_component_format, swap_rb);

        [encoder setComputePipelineState: pipeline];
        [encoder setBuffer: buffer offset: 0 atIndex: 0];
        [encoder setTexture: texture atIndex: 0];

        NSUInteger tg_w = pipeline.threadExecutionWidth;
        NSUInteger tg_h = pipeline.maxTotalThreadsPerThreadgroup / tg_w;

        [encoder dispatchThreads: MTLSizeMake(m_size.x(), m_size.y(), 1)
            threadsPerThreadgroup: MTLSizeMake(tg_w, tg_h, 1)];
        [encoder endEncoding];
    } else {
        id<MTLBlitCommandEncoder> blit_encoder = [command_buffer blitCommandEncoder];
        size_t bytes_per_row = bytes_per_pixel() * m_size.x();

        [blit_encoder copyFromBuffer: buffer
                        sourceOffset: 0
                   sourceBytesPerRow: bytes_per_row
                 sourceBytesPerImage: data_size
                          sourceSize: MTLSizeMake(m_size.x(), m_size.y(), 1)
                           toTexture: texture
                    destinationSlice: 0
                    destinationLevel: 0
                   destinationOrigin: MTLOriginMake(0, 0, 0)];

        [blit_encoder endEncoding];
    }

    if (!m_mipmap_manual &&
        (m_min_interpolation_mode == InterpolationMode::Trilinear ||
         m_mag_interpolation_mode == InterpolationMode::Trilinear)) {
        id<MTLBlitCommandEncoder> mipmap_encoder = [command_buffer blitCommandEncoder];
        [mipmap_encoder generateMipmapsForTexture: texture];
        [mipmap_encoder endEncoding];
    }

    if (callback) {
        [command_buffer addCompletedHandler: ^(id<MTLCommandBuffer> cb) {
            dispatch_async((__bridge dispatch_queue_t) metal_cleanup_queue(), ^{
                callback(payload);
            });
        }];
        [command_buffer commit];
    } else {
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }
}

void Texture::upload(const uint8_t *data) {
    upload_async(data, nullptr, nullptr);
}

void Texture::upload_sub_region(const uint8_t *data, const Vector2i& origin, const Vector2i& size) {
    if (m_samples > 1 && data != nullptr)
        throw std::runtime_error("Texture::upload_sub_region(): only implemented for samples=1!");

    id<MTLTexture> texture = (__bridge id<MTLTexture>) m_handle;
    id<MTLDevice> device = (__bridge id<MTLDevice>) metal_device();
    id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>) metal_command_queue();

    size_t bytes_per_row = bytes_per_pixel() * size.x();
    size_t data_size = bytes_per_row * size.y();

    // Copy into a CPU/GPU-shared staging buffer and blit it into the sub-region.
    // The command buffer retains the buffer until the GPU is finished.
    id<MTLBuffer> buffer = [device newBufferWithBytes: data
                                               length: data_size
                                              options: MTLResourceStorageModeShared];

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit_encoder = [command_buffer blitCommandEncoder];

    [blit_encoder copyFromBuffer: buffer
                    sourceOffset: 0
               sourceBytesPerRow: bytes_per_row
             sourceBytesPerImage: data_size
                      sourceSize: MTLSizeMake((NSUInteger) size.x(), (NSUInteger) size.y(), 1)
                       toTexture: texture
                destinationSlice: 0
                destinationLevel: 0
               destinationOrigin: MTLOriginMake((NSUInteger) origin.x(), (NSUInteger) origin.y(), 0)];

    [blit_encoder endEncoding];
    [command_buffer commit];

    if (!m_mipmap_manual &&
        (m_min_interpolation_mode == InterpolationMode::Trilinear ||
         m_mag_interpolation_mode == InterpolationMode::Trilinear))
        generate_mipmap();
}

void Texture::download(uint8_t *data) {
    id<MTLCommandQueue> command_queue =
        (__bridge id<MTLCommandQueue>) metal_command_queue();
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLBlitCommandEncoder> command_encoder =
        [command_buffer blitCommandEncoder];

    size_t row_bytes = bytes_per_pixel() * m_size.x(),
           img_bytes = row_bytes * m_size.y();

    id<MTLDevice> device = (__bridge id<MTLDevice>) metal_device();
    id<MTLTexture> texture = (__bridge id<MTLTexture>) m_handle;
    id<MTLBuffer> buffer =
        [device newBufferWithLength: img_bytes
                            options: MTLResourceStorageModeShared];

    [command_encoder
                 copyFromTexture: texture
                     sourceSlice: 0
                     sourceLevel: 0
                    sourceOrigin: MTLOriginMake(0, 0, 0)
                      sourceSize: MTLSizeMake(texture.width, texture.height, 1)
                        toBuffer: buffer
               destinationOffset: 0
          destinationBytesPerRow: row_bytes
        destinationBytesPerImage: img_bytes];

    [command_encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    memcpy(data, buffer.contents, img_bytes);
}

void Texture::resize(const Vector2i &size) {
    if (m_size == size)
        return;
    m_size = size;
    if (m_handle) {
        (void) (__bridge_transfer id<MTLTexture>) m_handle;
        m_handle = nullptr;
    }

    if (m_component_format == ComponentFormat::UInt32)
        m_component_format = ComponentFormat::UInt16;
    else if (m_component_format == ComponentFormat::Int32)
        m_component_format = ComponentFormat::Int16;

    if (m_pixel_format == PixelFormat::RGB)
        m_pixel_format = PixelFormat::RGBA;
    else if (m_pixel_format == PixelFormat::BGR)
        m_pixel_format = PixelFormat::BGRA;

    if (m_pixel_format == PixelFormat::BGRA &&
        m_component_format != ComponentFormat::UInt8)
        m_pixel_format = PixelFormat::RGBA;

    MTLPixelFormat pixel_format_mtl;
    switch (m_pixel_format) {
        case PixelFormat::R:
            switch (m_component_format) {
                case ComponentFormat::UInt8:   pixel_format_mtl = MTLPixelFormatR8Unorm;  break;
                case ComponentFormat::Int8:    pixel_format_mtl = MTLPixelFormatR8Snorm;  break;
                case ComponentFormat::UInt16:  pixel_format_mtl = MTLPixelFormatR16Unorm; break;
                case ComponentFormat::Int16:   pixel_format_mtl = MTLPixelFormatR16Snorm; break;
                case ComponentFormat::Float16: pixel_format_mtl = MTLPixelFormatR16Float; break;
                case ComponentFormat::Float32: pixel_format_mtl = MTLPixelFormatR32Float; break;
                default: throw std::runtime_error("Texture::Texture(): invalid component format!");
            }
            break;

        case PixelFormat::RA:
            switch (m_component_format) {
                case ComponentFormat::UInt8:   pixel_format_mtl = MTLPixelFormatRG8Unorm;  break;
                case ComponentFormat::Int8:    pixel_format_mtl = MTLPixelFormatRG8Snorm;  break;
                case ComponentFormat::UInt16:  pixel_format_mtl = MTLPixelFormatRG16Unorm; break;
                case ComponentFormat::Int16:   pixel_format_mtl = MTLPixelFormatRG16Snorm; break;
                case ComponentFormat::Float16: pixel_format_mtl = MTLPixelFormatRG16Float; break;
                case ComponentFormat::Float32: pixel_format_mtl = MTLPixelFormatRG32Float; break;
                default: throw std::runtime_error("Texture::Texture(): invalid component format!");
            }
            break;

        case PixelFormat::BGRA:
            switch (m_component_format) {
                case ComponentFormat::UInt8:   pixel_format_mtl = MTLPixelFormatBGRA8Unorm;  break;
                default: throw std::runtime_error("Texture::Texture(): invalid component format!");
            }
            break;

        case PixelFormat::RGBA:
            switch (m_component_format) {
                case ComponentFormat::UInt8:   pixel_format_mtl = MTLPixelFormatRGBA8Unorm;  break;
                case ComponentFormat::Int8:    pixel_format_mtl = MTLPixelFormatRGBA8Snorm;  break;
                case ComponentFormat::UInt16:  pixel_format_mtl = MTLPixelFormatRGBA16Unorm; break;
                case ComponentFormat::Int16:   pixel_format_mtl = MTLPixelFormatRGBA16Snorm; break;
                case ComponentFormat::Float16: pixel_format_mtl = MTLPixelFormatRGBA16Float; break;
                case ComponentFormat::Float32: pixel_format_mtl = MTLPixelFormatRGBA32Float; break;
                default: throw std::runtime_error("Texture::Texture(): invalid component format!");
            }
            break;

        case PixelFormat::Depth:
            switch (m_component_format) {
                case ComponentFormat::Int8:
                case ComponentFormat::UInt8:
                case ComponentFormat::Int16:
                case ComponentFormat::UInt16:
                    m_component_format = ComponentFormat::UInt16;
                    pixel_format_mtl = MTLPixelFormatDepth16Unorm;
                    break;

                case ComponentFormat::Int32:
                case ComponentFormat::UInt32:
                case ComponentFormat::Float16:
                case ComponentFormat::Float32:
                    m_component_format = ComponentFormat::Float32;
                    pixel_format_mtl = MTLPixelFormatDepth32Float;
                    break;

                default: throw std::runtime_error("Texture::Texture(): invalid component format!");
            }
            break;

        case PixelFormat::DepthStencil:
            switch (m_component_format) {
                case ComponentFormat::Int8:
                case ComponentFormat::UInt8:
                case ComponentFormat::Int16:
                case ComponentFormat::UInt16:
                case ComponentFormat::Int32:
                case ComponentFormat::UInt32: {
                    // Depth24Unorm_Stencil8 is unavailable on Apple Silicon
                    id<MTLDevice> dev = (__bridge id<MTLDevice>) metal_device();
                    if (dev.depth24Stencil8PixelFormatSupported) {
                        m_component_format = ComponentFormat::UInt32;
                        pixel_format_mtl = MTLPixelFormatDepth24Unorm_Stencil8;
                    } else {
                        m_component_format = ComponentFormat::Float32;
                        pixel_format_mtl = MTLPixelFormatDepth32Float_Stencil8;
                    }
                    break;
                }

                case ComponentFormat::Float16:
                case ComponentFormat::Float32:
                    m_component_format = ComponentFormat::Float32;
                    pixel_format_mtl = MTLPixelFormatDepth32Float_Stencil8;
                    break;

                default: throw std::runtime_error("Texture::Texture(): invalid component format!");
            }
            break;

        default:
            throw std::runtime_error("Texture::Texture(): invalid pixel format!");
    }

    bool mipmap = m_min_interpolation_mode == InterpolationMode::Trilinear ||
                  m_mag_interpolation_mode == InterpolationMode::Trilinear;
    id<MTLDevice> device = (__bridge id<MTLDevice>) metal_device();
    MTLTextureDescriptor *texture_desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat: pixel_format_mtl
                                                           width: (NSUInteger) m_size.x()
                                                          height: (NSUInteger) m_size.y()
                                                       mipmapped: mipmap];
    texture_desc.storageMode = MTLStorageModePrivate;
    texture_desc.usage = 0;

    if (m_samples > 1) {
        texture_desc.textureType = MTLTextureType2DMultisample;
        texture_desc.sampleCount = m_samples;
    }

    if (m_flags & (uint8_t) TextureFlags::ShaderRead)
        texture_desc.usage |= MTLTextureUsageShaderRead;
    if (m_flags & (uint8_t) TextureFlags::RenderTarget)
        texture_desc.usage |= MTLTextureUsageRenderTarget;
    if (m_flags & (uint8_t) TextureFlags::ShaderWrite)
        texture_desc.usage |= MTLTextureUsageShaderWrite;
    if (texture_desc.usage == 0)
        throw std::runtime_error("Texture::Texture(): flags must have at least one of "
                                 "ShaderRead, RenderTarget, and ShaderWrite!");

    id<MTLTexture> texture = [device newTextureWithDescriptor:texture_desc];
    m_handle = (__bridge_retained void *) texture;
}

void Texture::generate_mipmap() {
    id<MTLTexture> texture = (__bridge id<MTLTexture>) m_handle;
    id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>) metal_command_queue();
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLBlitCommandEncoder> command_encoder = [command_buffer blitCommandEncoder];

    [command_encoder generateMipmapsForTexture: texture];
    [command_encoder endEncoding];
    [command_buffer commit];
}

NAMESPACE_END(nanogui)
