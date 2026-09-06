#define NOMINMAX // windows.h's min/max macros shadow std::min/std::max otherwise
#include "raycaster.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <d3dcompiler.h>

namespace
{

// Fullscreen-triangle raymarcher. cam_* and box_* are world mm (LPS).
// window_lo/window_width are pre-converted into the texture's normalized
// [0,1] density units (see RenderRaycast) so this doesn't need hu_min/range.
const char* kShaderSource = R"hlsl(
Texture3D<float> VolumeTex : register(t0);
SamplerState VolumeSampler : register(s0);

cbuffer RaycastCB : register(b0)
{
    float3 cam_pos;     float aspect;
    float3 cam_right;   float tan_half_fov;
    float3 cam_up;      float step_size;
    float3 cam_forward; float pad0;
    float3 box_min;     float window_lo;
    float3 box_max;     float window_width;
};

struct VSOut { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };

VSOut VSMain(uint vid : SV_VertexID)
{
    VSOut o;
    float2 xy = float2((vid == 2) ? 3.0 : -1.0, (vid == 1) ? 3.0 : -1.0);
    o.pos = float4(xy, 0.0, 1.0);
    o.uv = xy;
    return o;
}

float4 PSMain(VSOut input) : SV_TARGET
{
    float3 ray_dir = normalize(cam_forward + input.uv.x * tan_half_fov * aspect * cam_right + input.uv.y * tan_half_fov * cam_up);

    float3 inv_dir = 1.0 / ray_dir;
    float3 t0v = (box_min - cam_pos) * inv_dir;
    float3 t1v = (box_max - cam_pos) * inv_dir;
    float3 tmin3 = min(t0v, t1v);
    float3 tmax3 = max(t0v, t1v);
    float tmin = max(max(tmin3.x, tmin3.y), max(tmin3.z, 0.0));
    float tmax = min(min(tmax3.x, tmax3.y), tmax3.z);

    float4 accum = float4(0, 0, 0, 0);
    if (tmax > tmin)
    {
        float t = tmin;
        [loop]
        for (int i = 0; i < 1024; ++i)
        {
            if (t >= tmax || accum.a >= 0.98)
                break;
            float3 world_pos = cam_pos + ray_dir * t;
            float3 uvw = (world_pos - box_min) / (box_max - box_min);
            float density = VolumeTex.SampleLevel(VolumeSampler, uvw, 0);
            float val = saturate((density - window_lo) / window_width);
            float sample_alpha = val * 0.12; // per-step opacity; tuned by eye for this transfer function
            accum.rgb += (1.0 - accum.a) * sample_alpha * val;
            accum.a += (1.0 - accum.a) * sample_alpha;
            t += step_size;
        }
    }

    if (accum.a <= 0.0001)
        return float4(0, 0, 0, 0);
    return float4(accum.rgb / accum.a, accum.a); // un-premultiply: ImGui::Image blends straight alpha
}
)hlsl";

// Must match the HLSL cbuffer's field order/types exactly: six (float3,
// float) pairs, each occupying one 16-byte register with no gaps.
struct RaycastCB
{
    float cam_pos[3];     float aspect;
    float cam_right[3];   float tan_half_fov;
    float cam_up[3];      float step_size;
    float cam_forward[3]; float pad0;
    float box_min[3];     float window_lo;
    float box_max[3];     float window_width;
};

void Normalize3(float v[3])
{
    float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len > 1e-8f) { v[0] /= len; v[1] /= len; v[2] /= len; }
}

void Cross3(const float a[3], const float b[3], float out[3])
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

bool CompileShader(const char* entry_point, const char* target, ID3DBlob** out_blob)
{
    ID3DBlob* error_blob = nullptr;
    HRESULT hr = D3DCompile(kShaderSource, strlen(kShaderSource), "raycast.hlsl", nullptr, nullptr,
        entry_point, target, 0, 0, out_blob, &error_blob);
    if (FAILED(hr))
    {
        fprintf(stderr, "Raycaster: failed to compile %s (%s): %s\n", entry_point, target,
            error_blob ? (const char*)error_blob->GetBufferPointer() : "(no error blob)");
        if (error_blob) error_blob->Release();
        return false;
    }
    if (error_blob) error_blob->Release();
    return true;
}

} // namespace

bool InitRaycaster(ID3D11Device* device, const Volume& volume, Raycaster& out)
{
    int16_t min_hu = *std::min_element(volume.data.begin(), volume.data.end());
    int16_t max_hu = *std::max_element(volume.data.begin(), volume.data.end());
    float range = (float)(max_hu - min_hu);
    if (range < 1.0f) range = 1.0f;
    out.hu_min = (float)min_hu;
    out.hu_range = range;

    std::vector<uint8_t> normalized(volume.data.size());
    for (size_t idx = 0; idx < volume.data.size(); ++idx)
    {
        float t = ((float)volume.data[idx] - min_hu) / range;
        normalized[idx] = (uint8_t)(std::clamp(t, 0.0f, 1.0f) * 255.0f + 0.5f);
    }

    D3D11_TEXTURE3D_DESC tex_desc = {};
    tex_desc.Width = volume.nx;
    tex_desc.Height = volume.ny;
    tex_desc.Depth = volume.nz;
    tex_desc.MipLevels = 1;
    tex_desc.Format = DXGI_FORMAT_R8_UNORM;
    tex_desc.Usage = D3D11_USAGE_IMMUTABLE;
    tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    D3D11_SUBRESOURCE_DATA init_data = {};
    init_data.pSysMem = normalized.data();
    init_data.SysMemPitch = (UINT)volume.nx * sizeof(uint8_t);
    init_data.SysMemSlicePitch = (UINT)volume.nx * volume.ny * sizeof(uint8_t);

    if (FAILED(device->CreateTexture3D(&tex_desc, &init_data, &out.volume_tex)))
    {
        fprintf(stderr, "Raycaster: CreateTexture3D failed\n");
        return false;
    }
    if (FAILED(device->CreateShaderResourceView(out.volume_tex, nullptr, &out.volume_srv)))
        return false;

    D3D11_SAMPLER_DESC sampler_desc = {};
    sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc.AddressU = sampler_desc.AddressV = sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    if (FAILED(device->CreateSamplerState(&sampler_desc, &out.sampler)))
        return false;

    ID3DBlob* vs_blob = nullptr;
    ID3DBlob* ps_blob = nullptr;
    if (!CompileShader("VSMain", "vs_5_0", &vs_blob))
        return false;
    if (!CompileShader("PSMain", "ps_5_0", &ps_blob))
    {
        vs_blob->Release();
        return false;
    }
    HRESULT vs_hr = device->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, &out.vertex_shader);
    HRESULT ps_hr = device->CreatePixelShader(ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr, &out.pixel_shader);
    vs_blob->Release();
    ps_blob->Release();
    if (FAILED(vs_hr) || FAILED(ps_hr))
        return false;

    D3D11_BUFFER_DESC cb_desc = {};
    cb_desc.ByteWidth = sizeof(RaycastCB);
    cb_desc.Usage = D3D11_USAGE_DYNAMIC;
    cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cb_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(device->CreateBuffer(&cb_desc, nullptr, &out.constant_buffer)))
        return false;

    D3D11_RASTERIZER_DESC rs_desc = {};
    rs_desc.FillMode = D3D11_FILL_SOLID;
    rs_desc.CullMode = D3D11_CULL_NONE;
    if (FAILED(device->CreateRasterizerState(&rs_desc, &out.rasterizer_state)))
        return false;

    D3D11_DEPTH_STENCIL_DESC ds_desc = {};
    ds_desc.DepthEnable = FALSE;
    ds_desc.StencilEnable = FALSE;
    if (FAILED(device->CreateDepthStencilState(&ds_desc, &out.depth_stencil_state)))
        return false;

    D3D11_TEXTURE2D_DESC rt_desc = {};
    rt_desc.Width = out.target_size;
    rt_desc.Height = out.target_size;
    rt_desc.MipLevels = 1;
    rt_desc.ArraySize = 1;
    rt_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rt_desc.SampleDesc.Count = 1;
    rt_desc.Usage = D3D11_USAGE_DEFAULT;
    rt_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(device->CreateTexture2D(&rt_desc, nullptr, &out.target_tex)))
        return false;
    if (FAILED(device->CreateRenderTargetView(out.target_tex, nullptr, &out.target_rtv)))
        return false;
    if (FAILED(device->CreateShaderResourceView(out.target_tex, nullptr, &out.target_srv)))
        return false;

    double dx = volume.nx * volume.spacing[0];
    double dy = volume.ny * volume.spacing[1];
    double dz = volume.nz * volume.spacing[2];
    out.distance = (float)std::sqrt(dx * dx + dy * dy + dz * dz) * 1.1f;

    return true;
}

CameraFrame ComputeCameraFrame(const Raycaster& rc, const Volume& volume)
{
    double box_min[3] = { volume.origin[0], volume.origin[1], volume.origin[2] };
    double box_max[3] = {
        volume.origin[0] + volume.nx * volume.spacing[0],
        volume.origin[1] + volume.ny * volume.spacing[1],
        volume.origin[2] + volume.nz * volume.spacing[2],
    };
    float center[3] = {
        (float)((box_min[0] + box_max[0]) * 0.5) + rc.target_offset[0],
        (float)((box_min[1] + box_max[1]) * 0.5) + rc.target_offset[1],
        (float)((box_min[2] + box_max[2]) * 0.5) + rc.target_offset[2],
    };

    float ce = cosf(rc.elevation), se = sinf(rc.elevation);
    float ca = cosf(rc.azimuth), sa = sinf(rc.azimuth);
    float dir[3] = { ce * ca, ce * sa, se }; // spherical, +z (superior) as the pole

    CameraFrame frame = {};
    frame.pos[0] = center[0] + rc.distance * dir[0];
    frame.pos[1] = center[1] + rc.distance * dir[1];
    frame.pos[2] = center[2] + rc.distance * dir[2];

    frame.forward[0] = center[0] - frame.pos[0];
    frame.forward[1] = center[1] - frame.pos[1];
    frame.forward[2] = center[2] - frame.pos[2];
    Normalize3(frame.forward);

    float world_up[3] = { 0.0f, 0.0f, 1.0f };
    Cross3(frame.forward, world_up, frame.right);
    if (std::sqrt(frame.right[0] * frame.right[0] + frame.right[1] * frame.right[1] + frame.right[2] * frame.right[2]) < 1e-4f)
    {
        world_up[0] = 0.0f; world_up[1] = 1.0f; world_up[2] = 0.0f; // near-pole fallback
        Cross3(frame.forward, world_up, frame.right);
    }
    Normalize3(frame.right);
    Cross3(frame.right, frame.forward, frame.up);
    Normalize3(frame.up);

    frame.tan_half_fov = tanf(0.5f * (45.0f * 3.14159265f / 180.0f));
    frame.aspect = 1.0f; // square render target

    return frame;
}

bool ProjectToNDC(const CameraFrame& cam, const float world_point[3], float out_ndc[2])
{
    float rel[3] = { world_point[0] - cam.pos[0], world_point[1] - cam.pos[1], world_point[2] - cam.pos[2] };
    float depth = rel[0] * cam.forward[0] + rel[1] * cam.forward[1] + rel[2] * cam.forward[2];
    if (depth <= 1e-3f)
        return false; // behind the camera

    float right_component = rel[0] * cam.right[0] + rel[1] * cam.right[1] + rel[2] * cam.right[2];
    float up_component = rel[0] * cam.up[0] + rel[1] * cam.up[1] + rel[2] * cam.up[2];
    out_ndc[0] = right_component / (depth * cam.tan_half_fov * cam.aspect);
    out_ndc[1] = up_component / (depth * cam.tan_half_fov);
    return true;
}

void RenderRaycast(ID3D11DeviceContext* context, Raycaster& rc, const Volume& volume, float window_width, float window_level)
{
    CameraFrame cam = ComputeCameraFrame(rc, volume);
    memcpy(rc.basis_right, cam.right, sizeof(cam.right));
    memcpy(rc.basis_up, cam.up, sizeof(cam.up));

    double box_min[3] = { volume.origin[0], volume.origin[1], volume.origin[2] };
    double box_max[3] = {
        volume.origin[0] + volume.nx * volume.spacing[0],
        volume.origin[1] + volume.ny * volume.spacing[1],
        volume.origin[2] + volume.nz * volume.spacing[2],
    };

    RaycastCB cb = {};
    memcpy(cb.cam_pos, cam.pos, sizeof(cam.pos));
    cb.aspect = cam.aspect;
    memcpy(cb.cam_right, cam.right, sizeof(cam.right));
    cb.tan_half_fov = cam.tan_half_fov;
    memcpy(cb.cam_up, cam.up, sizeof(cam.up));
    double min_spacing = std::min({ volume.spacing[0], volume.spacing[1], volume.spacing[2] });
    cb.step_size = (float)min_spacing * 0.75f;
    memcpy(cb.cam_forward, cam.forward, sizeof(cam.forward));
    cb.pad0 = 0.0f;
    for (int i = 0; i < 3; ++i) cb.box_min[i] = (float)box_min[i];
    cb.window_lo = (window_level - window_width * 0.5f - rc.hu_min) / rc.hu_range;
    for (int i = 0; i < 3; ++i) cb.box_max[i] = (float)box_max[i];
    cb.window_width = window_width / rc.hu_range;

    D3D11_MAPPED_SUBRESOURCE mapped;
    if (FAILED(context->Map(rc.constant_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
        return;
    memcpy(mapped.pData, &cb, sizeof(cb));
    context->Unmap(rc.constant_buffer, 0);

    D3D11_VIEWPORT viewport = { 0.0f, 0.0f, (float)rc.target_size, (float)rc.target_size, 0.0f, 1.0f };
    const float clear_color[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    context->ClearRenderTargetView(rc.target_rtv, clear_color);
    context->OMSetRenderTargets(1, &rc.target_rtv, nullptr);
    context->OMSetDepthStencilState(rc.depth_stencil_state, 0);
    context->RSSetState(rc.rasterizer_state);
    context->RSSetViewports(1, &viewport);
    context->IASetInputLayout(nullptr);
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context->VSSetShader(rc.vertex_shader, nullptr, 0);
    context->PSSetShader(rc.pixel_shader, nullptr, 0);
    context->PSSetShaderResources(0, 1, &rc.volume_srv);
    context->PSSetSamplers(0, 1, &rc.sampler);
    context->PSSetConstantBuffers(0, 1, &rc.constant_buffer);
    context->Draw(3, 0);

    ID3D11ShaderResourceView* null_srv = nullptr;
    context->PSSetShaderResources(0, 1, &null_srv); // unbind before it's next read as ImGui's Image source
}

void ReleaseRaycaster(Raycaster& rc)
{
    if (rc.target_srv) rc.target_srv->Release();
    if (rc.target_rtv) rc.target_rtv->Release();
    if (rc.target_tex) rc.target_tex->Release();
    if (rc.depth_stencil_state) rc.depth_stencil_state->Release();
    if (rc.rasterizer_state) rc.rasterizer_state->Release();
    if (rc.constant_buffer) rc.constant_buffer->Release();
    if (rc.pixel_shader) rc.pixel_shader->Release();
    if (rc.vertex_shader) rc.vertex_shader->Release();
    if (rc.sampler) rc.sampler->Release();
    if (rc.volume_srv) rc.volume_srv->Release();
    if (rc.volume_tex) rc.volume_tex->Release();
    rc = Raycaster{};
}
