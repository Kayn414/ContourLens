#pragma once

// A minimal single-pass GPU volume raycaster: the CT is uploaded once as a
// normalized Texture3D, and a full-screen pixel shader ray-marches it into
// an offscreen render target each frame (window/level and camera can change
// every frame without re-touching the 3D texture). Replaces the CPU-MIP
// placeholder for the "3D" panel. See the GUI planning doc's "3D panel"
// reading list (Will Usher's WebGL raycaster is the shader this is ported
// from, HLSL instead of GLSL) for the technique.

#include "volume.h"
#include <d3d11.h>

struct Raycaster
{
    ID3D11Texture3D* volume_tex = nullptr;
    ID3D11ShaderResourceView* volume_srv = nullptr;
    ID3D11SamplerState* sampler = nullptr;
    ID3D11VertexShader* vertex_shader = nullptr;
    ID3D11PixelShader* pixel_shader = nullptr;
    ID3D11Buffer* constant_buffer = nullptr;
    ID3D11RasterizerState* rasterizer_state = nullptr;
    ID3D11DepthStencilState* depth_stencil_state = nullptr;

    ID3D11Texture2D* target_tex = nullptr;
    ID3D11ShaderResourceView* target_srv = nullptr;
    ID3D11RenderTargetView* target_rtv = nullptr;
    int target_size = 768;

    float hu_min = 0.0f, hu_range = 1.0f; // volume_tex normalization, for converting HU window -> [0,1]

    // Orbit camera in world mm (LPS; +z is superior, used as "up").
    float azimuth = 0.7f;
    float elevation = 0.35f;
    float distance = 600.0f;
    float target_offset[3] = { 0.0f, 0.0f, 0.0f }; // right-drag pan, added to the volume-center look-at point

    // Camera right/up basis from the most recent RenderRaycast call, in world
    // mm -- the 3D panel's right-drag pan handler reads these (one frame
    // stale is fine) to move target_offset along the screen-space drag.
    float basis_right[3] = { 1.0f, 0.0f, 0.0f };
    float basis_up[3] = { 0.0f, 0.0f, 1.0f };
};

// Uploads `volume` as a normalized 3D texture and compiles the raymarch
// shaders. Returns false (logging to stderr) on failure.
bool InitRaycaster(ID3D11Device* device, const Volume& volume, Raycaster& out);

// The orbit camera's world-space position/basis for the current frame (mm,
// LPS). Shared by RenderRaycast (feeds the raymarch shader's constant
// buffer) and by the "3D" panel's slice-plane overlay (projects world points
// into the same view so the outlines line up with the raycast exactly).
struct CameraFrame
{
    float pos[3];
    float right[3];
    float up[3];
    float forward[3];
    float tan_half_fov;
    float aspect;
};

CameraFrame ComputeCameraFrame(const Raycaster& rc, const Volume& volume);

// Projects a world-space point into the raycast image's [-1,1] NDC-like
// space (same convention as the shader: +x right, +y up). Returns false
// (point behind the camera) if it shouldn't be drawn.
bool ProjectToNDC(const CameraFrame& cam, const float world_point[3], float out_ndc[2]);

// Ray-marches the volume from the current camera into `rc.target_tex`, using
// `window_width`/`window_level` (same HU units as the 2D slice windowing) as
// the opacity/color transfer function. Call once per frame while the 3D
// panel is visible; cheap enough to always re-render (no dirty tracking).
void RenderRaycast(ID3D11DeviceContext* context, Raycaster& rc, const Volume& volume, float window_width, float window_level);

void ReleaseRaycaster(Raycaster& rc);
