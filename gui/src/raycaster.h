#pragma once

// A minimal single-pass GPU volume raycaster: the CT is uploaded once as a
// normalized Texture3D, and a full-screen pixel shader ray-marches it into
// an offscreen render target each frame (window/level and camera can change
// every frame without re-touching the 3D texture). Replaces the CPU-MIP
// placeholder for the "3D" panel. See the GUI planning doc's "3D panel"
// reading list (Will Usher's WebGL raycaster is the shader this is ported
// from, HLSL instead of GLSL) for the technique.

#include "volume.h"
#include <array>
#include <cstdint>
#include <d3d11.h>
#include <vector>

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
    float default_distance = 600.0f; // set once in InitRaycaster; Space resets `distance` back to this
    float target_offset[3] = { 0.0f, 0.0f, 0.0f }; // right-drag pan, added to the volume-center look-at point

    // Axis lock (double-click a slice plane in the 3D panel): -1 = free
    // orbit via azimuth/elevation above; 0/1/2 = locked to rotate purely
    // around world X/Y/Z (sagittal/coronal/axial plane normal respectively),
    // parameterized by locked_angle instead. See ComputeCameraFrame.
    int locked_axis = -1;
    float locked_angle = 0.0f;

    // Camera right/up basis from the most recent RenderRaycast call, in world
    // mm -- the 3D panel's right-drag pan handler reads these (one frame
    // stale is fine) to move target_offset along the screen-space drag.
    float basis_right[3] = { 1.0f, 0.0f, 0.0f };
    float basis_up[3] = { 0.0f, 0.0f, 1.0f };

    // Optional label-volume overlays (ground truth / prediction), tinting
    // the raymarch wherever a structure is present -- see UploadLabelVolume.
    // Colors are pre-resolved per local label id (index = id - 1, same
    // convention as main.cpp's BuildLabelColors) so the shader needs no
    // canonical-name lookup, just a per-volume LUT indexed by the sampled id.
    static const int kMaxLabels = 16;
    ID3D11Texture3D* gt_label_tex = nullptr;
    ID3D11ShaderResourceView* gt_label_srv = nullptr;
    bool has_gt_labels = false;
    float gt_colors[kMaxLabels][4] = {};

    ID3D11Texture3D* pred_label_tex = nullptr;
    ID3D11ShaderResourceView* pred_label_srv = nullptr;
    bool has_pred_labels = false;
    float pred_colors[kMaxLabels][4] = {};

    ID3D11SamplerState* label_sampler = nullptr; // point-filtered: never interpolate between label ids
};

// Per-frame 3D-overlay state, mirroring the 2D panels' overlay controls
// (View > Structure overlays / Prediction overlay, their opacity sliders,
// and per-structure checkboxes packed into a bitmask) so toggling those also
// affects the "3D" panel consistently.
struct RaycastOverlayState
{
    bool gt_enabled = false;
    float gt_alpha = 0.45f;
    uint32_t gt_visible_mask = 0xFFFFFFFFu; // bit (id-1) set = that structure is visible

    bool pred_enabled = false;
    float pred_alpha = 0.45f;
    uint32_t pred_visible_mask = 0xFFFFFFFFu;
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

// Uploads a label volume (ground truth or prediction, per `is_prediction`)
// as a point-filtered Texture3D, and stores `colors` (indexed by local label
// id - 1, e.g. from main.cpp's BuildLabelColors) for the shader's LUT.
// Replaces any previously-uploaded texture for that slot (safe to call again
// after a fresh "Run Inference" reloads the prediction). Returns false
// (logging to stderr) on failure; `rc.has_gt_labels`/`has_pred_labels`
// reflects whether that slot currently has valid data.
bool UploadLabelVolume(ID3D11Device* device, Raycaster& rc, const LabelVolume& labels,
    const std::vector<std::array<uint8_t, 3>>& colors, bool is_prediction);

// Projects a world-space point into the raycast image's [-1,1] NDC-like
// space (same convention as the shader: +x right, +y up). Returns false
// (point behind the camera) if it shouldn't be drawn.
bool ProjectToNDC(const CameraFrame& cam, const float world_point[3], float out_ndc[2]);

// Ray-marches the volume from the current camera into `rc.target_tex`, using
// `window_width`/`window_level` (same HU units as the 2D slice windowing) as
// the opacity/color transfer function. Call once per frame while the 3D
// panel is visible; cheap enough to always re-render (no dirty tracking).
void RenderRaycast(ID3D11DeviceContext* context, Raycaster& rc, const Volume& volume, float window_width, float window_level,
    const RaycastOverlayState& overlay = RaycastOverlayState());

void ReleaseRaycaster(Raycaster& rc);
