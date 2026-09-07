// Minimal Dear ImGui + Win32 + DirectX 11 bootstrap.
// Modeled closely on imgui/examples/example_win32_directx11/main.cpp (docking branch).

#define NOMINMAX // windows.h's min/max macros shadow std::min/std::max otherwise
#include "imgui.h"
#include "imgui_internal.h" // DockBuilder* - forced 2x2 layout, built once on first frame
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include "implot.h"
#include "volume.h"
#include "raycaster.h"
#include "dvh.h"
#include "inference.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <vector>
#include <d3d11.h>
#include <tchar.h>

// Canonical structure name -> color, so the SAME structure always gets the
// SAME color whether it comes from the ground-truth masks (whose label list
// is alphabetical filenames, includes "brain", excludes "spinal_cord" - see
// source/data/phantom.py) or the model prediction (source/data/hanseg.py's
// LABEL_IDS order). Matching by name (not by list position/index) is what
// makes overlaying both meaningful for comparison. Unrecognized names fall
// back to the last color.
static const char* kCanonicalStructureNames[] = {
    "brainstem", "optic_chiasm", "optic_nerve_l", "optic_nerve_r", "lens_l",
    "lens_r", "eyeball_l", "eyeball_r", "spinal_cord", "brain",
};
static const uint8_t kCanonicalStructureColors[][3] = {
    { 230, 60, 60 },   { 60, 160, 230 },  { 250, 200, 60 },
    { 120, 220, 120 }, { 200, 100, 220 }, { 80, 220, 200 },
    { 240, 140, 60 },  { 160, 160, 250 }, { 250, 250, 120 }, { 200, 200, 200 },
};
static const int kNumCanonicalStructures = sizeof(kCanonicalStructureNames) / sizeof(kCanonicalStructureNames[0]);

// -1 if `name` isn't one of the canonical structures (e.g. an unexpected label).
static int CanonIndexForStructureName(const std::string& name)
{
    for (int i = 0; i < kNumCanonicalStructures; ++i)
        if (name == kCanonicalStructureNames[i])
            return i;
    return -1;
}

static const uint8_t* ColorForStructureName(const std::string& name)
{
    int i = CanonIndexForStructureName(name);
    return kCanonicalStructureColors[i >= 0 ? i : kNumCanonicalStructures - 1];
}

// Precomputed per-label-id (index = id - 1) colors for one label volume, so
// UploadSlice's hot loop does array lookups instead of string compares.
static std::vector<std::array<uint8_t, 3>> BuildLabelColors(const std::vector<std::string>& names)
{
    std::vector<std::array<uint8_t, 3>> colors(names.size());
    for (size_t i = 0; i < names.size(); ++i)
    {
        const uint8_t* c = ColorForStructureName(names[i]);
        colors[i] = { c[0], c[1], c[2] };
    }
    return colors;
}

// Packs a per-structure visibility vector into a bitmask (bit i = label id
// i+1), for the 3D raycaster's shader (see RaycastOverlayState) -- the 2D
// panels use the vector directly, but a per-frame constant buffer needs a
// fixed-size representation.
static uint32_t PackVisibleMask(const std::vector<bool>& visible)
{
    uint32_t mask = 0;
    for (size_t i = 0; i < visible.size() && i < 32; ++i)
        if (visible[i])
            mask |= (1u << i);
    return mask;
}

// One Dice score per canonical structure (index = CanonIndexForStructureName),
// NaN where that structure isn't present in both `gt` and `pred`'s label
// lists or (rare) their names don't map to a canonical structure. Powers the
// crosshair readout's "how good is the model here" line -- computed once
// (a full-volume pass) whenever both are loaded/reloaded, never per frame.
static std::vector<float> ComputePerStructureDice(const LabelVolume& gt, const LabelVolume& pred)
{
    std::vector<float> dice(kNumCanonicalStructures, std::numeric_limits<float>::quiet_NaN());
    if (gt.nx != pred.nx || gt.ny != pred.ny || gt.nz != pred.nz || gt.data.empty())
        return dice;

    std::vector<int> gt_id_to_canon(gt.labels.size() + 1, -1);   // index 0 (background) stays -1
    for (size_t i = 0; i < gt.labels.size(); ++i)
        gt_id_to_canon[i + 1] = CanonIndexForStructureName(gt.labels[i]);

    std::vector<int> pred_id_to_canon(pred.labels.size() + 1, -1);
    for (size_t i = 0; i < pred.labels.size(); ++i)
        pred_id_to_canon[i + 1] = CanonIndexForStructureName(pred.labels[i]);

    std::vector<size_t> gt_count(kNumCanonicalStructures, 0);
    std::vector<size_t> pred_count(kNumCanonicalStructures, 0);
    std::vector<size_t> intersection(kNumCanonicalStructures, 0);

    for (size_t v = 0; v < gt.data.size(); ++v)
    {
        int gt_canon = gt.data[v] < gt_id_to_canon.size() ? gt_id_to_canon[gt.data[v]] : -1;
        int pred_canon = pred.data[v] < pred_id_to_canon.size() ? pred_id_to_canon[pred.data[v]] : -1;
        if (gt_canon >= 0) gt_count[gt_canon]++;
        if (pred_canon >= 0) pred_count[pred_canon]++;
        if (gt_canon >= 0 && gt_canon == pred_canon) intersection[gt_canon]++;
    }

    for (int c = 0; c < kNumCanonicalStructures; ++c)
    {
        size_t denom = gt_count[c] + pred_count[c];
        if (denom > 0)
            dice[c] = 2.0f * (float)intersection[c] / (float)denom;
    }
    return dice;
}

static ID3D11Device* g_pd3dDevice = nullptr;
static ID3D11DeviceContext* g_pd3dDeviceContext = nullptr;
static IDXGISwapChain* g_pSwapChain = nullptr;
static bool g_SwapChainOccluded = false;
static UINT g_ResizeWidth = 0, g_ResizeHeight = 0;
static ID3D11RenderTargetView* g_mainRenderTargetView = nullptr;

bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Controls sidebar | Eclipse-style 2x2 MPR layout (axial | coronal
//                                                   sagittal | 3D)
// Built once (not every frame, or user drags get wiped). Layout is fixed for
// now (io.IniFilename == nullptr) rather than user-re-dockable + persisted.
static void BuildDockLayout(ImGuiID dockspace_id)
{
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

    ImGuiID sidebar, grid;
    ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.22f, &sidebar, &grid);

    ImGuiID top, bottom;
    ImGui::DockBuilderSplitNode(grid, ImGuiDir_Up, 0.5f, &top, &bottom);

    ImGuiID top_left, top_right;
    ImGui::DockBuilderSplitNode(top, ImGuiDir_Left, 0.5f, &top_left, &top_right);

    ImGuiID bottom_left, bottom_right;
    ImGui::DockBuilderSplitNode(bottom, ImGuiDir_Left, 0.5f, &bottom_left, &bottom_right);

    ImGui::DockBuilderDockWindow("Controls", sidebar);
    ImGui::DockBuilderDockWindow("Axial", top_left);
    ImGui::DockBuilderDockWindow("Coronal", top_right);
    ImGui::DockBuilderDockWindow("Sagittal", bottom_left);
    ImGui::DockBuilderDockWindow("3D", bottom_right);

    ImGui::DockBuilderFinish(dockspace_id);
}

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Axial slice texture: one RGBA8 DYNAMIC texture sized to the volume's
// (nx, ny), re-uploaded only when the slice index or W/L changes.
static bool CreateSliceTexture(int width, int height, ID3D11Texture2D** out_tex, ID3D11ShaderResourceView** out_srv)
{
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DYNAMIC;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(g_pd3dDevice->CreateTexture2D(&desc, nullptr, out_tex)))
        return false;

    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = desc.Format;
    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MipLevels = 1;
    if (FAILED(g_pd3dDevice->CreateShaderResourceView(*out_tex, &srv_desc, out_srv)))
        return false;

    return true;
}

// One overlay (ground truth masks, model prediction, ...) composited onto a
// slice's grayscale: `get_label(col, row)` returns a label id (0 = none);
// `colors`/`visible` are indexed by id-1 (see BuildLabelColors/LoadLabelVolume).
// Multiple layers blend in order, each std::function so UploadSlice can take
// a plain vector of them instead of an ever-growing template parameter pack.
struct OverlayLayer
{
    std::function<uint8_t(int, int)> get_label;
    bool enabled = false;
    float alpha = 0.45f;
    const std::vector<bool>* visible = nullptr;
    const std::vector<std::array<uint8_t, 3>>* colors = nullptr;
};

// Windows HU to a grayscale byte: (hu - (wl - ww/2)) / ww, clamped to [0,1],
// then composites each enabled overlay layer's structure color on top.
static void UploadSlice(ID3D11Texture2D* tex, int width, int height, float window_width, float window_level,
    const std::function<int16_t(int, int)>& get_voxel, const std::vector<OverlayLayer>& layers)
{
    D3D11_MAPPED_SUBRESOURCE mapped;
    if (FAILED(g_pd3dDeviceContext->Map(tex, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
        return;

    const float lo = window_level - window_width * 0.5f;
    for (int row = 0; row < height; ++row)
    {
        uint8_t* dst = (uint8_t*)mapped.pData + (size_t)row * mapped.RowPitch; // honour RowPitch, never assume width*bpp
        for (int col = 0; col < width; ++col)
        {
            float t = (get_voxel(col, row) - lo) / window_width;
            uint8_t gray = (uint8_t)(std::clamp(t, 0.0f, 1.0f) * 255.0f + 0.5f);
            float r = gray, g = gray, b = gray;

            for (const OverlayLayer& layer : layers)
            {
                if (!layer.enabled)
                    continue;
                uint8_t label = layer.get_label(col, row);
                if (label == 0)
                    continue;
                size_t li = (size_t)label - 1;
                bool visible = !layer.visible || (li < layer.visible->size() && (*layer.visible)[li]);
                if (!visible || !layer.colors || li >= layer.colors->size())
                    continue;
                const auto& c = (*layer.colors)[li];
                r = r * (1.0f - layer.alpha) + c[0] * layer.alpha;
                g = g * (1.0f - layer.alpha) + c[1] * layer.alpha;
                b = b * (1.0f - layer.alpha) + c[2] * layer.alpha;
            }

            dst[col * 4 + 0] = (uint8_t)r;
            dst[col * 4 + 1] = (uint8_t)g;
            dst[col * 4 + 2] = (uint8_t)b;
            dst[col * 4 + 3] = 255;
        }
    }
    g_pd3dDeviceContext->Unmap(tex, 0);
}

// Fits a (tex_w x tex_h) image with physical pixel spacing (sp_w, sp_h) into
// `avail`, preserving the real-world aspect ratio so anisotropic spacing
// (e.g. thicker slice pitch than in-plane pixel size) doesn't stretch it.
static ImVec2 FitImageSize(ImVec2 avail, int tex_w, int tex_h, double sp_w, double sp_h)
{
    float phys_w = (float)(tex_w * sp_w);
    float phys_h = (float)(tex_h * sp_h);
    float scale = std::min(avail.x / phys_w, avail.y / phys_h);
    if (!(scale > 0.0f))
        scale = 1.0f;
    return ImVec2(phys_w * scale, phys_h * scale);
}

// If the item just drawn (expected to be an ImGui::Image of a (tex_w x
// tex_h) slice texture, optionally vertically flipped per `flipped`) is
// being clicked/dragged, converts the mouse position to texture pixel
// coordinates. Returns false (leaving *out_col/*out_row untouched) otherwise.
static bool PanelClicked(ImVec2 item_min, ImVec2 item_size, int tex_w, int tex_h, bool flipped, int* out_col, int* out_row)
{
    if (!ImGui::IsItemHovered() || !ImGui::IsMouseDown(ImGuiMouseButton_Left))
        return false;

    ImVec2 mouse = ImGui::GetIO().MousePos;
    float fx = (mouse.x - item_min.x) / item_size.x;
    float fy = (mouse.y - item_min.y) / item_size.y;
    if (fx < 0.0f || fx > 1.0f || fy < 0.0f || fy > 1.0f)
        return false;

    int col = std::clamp((int)(fx * tex_w), 0, tex_w - 1);
    int display_row = std::clamp((int)(fy * tex_h), 0, tex_h - 1);
    *out_col = col;
    *out_row = flipped ? (tex_h - 1 - display_row) : display_row;
    return true;
}

// Ground-truth/prediction context for the crosshair readout's second line.
// Bundled into one struct rather than five parameters since all three MPR
// panels pass the exact same values, only the voxel coordinates differ.
struct StructureComparisonContext
{
    bool masks_loaded = false;
    const LabelVolume* label_volume = nullptr;      // ground truth
    bool prediction_loaded = false;
    const LabelVolume* prediction_volume = nullptr;
    const std::vector<float>* structure_dice = nullptr; // indexed by CanonIndexForStructureName
};

// Crosshair through the shared cursor voxel (ci, cj, ck), projected into
// this panel's (col, row) in-plane coordinates, plus an (i,j,k)/HU readout
// and, when ground truth and/or a prediction are loaded, a second line
// showing what each says is at the cursor and that structure's Dice score.
static void DrawCrosshairAndReadout(ImVec2 item_min, ImVec2 item_size, int tex_w, int tex_h, int col, int row, bool flipped,
    const Volume& vol, int ci, int cj, int ck, const StructureComparisonContext& cmp)
{
    int display_row = flipped ? (tex_h - 1 - row) : row;
    float sx = item_min.x + ((float)col + 0.5f) / tex_w * item_size.x;
    float sy = item_min.y + ((float)display_row + 0.5f) / tex_h * item_size.y;

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImU32 crosshair_color = IM_COL32(255, 230, 0, 180);
    draw_list->AddLine(ImVec2(sx, item_min.y), ImVec2(sx, item_min.y + item_size.y), crosshair_color);
    draw_list->AddLine(ImVec2(item_min.x, sy), ImVec2(item_min.x + item_size.x, sy), crosshair_color);

    // Drawn as an overlay on the image (not separate ImGui::Text lines) so it
    // can't push the panel's content past the dock quadrant's fixed height.
    char line1[64];
    snprintf(line1, sizeof(line1), "(i=%d, j=%d, k=%d)  HU=%d", ci, cj, ck, vol.At(ci, cj, ck));
    ImVec2 line1_size = ImGui::CalcTextSize(line1);

    char line2[128];
    bool have_line2 = cmp.masks_loaded || cmp.prediction_loaded;
    ImU32 line2_color = IM_COL32(220, 220, 220, 255);
    if (have_line2)
    {
        std::string gt_name = "-", pred_name = "-";
        int gt_canon = -1, pred_canon = -1;
        if (cmp.masks_loaded)
        {
            uint8_t id = cmp.label_volume->At(ci, cj, ck);
            gt_name = id == 0 ? "background" : cmp.label_volume->labels[id - 1];
            gt_canon = id == 0 ? -1 : CanonIndexForStructureName(gt_name);
        }
        if (cmp.prediction_loaded)
        {
            uint8_t id = cmp.prediction_volume->At(ci, cj, ck);
            pred_name = id == 0 ? "background" : cmp.prediction_volume->labels[id - 1];
            pred_canon = id == 0 ? -1 : CanonIndexForStructureName(pred_name);
        }

        if (!cmp.masks_loaded)
            line2_color = IM_COL32(220, 220, 220, 255); // prediction only, nothing to compare against
        else if (!cmp.prediction_loaded)
            line2_color = IM_COL32(220, 220, 220, 255); // ground truth only, ditto
        else if (gt_canon == -1 && pred_canon == -1)
            line2_color = IM_COL32(180, 180, 180, 255); // both background - not interesting either way
        else if (gt_canon == pred_canon)
            line2_color = IM_COL32(90, 220, 130, 255);  // agree on a real structure
        else
            line2_color = IM_COL32(230, 90, 90, 255);   // disagree (false positive/negative here)

        int relevant_canon = gt_canon >= 0 ? gt_canon : pred_canon;
        float dice_val = (relevant_canon >= 0 && cmp.structure_dice && relevant_canon < (int)cmp.structure_dice->size())
            ? (*cmp.structure_dice)[relevant_canon] : std::numeric_limits<float>::quiet_NaN();

        if (cmp.masks_loaded && cmp.prediction_loaded)
        {
            if (!std::isnan(dice_val))
                snprintf(line2, sizeof(line2), "GT: %s  Pred: %s  Dice: %.2f", gt_name.c_str(), pred_name.c_str(), dice_val);
            else
                snprintf(line2, sizeof(line2), "GT: %s  Pred: %s", gt_name.c_str(), pred_name.c_str());
        }
        else if (cmp.masks_loaded)
            snprintf(line2, sizeof(line2), "GT: %s", gt_name.c_str());
        else
            snprintf(line2, sizeof(line2), "Pred: %s", pred_name.c_str());
    }

    ImVec2 text_pos(item_min.x + 4.0f, item_min.y + 4.0f);
    ImVec2 box_size = line1_size;
    ImVec2 line2_size(0, 0);
    if (have_line2)
    {
        line2_size = ImGui::CalcTextSize(line2);
        box_size.x = std::max(box_size.x, line2_size.x);
        box_size.y += line2_size.y + 2.0f;
    }
    draw_list->AddRectFilled(text_pos, ImVec2(text_pos.x + box_size.x + 4.0f, text_pos.y + box_size.y + 4.0f), IM_COL32(0, 0, 0, 160));
    draw_list->AddText(ImVec2(text_pos.x + 2.0f, text_pos.y + 2.0f), IM_COL32(255, 255, 255, 255), line1);
    if (have_line2)
        draw_list->AddText(ImVec2(text_pos.x + 2.0f, text_pos.y + 2.0f + line1_size.y + 2.0f), line2_color, line2);
}

// Draws one slice plane's outline (+ a faint fill) into the 3D raycast
// image, projecting its 4 world-space corners through the same camera used
// to render that frame's raycast (see raycaster.h's CameraFrame) so the
// outline lines up with the volume exactly. Skips drawing if any corner
// falls behind the camera (cheap near-plane clip, fine for a UI overlay).
// Shared by DrawSlicePlaneOutline (rendering) and PointOnSlicePlane (double-
// click hit-testing) so both project the same 4 corners the same way.
// Returns false (leaving out_screen untouched) if any corner is behind the camera.
static bool ProjectPlaneCorners(ImVec2 item_min, ImVec2 item_size, const CameraFrame& cam, const float corners_world[4][3], ImVec2 out_screen[4])
{
    for (int c = 0; c < 4; ++c)
    {
        float ndc[2];
        if (!ProjectToNDC(cam, corners_world[c], ndc))
            return false;
        out_screen[c].x = item_min.x + (ndc[0] * 0.5f + 0.5f) * item_size.x;
        out_screen[c].y = item_min.y + (1.0f - (ndc[1] * 0.5f + 0.5f)) * item_size.y;
    }
    return true;
}

// Even-odd point-in-polygon test (works for our convex screen quads either winding order).
static bool PointInQuad(ImVec2 p, const ImVec2 quad[4])
{
    bool inside = false;
    for (int i = 0, j = 3; i < 4; j = i++)
    {
        bool crosses = ((quad[i].y > p.y) != (quad[j].y > p.y));
        if (crosses)
        {
            float x_at_p_y = quad[i].x + (p.y - quad[i].y) * (quad[j].x - quad[i].x) / (quad[j].y - quad[i].y);
            if (p.x < x_at_p_y)
                inside = !inside;
        }
    }
    return inside;
}

static void DrawSlicePlaneOutline(ImVec2 item_min, ImVec2 item_size, const CameraFrame& cam, const float corners_world[4][3], ImU32 color)
{
    ImVec2 screen[4];
    if (!ProjectPlaneCorners(item_min, item_size, cam, corners_world, screen))
        return;

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddConvexPolyFilled(screen, 4, (color & 0x00FFFFFF) | 0x20000000);
    draw_list->AddPolyline(screen, 4, color, ImDrawFlags_Closed, 2.0f);
}

int main(int argc, char** argv)
{
    // Optional argv[1]: a case directory under <repo>/data, e.g. "phantom"
    // (default) or "hanseg_cases/case_03" - see source/export_gui_volume.py.
    // The plan doc flagged this as the natural next step past a hardcoded
    // path: "plan a later argv[1] + file dialog."
    std::string case_dir = (argc > 1) ? argv[1] : "phantom";
    std::string case_data_dir = std::string(DICOM_RT_DATA_DIR) + "/" + case_dir + "/gui_export";

    std::wstring window_title = L"DICOM RT Viewer - ";
    for (char c : case_dir) window_title += (wchar_t)c; // case_dir is plain ASCII in practice

    WNDCLASSEXW wc = {
        sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L,
        GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr,
        L"DicomRtGuiWindowClass", nullptr
    };
    ::RegisterClassExW(&wc);
    HWND hwnd = ::CreateWindowW(
        wc.lpszClassName, window_title.c_str(), WS_OVERLAPPEDWINDOW,
        100, 100, 1280, 800, nullptr, nullptr, wc.hInstance, nullptr);

    if (!CreateDeviceD3D(hwnd))
    {
        CleanupDeviceD3D();
        ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    ::ShowWindow(hwnd, SW_SHOWDEFAULT);
    ::UpdateWindow(hwnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.IniFilename = nullptr; // layout is fixed/programmatic (BuildDockLayout); don't let imgui.ini override it

    ImGui::StyleColorsDark();
    ImPlot::CreateContext();

    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    bool dock_layout_built = false;
    bool needs_initial_focus = false; // set the frame after the layout is built, since "Axial" isn't a known window name until it's Begin()'d once
    ImVec4 clear_color = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);

    // Loads whichever case argv[1] selected (see source/export_gui_volume.py).
    // Orthogonal MPR only: slicing below assumes an axis-aligned volume
    // (identity direction matrix) and ignores Volume::direction entirely.
    // Oblique reformatting would need a resample through that matrix - deferred.
    Volume ct_volume;
    bool ct_loaded = LoadVolume(case_data_dir + "/ct", ct_volume);

    ID3D11Texture2D* axial_tex = nullptr;
    ID3D11ShaderResourceView* axial_srv = nullptr;
    ID3D11Texture2D* coronal_tex = nullptr;
    ID3D11ShaderResourceView* coronal_srv = nullptr;
    ID3D11Texture2D* sagittal_tex = nullptr;
    ID3D11ShaderResourceView* sagittal_srv = nullptr;

    // One shared world-space cursor, in voxel indices: cursor_k drives the
    // axial slice, cursor_j the coronal slice, cursor_i the sagittal slice.
    // Clicking in a panel updates the OTHER two (the in-plane coordinates),
    // which is what links the three views together.
    int cursor_i = 0, cursor_j = 0, cursor_k = 0;
    bool k_dirty = true, j_dirty = true, i_dirty = true;

    // Width/Level, per MPR panel (index via kAxial/kCoronal/kSagittal below).
    // Default ("global") mode keeps all three in sync on every change, same
    // as the single shared value this used to be; "G" flips to isolated mode
    // where each panel's W/L is independent. Slice is NOT part of this
    // toggle -- it's always per-panel, since the three planes' slice indices
    // are physically different quantities regardless of this setting.
    enum { kAxial = 0, kCoronal = 1, kSagittal = 2 };
    float window_width[3] = { 400.0f, 400.0f, 400.0f };
    float window_level[3] = { 40.0f, 40.0f, 40.0f };
    bool global_wl_mode = true; // "G" toggles
    bool* const wl_dirty[3] = { &k_dirty, &j_dirty, &i_dirty }; // panel index -> its own texture's dirty flag

    // Sets panel `p`'s W/L; in global mode, propagates to all three panels
    // (and dirties all three) rather than just `p`.
    auto SetWindowLevel = [&](int p, float ww, float wl)
    {
        if (global_wl_mode)
        {
            for (int i = 0; i < 3; ++i) { window_width[i] = ww; window_level[i] = wl; *wl_dirty[i] = true; }
        }
        else
        {
            window_width[p] = ww; window_level[p] = wl; *wl_dirty[p] = true;
        }
    };

    // Panel focus (click anywhere in a panel, or Q/E to cycle) + which
    // parameter Shift+S/W/L selected as the target of A/D and the mouse
    // wheel (when not Ctrl-zooming). The focused panel's active parameter is
    // highlighted in the sidebar so it's clear what's about to change.
    enum class PanelId { Axial, Coronal, Sagittal, ThreeD };
    enum class ParamFocus { Slice, Width, Level };
    PanelId focused_panel = PanelId::Axial;
    ParamFocus active_param = ParamFocus::Slice;
    float param_step = 50.0f; // Shift+A/Shift+D step for the active parameter; user-adjustable in the sidebar
    const char* kPanelWindowNames[3] = { "Axial", "Coronal", "Sagittal" }; // Q/E cycle these; 3D keeps its own controls

    // OAR mask/contour overlay (source/export_gui_volume.py --export-masks).
    // Optional: the MPR panels just show plain CT if this fails to load.
    LabelVolume label_volume;
    bool masks_loaded = false;
    bool show_overlays = true;   // "View > Structure overlays" toggle
    float overlay_alpha = 0.45f;
    std::vector<bool> label_visible;
    std::vector<std::array<uint8_t, 3>> label_colors;

    // Model prediction overlay (source/run_inference_for_gui.py, launched by
    // the "Run Inference" button below). Independent from the ground-truth
    // layer above so both can be shown/compared at once; same color per
    // structure name (ColorForStructureName), different toggle/alpha/checkboxes.
    LabelVolume prediction_volume;
    bool prediction_loaded = false;
    bool show_prediction_overlay = false;
    float prediction_overlay_alpha = 0.45f;
    std::vector<bool> prediction_label_visible;
    std::vector<std::array<uint8_t, 3>> prediction_label_colors;
    InferenceJob inference_job;

    // Per-structure Dice (ground truth vs. prediction), for the crosshair
    // readout's accuracy line. Recomputed (full-volume pass, so not every
    // frame) whenever both are loaded/reloaded.
    std::vector<float> structure_dice;

    // Raycast 3D panel.
    Raycaster raycaster;
    bool raycaster_ok = false;
    bool show_slice_planes = true; // axial/coronal/sagittal cut-plane outlines, at the shared cursor

    // DVH. "View > DVH" toggles the window's visibility. Prefers a real dose
    // export (source/export_gui_volume.py --dose <RTDOSE.dcm>) if present at
    // gui_export/dose(.bin/.json); falls back to a synthetic Gaussian blob
    // (see the session decision: no RTDOSE exists for the phantom case) so
    // the DVH plot still has something to show end-to-end.
    std::vector<DVHCurve> dvh_curves;
    bool show_dvh = false;
    bool dvh_needs_repositioning = true; // forces a sane pos/size the first time (and after toggled back on)
    bool dvh_dose_is_synthetic = true;

    if (ct_loaded)
    {
        cursor_i = ct_volume.nx / 2;
        cursor_j = ct_volume.ny / 2;
        cursor_k = ct_volume.nz / 2;
        auto [min_it, max_it] = std::minmax_element(ct_volume.data.begin(), ct_volume.data.end());
        printf("Loaded CT volume: %dx%dx%d spacing=(%.3f,%.3f,%.3f) HU range=[%d,%d]\n",
            ct_volume.nx, ct_volume.ny, ct_volume.nz,
            ct_volume.spacing[0], ct_volume.spacing[1], ct_volume.spacing[2],
            (int)*min_it, (int)*max_it);

        if (!CreateSliceTexture(ct_volume.nx, ct_volume.ny, &axial_tex, &axial_srv) ||
            !CreateSliceTexture(ct_volume.nx, ct_volume.nz, &coronal_tex, &coronal_srv) ||
            !CreateSliceTexture(ct_volume.ny, ct_volume.nz, &sagittal_tex, &sagittal_srv))
        {
            fprintf(stderr, "Failed to create slice textures\n");
            ct_loaded = false;
        }

        raycaster_ok = InitRaycaster(g_pd3dDevice, ct_volume, raycaster);
        if (!raycaster_ok)
            fprintf(stderr, "Failed to initialize the GPU raycaster; 3D panel will be unavailable\n");

        masks_loaded = LoadLabelVolume(case_data_dir + "/masks", label_volume);
        if (masks_loaded && (label_volume.nx != ct_volume.nx || label_volume.ny != ct_volume.ny || label_volume.nz != ct_volume.nz))
        {
            fprintf(stderr, "Mask volume shape %dx%dx%d doesn't match CT shape %dx%dx%d; disabling overlays\n",
                label_volume.nx, label_volume.ny, label_volume.nz, ct_volume.nx, ct_volume.ny, ct_volume.nz);
            masks_loaded = false;
        }
        if (masks_loaded)
        {
            label_visible.assign(label_volume.labels.size(), true);
            label_colors = BuildLabelColors(label_volume.labels);
            if (raycaster_ok)
                UploadLabelVolume(g_pd3dDevice, raycaster, label_volume, label_colors, false);

            DoseVolume dose_volume;
            bool real_dose_loaded = LoadDoseVolume(case_data_dir + "/dose", dose_volume);
            if (real_dose_loaded && (dose_volume.nx != ct_volume.nx || dose_volume.ny != ct_volume.ny || dose_volume.nz != ct_volume.nz))
            {
                fprintf(stderr, "Dose volume shape %dx%dx%d doesn't match CT shape %dx%dx%d; ignoring it\n",
                    dose_volume.nx, dose_volume.ny, dose_volume.nz, ct_volume.nx, ct_volume.ny, ct_volume.nz);
                real_dose_loaded = false;
            }

            if (real_dose_loaded)
            {
                dvh_dose_is_synthetic = false;
                dvh_curves = ComputeDVHCurves(dose_volume.data, label_volume);
            }
            else
            {
                dvh_dose_is_synthetic = true;
                dvh_curves = ComputeDVHCurves(GenerateSyntheticDose(ct_volume), label_volume);
            }
        }
        else
        {
            fprintf(stderr, "Failed to load mask volume (see LoadLabelVolume errors above); run "
                             "source/export_gui_volume.py --export-masks first. Overlays/DVH disabled.\n");
        }

        // A prediction from a previous "Run Inference" click may already be on disk.
        prediction_loaded = LoadLabelVolume(case_data_dir + "/prediction", prediction_volume);
        if (prediction_loaded && (prediction_volume.nx != ct_volume.nx || prediction_volume.ny != ct_volume.ny || prediction_volume.nz != ct_volume.nz))
        {
            fprintf(stderr, "Prediction volume shape %dx%dx%d doesn't match CT shape %dx%dx%d; ignoring it\n",
                prediction_volume.nx, prediction_volume.ny, prediction_volume.nz, ct_volume.nx, ct_volume.ny, ct_volume.nz);
            prediction_loaded = false;
        }
        if (prediction_loaded)
        {
            prediction_label_visible.assign(prediction_volume.labels.size(), true);
            prediction_label_colors = BuildLabelColors(prediction_volume.labels);
            show_prediction_overlay = true; // matches show_overlays' always-on-when-loaded default
            if (raycaster_ok)
                UploadLabelVolume(g_pd3dDevice, raycaster, prediction_volume, prediction_label_colors, true);
        }
        if (masks_loaded && prediction_loaded)
            structure_dice = ComputePerStructureDice(label_volume, prediction_volume);
    }
    else
    {
        fprintf(stderr, "Failed to load CT volume (see LoadVolume errors above); run source/export_gui_volume.py first\n");
    }

    bool done = false;
    while (!done)
    {
        MSG msg;
        while (::PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE))
        {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT)
                done = true;
        }
        if (done)
            break;

        if (g_SwapChainOccluded && g_pSwapChain->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED)
        {
            ::Sleep(10);
            continue;
        }
        g_SwapChainOccluded = false;

        if (g_ResizeWidth != 0 && g_ResizeHeight != 0)
        {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, g_ResizeWidth, g_ResizeHeight, DXGI_FORMAT_UNKNOWN, 0);
            g_ResizeWidth = g_ResizeHeight = 0;
            CreateRenderTarget();
        }

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        ImGuiID dockspace_id = ImGui::DockSpaceOverViewport(0, nullptr, ImGuiDockNodeFlags_PassthruCentralNode);
        if (!dock_layout_built)
        {
            BuildDockLayout(dockspace_id);
            dock_layout_built = true;
            needs_initial_focus = true;
        }
        else if (needs_initial_focus)
        {
            ImGui::SetWindowFocus("Axial"); // otherwise the last-Begin()'d panel (3D) gets implicit initial focus
            needs_initial_focus = false;
        }

        // If a "Run Inference" job finished this frame, reload its output.
        if (PollInferenceJob(inference_job))
        {
            if (inference_job.exit_code == 0)
            {
                prediction_loaded = LoadLabelVolume(case_data_dir + "/prediction", prediction_volume);
                if (prediction_loaded && (prediction_volume.nx != ct_volume.nx || prediction_volume.ny != ct_volume.ny || prediction_volume.nz != ct_volume.nz))
                {
                    fprintf(stderr, "Prediction volume shape %dx%dx%d doesn't match CT shape %dx%dx%d; ignoring it\n",
                        prediction_volume.nx, prediction_volume.ny, prediction_volume.nz, ct_volume.nx, ct_volume.ny, ct_volume.nz);
                    prediction_loaded = false;
                }
                if (prediction_loaded)
                {
                    prediction_label_visible.assign(prediction_volume.labels.size(), true);
                    prediction_label_colors = BuildLabelColors(prediction_volume.labels);
                    show_prediction_overlay = true;
                    k_dirty = j_dirty = i_dirty = true;
                    if (raycaster_ok)
                        UploadLabelVolume(g_pd3dDevice, raycaster, prediction_volume, prediction_label_colors, true);
                }
                if (masks_loaded && prediction_loaded)
                    structure_dice = ComputePerStructureDice(label_volume, prediction_volume);
            }
            else
            {
                fprintf(stderr, "Inference job exited with code %lu; see its console output above\n", inference_job.exit_code);
            }
        }

        bool overlay_toggled_from_menu = false;
        if (ImGui::BeginMainMenuBar())
        {
            if (ImGui::BeginMenu("File"))
            {
                if (ImGui::MenuItem("Exit"))
                    done = true;
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View"))
            {
                if (ImGui::MenuItem("Structure overlays", nullptr, &show_overlays, masks_loaded))
                    overlay_toggled_from_menu = true;
                if (ImGui::MenuItem("Prediction overlay", nullptr, &show_prediction_overlay, prediction_loaded))
                    overlay_toggled_from_menu = true;
                if (ImGui::MenuItem("DVH", nullptr, &show_dvh, masks_loaded) && show_dvh)
                    dvh_needs_repositioning = true; // snap it back into the main window, wherever it drifted to before
                ImGui::Separator();
                if (ImGui::MenuItem("Reset Layout"))
                    dock_layout_built = false; // rebuilds Controls/Axial/Coronal/Sagittal/3D at their default proportions next frame
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }
        if (overlay_toggled_from_menu)
            k_dirty = j_dirty = i_dirty = true;

        // Global keyboard shortcuts: Q/E cycle which MPR panel is focused;
        // Shift+S/W/L pick which parameter A/D and the mouse wheel adjust;
        // G toggles Width/Level between synced-across-all-three and
        // per-panel-isolated. GetKeyState for Ctrl/Shift (not io.KeyCtrl/
        // KeyShift) for the same reason as the zoom fix: those only refresh
        // on a fresh WM_KEYDOWN/UP reaching this window, not the live state.
        if (ct_loaded)
        {
            if (ImGui::IsKeyPressed(ImGuiKey_Q, false))
            {
                int cur = (focused_panel == PanelId::ThreeD) ? 0 : (int)focused_panel;
                focused_panel = (PanelId)((cur + 2) % 3);
                ImGui::SetWindowFocus(kPanelWindowNames[(int)focused_panel]);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_E, false))
            {
                int cur = (focused_panel == PanelId::ThreeD) ? 0 : (int)focused_panel;
                focused_panel = (PanelId)((cur + 1) % 3);
                ImGui::SetWindowFocus(kPanelWindowNames[(int)focused_panel]);
            }

            bool shift_held = (::GetKeyState(VK_SHIFT) & 0x8000) != 0;
            if (shift_held && ImGui::IsKeyPressed(ImGuiKey_S, false)) active_param = ParamFocus::Slice;
            if (shift_held && ImGui::IsKeyPressed(ImGuiKey_W, false)) active_param = ParamFocus::Width;
            if (shift_held && ImGui::IsKeyPressed(ImGuiKey_L, false)) active_param = ParamFocus::Level;

            if (ImGui::IsKeyPressed(ImGuiKey_G, false))
            {
                global_wl_mode = !global_wl_mode;
                if (global_wl_mode)
                {
                    int p = (focused_panel == PanelId::ThreeD) ? (int)kAxial : (int)focused_panel;
                    SetWindowLevel(p, window_width[p], window_level[p]); // propagate focused panel's W/L to all three
                }
            }

            // Shift+A/Shift+D adjust whichever parameter is active, by
            // param_step, for the focused panel. Wheel-based adjustment was
            // dropped (hover/focus detection over it proved unreliable);
            // this is the sole adjustment control now besides the sidebar sliders.
            if (focused_panel != PanelId::ThreeD)
            {
                int p = (int)focused_panel;
                float delta = 0.0f;
                if (shift_held && ImGui::IsKeyPressed(ImGuiKey_A)) delta = -param_step;
                if (shift_held && ImGui::IsKeyPressed(ImGuiKey_D)) delta = param_step;
                if (delta != 0.0f)
                {
                    if (active_param == ParamFocus::Slice)
                    {
                        int* cursor_ptr = (p == kAxial) ? &cursor_k : (p == kCoronal) ? &cursor_j : &cursor_i;
                        int max_val = (p == kAxial) ? ct_volume.nz - 1 : (p == kCoronal) ? ct_volume.ny - 1 : ct_volume.nx - 1;
                        *cursor_ptr = std::clamp(*cursor_ptr + (int)delta, 0, max_val);
                        *wl_dirty[p] = true;
                    }
                    else if (active_param == ParamFocus::Width)
                        SetWindowLevel(p, std::clamp(window_width[p] + delta, 1.0f, 4000.0f), window_level[p]);
                    else
                        SetWindowLevel(p, window_width[p], std::clamp(window_level[p] + delta, -1000.0f, 1000.0f));
                }
            }
        }

        StructureComparisonContext cmp;
        cmp.masks_loaded = masks_loaded;
        cmp.label_volume = &label_volume;
        cmp.prediction_loaded = prediction_loaded;
        cmp.prediction_volume = &prediction_volume;
        cmp.structure_dice = &structure_dice;

        ImGui::Begin("Controls");
        if (ct_loaded)
        {
            static const char* kPanelLabel[4] = { "Axial", "Coronal", "Sagittal", "3D" };
            ImGui::Text("Focused panel: %s  (Q/E to cycle)", kPanelLabel[(int)focused_panel]);
            ImGui::Text("W/L mode: %s  (G to toggle)", global_wl_mode ? "Global (shared)" : "Isolated (per-panel)");
            ImGui::Separator();

            int active_p = (focused_panel == PanelId::ThreeD) ? (int)PanelId::Axial : (int)focused_panel;
            auto PushHighlight = [&](bool on) { if (on) ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.35f, 0.55f, 0.30f, 1.0f)); };
            auto PopHighlight = [&](bool on) { if (on) ImGui::PopStyleColor(); };

            if (focused_panel != PanelId::ThreeD)
            {
                int* cursor_ptr = (active_p == kAxial) ? &cursor_k : (active_p == kCoronal) ? &cursor_j : &cursor_i;
                int max_val = (active_p == kAxial) ? ct_volume.nz - 1 : (active_p == kCoronal) ? ct_volume.ny - 1 : ct_volume.nx - 1;
                bool on = active_param == ParamFocus::Slice;
                PushHighlight(on);
                ImGui::SetNextItemWidth(120.0f);
                if (ImGui::SliderInt("Slice", cursor_ptr, 0, max_val))
                    *wl_dirty[active_p] = true;
                PopHighlight(on);
            }
            else
            {
                ImGui::TextDisabled("Slice - N/A for 3D (drag to orbit)");
            }

            {
                bool on = active_param == ParamFocus::Width;
                PushHighlight(on);
                ImGui::SetNextItemWidth(120.0f);
                float ww = window_width[active_p];
                if (ImGui::SliderFloat("Width", &ww, 1.0f, 4000.0f, "%.0f"))
                    SetWindowLevel(active_p, ww, window_level[active_p]);
                PopHighlight(on);
            }
            {
                bool on = active_param == ParamFocus::Level;
                PushHighlight(on);
                ImGui::SetNextItemWidth(120.0f);
                float wl = window_level[active_p];
                if (ImGui::SliderFloat("Level", &wl, -1000.0f, 1000.0f, "%.0f"))
                    SetWindowLevel(active_p, window_width[active_p], wl);
                PopHighlight(on);
            }

            if (ImGui::Button("Soft 400/40")) SetWindowLevel(active_p, 400, 40);
            ImGui::SameLine();
            if (ImGui::Button("Lung 1500/-600")) SetWindowLevel(active_p, 1500, -600);
            if (ImGui::Button("Bone 2000/500")) SetWindowLevel(active_p, 2000, 500);
            ImGui::SameLine();
            if (ImGui::Button("Brain 80/40")) SetWindowLevel(active_p, 80, 40);

            ImGui::SetNextItemWidth(120.0f);
            ImGui::SliderFloat("Shift+A/D step", &param_step, 1.0f, 200.0f, "%.0f");

            ImGui::Separator();
            ImGui::TextWrapped(
                "Click a panel (or Q/E) to focus it. Shift+S/W/L picks Slice/"
                "Width/Level (highlighted above); Shift+A/Shift+D adjusts it "
                "by the step above. G toggles Width/Level between "
                "shared-across-all-three and per-panel.");
            ImGui::Separator();

            if (masks_loaded && ImGui::CollapsingHeader("Structures (ground truth)"))
            {
                bool overlay_ui_changed = ImGui::SliderFloat("Overlay opacity##gt", &overlay_alpha, 0.0f, 1.0f, "%.2f");
                for (size_t li = 0; li < label_volume.labels.size(); ++li)
                {
                    ImGui::PushID((int)li);
                    const auto& c = label_colors[li];
                    ImGui::ColorButton("##swatch", ImVec4(c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f, 1.0f),
                        ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoPicker, ImVec2(12, 12));
                    ImGui::SameLine();
                    bool visible = label_visible[li];
                    if (ImGui::Checkbox(label_volume.labels[li].c_str(), &visible))
                    {
                        label_visible[li] = visible;
                        overlay_ui_changed = true;
                    }
                    ImGui::PopID();
                }
                if (overlay_ui_changed)
                    k_dirty = j_dirty = i_dirty = true;
            }

            if (ImGui::CollapsingHeader("Prediction"))
            {
                ImGui::BeginDisabled(inference_job.running);
                if (ImGui::Button("Run Inference"))
                    StartInferenceJob(inference_job, DICOM_RT_REPO_DIR, case_dir);
                ImGui::EndDisabled();
                if (inference_job.running)
                {
                    ImGui::Text("Running... (%.0fs) - nnU-Net 4-fold ensemble, see console for progress", (GetTickCount() - inference_job.start_tick_ms) / 1000.0f);
                }
                else if (!prediction_loaded)
                {
                    ImGui::TextUnformatted("No prediction loaded yet.");
                }

                if (prediction_loaded)
                {
                    bool overlay_ui_changed = ImGui::SliderFloat("Overlay opacity##pred", &prediction_overlay_alpha, 0.0f, 1.0f, "%.2f");
                    for (size_t li = 0; li < prediction_volume.labels.size(); ++li)
                    {
                        ImGui::PushID((int)(li + 1000));
                        const auto& c = prediction_label_colors[li];
                        ImGui::ColorButton("##swatch", ImVec4(c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f, 1.0f),
                            ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoPicker, ImVec2(12, 12));
                        ImGui::SameLine();
                        bool visible = prediction_label_visible[li];
                        if (ImGui::Checkbox(prediction_volume.labels[li].c_str(), &visible))
                        {
                            prediction_label_visible[li] = visible;
                            overlay_ui_changed = true;
                        }
                        ImGui::PopID();
                    }
                    if (overlay_ui_changed)
                        k_dirty = j_dirty = i_dirty = true;
                }
            }
        }
        else
        {
            ImGui::TextUnformatted("No CT loaded, see console");
        }
        ImGui::End();

        ImGui::Begin("Axial");
        if (ImGui::IsWindowFocused()) focused_panel = PanelId::Axial;
        if (ct_loaded)
        {
            if (k_dirty)
            {
                std::vector<OverlayLayer> layers = {
                    { [&](int i, int j) { return masks_loaded ? label_volume.At(i, j, cursor_k) : (uint8_t)0; },
                      show_overlays, overlay_alpha, &label_visible, &label_colors },
                    { [&](int i, int j) { return prediction_loaded ? prediction_volume.At(i, j, cursor_k) : (uint8_t)0; },
                      show_prediction_overlay, prediction_overlay_alpha, &prediction_label_visible, &prediction_label_colors },
                };
                UploadSlice(axial_tex, ct_volume.nx, ct_volume.ny, window_width[kAxial], window_level[kAxial],
                    [&](int i, int j) { return ct_volume.At(i, j, cursor_k); }, layers);
                k_dirty = false;
            }

            ImVec2 size = FitImageSize(ImGui::GetContentRegionAvail(), ct_volume.nx, ct_volume.ny, ct_volume.spacing[0], ct_volume.spacing[1]);
            ImGui::Image((ImTextureID)(intptr_t)axial_srv, size);

            ImVec2 item_min = ImGui::GetItemRectMin();
            int click_i, click_j;
            if (PanelClicked(item_min, size, ct_volume.nx, ct_volume.ny, false, &click_i, &click_j))
            {
                cursor_i = click_i; cursor_j = click_j;
                j_dirty = i_dirty = true; // coronal/sagittal now pass through a different point
            }
            DrawCrosshairAndReadout(item_min, size, ct_volume.nx, ct_volume.ny, cursor_i, cursor_j, false,
                ct_volume, cursor_i, cursor_j, cursor_k, cmp);
        }
        else
        {
            ImGui::TextUnformatted("Axial (transversal) - failed to load CT volume, see console");
        }
        ImGui::End();

        ImGui::Begin("Coronal");
        if (ImGui::IsWindowFocused()) focused_panel = PanelId::Coronal;
        if (ct_loaded)
        {
            if (j_dirty)
            {
                // width=x, height=z; row 0 = k=0 (inferior) -- flip via UV below so superior is on top.
                std::vector<OverlayLayer> layers = {
                    { [&](int i, int k) { return masks_loaded ? label_volume.At(i, cursor_j, k) : (uint8_t)0; },
                      show_overlays, overlay_alpha, &label_visible, &label_colors },
                    { [&](int i, int k) { return prediction_loaded ? prediction_volume.At(i, cursor_j, k) : (uint8_t)0; },
                      show_prediction_overlay, prediction_overlay_alpha, &prediction_label_visible, &prediction_label_colors },
                };
                UploadSlice(coronal_tex, ct_volume.nx, ct_volume.nz, window_width[kCoronal], window_level[kCoronal],
                    [&](int i, int k) { return ct_volume.At(i, cursor_j, k); }, layers);
                j_dirty = false;
            }

            ImVec2 size = FitImageSize(ImGui::GetContentRegionAvail(), ct_volume.nx, ct_volume.nz, ct_volume.spacing[0], ct_volume.spacing[2]);
            ImGui::Image((ImTextureID)(intptr_t)coronal_srv, size, ImVec2(0, 1), ImVec2(1, 0));

            ImVec2 item_min = ImGui::GetItemRectMin();
            int click_i, click_k;
            if (PanelClicked(item_min, size, ct_volume.nx, ct_volume.nz, true, &click_i, &click_k))
            {
                cursor_i = click_i; cursor_k = click_k;
                i_dirty = k_dirty = true;
            }
            DrawCrosshairAndReadout(item_min, size, ct_volume.nx, ct_volume.nz, cursor_i, cursor_k, true,
                ct_volume, cursor_i, cursor_j, cursor_k, cmp);
        }
        else
        {
            ImGui::TextUnformatted("Coronal (frontal) - failed to load CT volume, see console");
        }
        ImGui::End();

        ImGui::Begin("Sagittal");
        if (ImGui::IsWindowFocused()) focused_panel = PanelId::Sagittal;
        if (ct_loaded)
        {
            if (i_dirty)
            {
                // width=y, height=z; same vertical flip as coronal (superior on top).
                std::vector<OverlayLayer> layers = {
                    { [&](int j, int k) { return masks_loaded ? label_volume.At(cursor_i, j, k) : (uint8_t)0; },
                      show_overlays, overlay_alpha, &label_visible, &label_colors },
                    { [&](int j, int k) { return prediction_loaded ? prediction_volume.At(cursor_i, j, k) : (uint8_t)0; },
                      show_prediction_overlay, prediction_overlay_alpha, &prediction_label_visible, &prediction_label_colors },
                };
                UploadSlice(sagittal_tex, ct_volume.ny, ct_volume.nz, window_width[kSagittal], window_level[kSagittal],
                    [&](int j, int k) { return ct_volume.At(cursor_i, j, k); }, layers);
                i_dirty = false;
            }

            ImVec2 size = FitImageSize(ImGui::GetContentRegionAvail(), ct_volume.ny, ct_volume.nz, ct_volume.spacing[1], ct_volume.spacing[2]);
            ImGui::Image((ImTextureID)(intptr_t)sagittal_srv, size, ImVec2(0, 1), ImVec2(1, 0));

            ImVec2 item_min = ImGui::GetItemRectMin();
            int click_j, click_k;
            if (PanelClicked(item_min, size, ct_volume.ny, ct_volume.nz, true, &click_j, &click_k))
            {
                cursor_j = click_j; cursor_k = click_k;
                j_dirty = k_dirty = true;
            }
            DrawCrosshairAndReadout(item_min, size, ct_volume.ny, ct_volume.nz, cursor_j, cursor_k, true,
                ct_volume, cursor_i, cursor_j, cursor_k, cmp);
        }
        else
        {
            ImGui::TextUnformatted("Sagittal - failed to load CT volume, see console");
        }
        ImGui::End();

        ImGui::Begin("3D");
        if (ImGui::IsWindowFocused()) focused_panel = PanelId::ThreeD;
        if (ct_loaded && raycaster_ok)
        {
            // Space: full reset (also the way out of axis-lock mode).
            if (ImGui::IsWindowHovered() && ImGui::IsKeyPressed(ImGuiKey_Space, false))
            {
                raycaster.locked_axis = -1;
                raycaster.azimuth = 0.7f;
                raycaster.elevation = 0.35f;
                raycaster.distance = raycaster.default_distance;
                raycaster.target_offset[0] = raycaster.target_offset[1] = raycaster.target_offset[2] = 0.0f;
            }

            if (ImGui::IsWindowHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Left))
            {
                if (raycaster.locked_axis < 0)
                {
                    raycaster.azimuth += io.MouseDelta.x * 0.01f;
                    raycaster.elevation = std::clamp(raycaster.elevation - io.MouseDelta.y * 0.01f, -1.5f, 1.5f);
                }
                else
                {
                    // One rotational DOF while locked: spin around the locked axis only.
                    raycaster.locked_angle += io.MouseDelta.x * 0.01f;
                }
            }
            if (ImGui::IsWindowHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Right))
            {
                // Pan the orbit target along the camera's own right/up (from last frame's
                // render), scaled by distance so the pan speed feels consistent when zoomed.
                float pan_scale = raycaster.distance * 0.0015f;
                for (int a = 0; a < 3; ++a)
                {
                    raycaster.target_offset[a] -= raycaster.basis_right[a] * io.MouseDelta.x * pan_scale;
                    raycaster.target_offset[a] += raycaster.basis_up[a] * io.MouseDelta.y * pan_scale;
                }
            }
            if (ImGui::IsWindowHovered() && io.MouseWheel != 0.0f)
                raycaster.distance = std::clamp(raycaster.distance * powf(0.9f, io.MouseWheel), 50.0f, 5000.0f);

            RaycastOverlayState raycast_overlay;
            raycast_overlay.gt_enabled = show_overlays;
            raycast_overlay.gt_alpha = overlay_alpha;
            raycast_overlay.gt_visible_mask = PackVisibleMask(label_visible);
            raycast_overlay.pred_enabled = show_prediction_overlay;
            raycast_overlay.pred_alpha = prediction_overlay_alpha;
            raycast_overlay.pred_visible_mask = PackVisibleMask(prediction_label_visible);
            RenderRaycast(g_pd3dDeviceContext, raycaster, ct_volume, window_width[kAxial], window_level[kAxial], raycast_overlay);

            // Standard anatomical viewpoints (3D Slicer convention): also clears any axis lock.
            auto StandardView = [&](float az, float elev)
            {
                raycaster.locked_axis = -1;
                raycaster.azimuth = az;
                raycaster.elevation = elev;
                raycaster.target_offset[0] = raycaster.target_offset[1] = raycaster.target_offset[2] = 0.0f;
            };
            const float kHalfPi = 1.57079632679f;
            if (ImGui::Button("A")) StandardView(-kHalfPi, 0.0f);
            ImGui::SameLine();
            if (ImGui::Button("P")) StandardView(kHalfPi, 0.0f);
            ImGui::SameLine();
            if (ImGui::Button("L")) StandardView(0.0f, 0.0f);
            ImGui::SameLine();
            if (ImGui::Button("R")) StandardView(3.14159265f, 0.0f);
            ImGui::SameLine();
            if (ImGui::Button("S")) StandardView(0.0f, kHalfPi);
            ImGui::SameLine();
            if (ImGui::Button("I")) StandardView(0.0f, -kHalfPi);
            ImGui::SameLine();
            ImGui::TextDisabled("(view)");

            ImGui::Checkbox("Show slice planes", &show_slice_planes);
            ImGui::SameLine();
            if (raycaster.locked_axis < 0)
                ImGui::TextUnformatted("(drag to orbit, right-drag to pan, wheel to zoom, dbl-click a plane to lock to it)");
            else
                ImGui::TextUnformatted("(locked to one axis - drag to spin, Space to reset)");
            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImVec2 size = FitImageSize(avail, raycaster.target_size, raycaster.target_size, 1.0, 1.0);
            ImGui::Image((ImTextureID)(intptr_t)raycaster.target_srv, size);

            if (show_slice_planes)
            {
                ImVec2 item_min = ImGui::GetItemRectMin();
                CameraFrame cam = ComputeCameraFrame(raycaster, ct_volume);

                float xmin = (float)ct_volume.origin[0], xmax = (float)(ct_volume.origin[0] + ct_volume.nx * ct_volume.spacing[0]);
                float ymin = (float)ct_volume.origin[1], ymax = (float)(ct_volume.origin[1] + ct_volume.ny * ct_volume.spacing[1]);
                float zmin = (float)ct_volume.origin[2], zmax = (float)(ct_volume.origin[2] + ct_volume.nz * ct_volume.spacing[2]);
                float xk = (float)(ct_volume.origin[0] + (cursor_i + 0.5) * ct_volume.spacing[0]);
                float yk = (float)(ct_volume.origin[1] + (cursor_j + 0.5) * ct_volume.spacing[1]);
                float zk = (float)(ct_volume.origin[2] + (cursor_k + 0.5) * ct_volume.spacing[2]);

                // Slicer's color convention: axial=red, coronal=yellow, sagittal=green.
                // locked_axis convention (see raycaster.h): 0=X(sagittal normal), 1=Y(coronal normal), 2=Z(axial normal).
                float axial_corners[4][3] = { { xmin, ymin, zk }, { xmax, ymin, zk }, { xmax, ymax, zk }, { xmin, ymax, zk } };
                float coronal_corners[4][3] = { { xmin, yk, zmin }, { xmax, yk, zmin }, { xmax, yk, zmax }, { xmin, yk, zmax } };
                float sagittal_corners[4][3] = { { xk, ymin, zmin }, { xk, ymax, zmin }, { xk, ymax, zmax }, { xk, ymin, zmax } };

                bool double_clicked = ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);
                bool was_free = raycaster.locked_axis < 0; // only ENTER a lock from free mode via double-click, never re-target an existing one
                ImVec2 mouse = io.MousePos;
                ImVec2 screen_quad[4];

                if (raycaster.locked_axis < 0 || raycaster.locked_axis == 2)
                {
                    DrawSlicePlaneOutline(item_min, size, cam, axial_corners, IM_COL32(230, 60, 60, 255));
                    if (double_clicked && was_free && ProjectPlaneCorners(item_min, size, cam, axial_corners, screen_quad) && PointInQuad(mouse, screen_quad))
                        { raycaster.locked_axis = 2; raycaster.locked_angle = 0.0f; }
                }
                if (raycaster.locked_axis < 0 || raycaster.locked_axis == 1)
                {
                    DrawSlicePlaneOutline(item_min, size, cam, coronal_corners, IM_COL32(230, 210, 60, 255));
                    if (double_clicked && was_free && ProjectPlaneCorners(item_min, size, cam, coronal_corners, screen_quad) && PointInQuad(mouse, screen_quad))
                        { raycaster.locked_axis = 1; raycaster.locked_angle = 0.0f; }
                }
                if (raycaster.locked_axis < 0 || raycaster.locked_axis == 0)
                {
                    DrawSlicePlaneOutline(item_min, size, cam, sagittal_corners, IM_COL32(60, 220, 120, 255));
                    if (double_clicked && was_free && ProjectPlaneCorners(item_min, size, cam, sagittal_corners, screen_quad) && PointInQuad(mouse, screen_quad))
                        { raycaster.locked_axis = 0; raycaster.locked_angle = 0.0f; }
                }
            }
        }
        else
        {
            ImGui::TextUnformatted("3D - failed to load CT volume or initialize the raycaster, see console");
        }
        ImGui::End();

        if (show_dvh)
        {
            if (dvh_needs_repositioning)
            {
                ImVec2 vp_size = ImGui::GetMainViewport()->Size;
                ImVec2 vp_pos = ImGui::GetMainViewport()->Pos;
                ImGui::SetNextWindowPos(ImVec2(vp_pos.x + vp_size.x * 0.5f, vp_pos.y + vp_size.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
                ImGui::SetNextWindowSize(ImVec2(std::min(600.0f, vp_size.x * 0.6f), std::min(450.0f, vp_size.y * 0.6f)), ImGuiCond_Always);
                dvh_needs_repositioning = false;
            }
            ImGui::Begin("DVH", &show_dvh);
            if (!dvh_curves.empty())
            {
                if (dvh_dose_is_synthetic)
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Synthetic dose (Gaussian blob) - NOT a real treatment plan.");
                else
                    ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Dose: %s/dose (real RTDOSE export)", case_data_dir.c_str());
                if (ImPlot::BeginPlot("Dose-Volume Histogram", ImVec2(-1, -1)))
                {
                    ImPlot::SetupAxes("Dose (Gy)", "Volume (%)");
                    for (size_t li = 0; li < dvh_curves.size(); ++li)
                    {
                        const uint8_t* c = ColorForStructureName(dvh_curves[li].name);
                        ImPlotSpec spec;
                        spec.LineColor = ImVec4(c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f, 1.0f);
                        ImPlot::PlotLine(dvh_curves[li].name.c_str(), dvh_curves[li].dose_gy.data(), dvh_curves[li].volume_pct.data(),
                            (int)dvh_curves[li].dose_gy.size(), spec);
                    }
                    ImPlot::EndPlot();
                }
            }
            else
            {
                ImGui::TextUnformatted("No structures loaded for DVH.");
            }
            ImGui::End();
        }

        ImGui::Render();
        const float clear_color_with_alpha[4] = {
            clear_color.x * clear_color.w, clear_color.y * clear_color.w,
            clear_color.z * clear_color.w, clear_color.w
        };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color_with_alpha);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        HRESULT hr = g_pSwapChain->Present(1, 0);
        g_SwapChainOccluded = (hr == DXGI_STATUS_OCCLUDED);
    }

    if (axial_srv) axial_srv->Release();
    if (axial_tex) axial_tex->Release();
    if (coronal_srv) coronal_srv->Release();
    if (coronal_tex) coronal_tex->Release();
    if (sagittal_srv) sagittal_srv->Release();
    if (sagittal_tex) sagittal_tex->Release();
    if (raycaster_ok) ReleaseRaycaster(raycaster);
    if (inference_job.process_handle) CloseHandle(inference_job.process_handle); // let a still-running job finish on its own

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow(hwnd);
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);

    return 0;
}

bool CreateDeviceD3D(HWND hWnd)
{
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    UINT createDeviceFlags = 0;
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0 };
    HRESULT res = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags,
        featureLevelArray, 2, D3D11_SDK_VERSION, &sd,
        &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (res == DXGI_ERROR_UNSUPPORTED)
        res = D3D11CreateDeviceAndSwapChain(
            nullptr, D3D_DRIVER_TYPE_WARP, nullptr, createDeviceFlags,
            featureLevelArray, 2, D3D11_SDK_VERSION, &sd,
            &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (res != S_OK)
        return false;

    CreateRenderTarget();
    return true;
}

void CleanupDeviceD3D()
{
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = nullptr; }
}

void CreateRenderTarget()
{
    ID3D11Texture2D* pBackBuffer;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_mainRenderTargetView);
    pBackBuffer->Release();
}

void CleanupRenderTarget()
{
    if (g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = nullptr; }
}

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg)
    {
    case WM_SIZE:
        if (wParam == SIZE_MINIMIZED)
            return 0;
        g_ResizeWidth = (UINT)LOWORD(lParam);
        g_ResizeHeight = (UINT)HIWORD(lParam);
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU)
            return 0;
        break;
    case WM_DESTROY:
        ::PostQuitMessage(0);
        return 0;
    }
    return ::DefWindowProcW(hWnd, msg, wParam, lParam);
}
