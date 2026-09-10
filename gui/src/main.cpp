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
#include "inference_config.h"
#include "structure_metrics.h"
#include "imgui_stdlib.h" // ImGui::InputText overloads taking std::string*
#include <shellapi.h>     // DragAcceptFiles/DragQueryFileW (after inference.h's <windows.h>)
#include <algorithm>
#include <array>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <d3d11.h>
#include <tchar.h>

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

static ID3D11Device* g_pd3dDevice = nullptr;
static ID3D11DeviceContext* g_pd3dDeviceContext = nullptr;
static IDXGISwapChain* g_pSwapChain = nullptr;
static bool g_SwapChainOccluded = false;
static UINT g_ResizeWidth = 0, g_ResizeHeight = 0;
static std::vector<std::string> g_DroppedPaths; // UTF-8 paths from the last WM_DROPFILES, consumed by the render loop
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

// Per-MPR-panel zoom/pan: which part of the slice is shown, as fractions of
// the displayed (fit-to-panel, superior-on-top) image. The mouse wheel over
// the image zooms about the mouse, right-drag pans, middle-click resets.
// Implemented as a UV sub-rectangle of the same fit-size ImGui::Image, not a
// bigger image in a scrolling child window: the earlier Ctrl+wheel zoom did
// that, and its panel-level IsWindowHovered() check never saw the mouse while
// it was over the child region -- why wheel input seemed unreliable.
struct SliceView
{
    static constexpr float kMaxZoom = 8.0f;
    float zoom = 1.0f;              // 1 = the whole slice
    float left = 0.0f, top = 0.0f;  // visible region's top-left corner, in [0, 1 - 1/zoom]

    float Extent() const { return 1.0f / zoom; }
    void Clamp()
    {
        zoom = std::clamp(zoom, 1.0f, kMaxZoom);
        left = std::clamp(left, 0.0f, 1.0f - Extent());
        top = std::clamp(top, 0.0f, 1.0f - Extent());
    }
};

// ImGui::Image uv0/uv1 showing `view`. `flipped` = coronal/sagittal, whose
// texture row 0 is inferior and is displayed at the bottom.
static void SliceViewUV(const SliceView& view, bool flipped, ImVec2* uv0, ImVec2* uv1)
{
    float extent = view.Extent();
    *uv0 = ImVec2(view.left, flipped ? 1.0f - view.top : view.top);
    *uv1 = ImVec2(view.left + extent, flipped ? 1.0f - (view.top + extent) : view.top + extent);
}

// Wheel zoom (about the mouse), right-drag pan and middle-click reset for the
// slice image item just drawn at item_min/item_size.
static void UpdateSliceView(SliceView& view, ImVec2 item_min, ImVec2 item_size)
{
    if (!ImGui::IsItemHovered())
        return;
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 mouse((io.MousePos.x - item_min.x) / item_size.x, (io.MousePos.y - item_min.y) / item_size.y); // fraction of the displayed image

    if (io.MouseWheel != 0.0f)
    {
        float point_x = view.left + mouse.x * view.Extent(); // the slice point under the mouse stays under it
        float point_y = view.top + mouse.y * view.Extent();
        view.zoom = std::clamp(view.zoom * std::pow(1.2f, io.MouseWheel), 1.0f, SliceView::kMaxZoom);
        view.left = point_x - mouse.x * view.Extent();
        view.top = point_y - mouse.y * view.Extent();
    }
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Right) && view.zoom > 1.0f)
    {
        view.left -= io.MouseDelta.x / item_size.x * view.Extent();
        view.top -= io.MouseDelta.y / item_size.y * view.Extent();
    }
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle))
        view = SliceView{};
    view.Clamp();
}

// If the item just drawn (expected to be an ImGui::Image of a (tex_w x
// tex_h) slice texture showing `view`, optionally vertically flipped per
// `flipped`) is being clicked/dragged, converts the mouse position to texture
// pixel coordinates. Returns false (leaving *out_col/*out_row untouched) otherwise.
static bool PanelClicked(ImVec2 item_min, ImVec2 item_size, int tex_w, int tex_h, bool flipped, const SliceView& view, int* out_col, int* out_row)
{
    if (!ImGui::IsItemHovered() || !ImGui::IsMouseDown(ImGuiMouseButton_Left))
        return false;

    ImVec2 mouse = ImGui::GetIO().MousePos;
    float mx = (mouse.x - item_min.x) / item_size.x;
    float my = (mouse.y - item_min.y) / item_size.y;
    if (mx < 0.0f || mx > 1.0f || my < 0.0f || my > 1.0f)
        return false;
    float fx = view.left + mx * view.Extent(); // displayed-image fraction -> whole-slice fraction
    float fy = view.top + my * view.Extent();

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
    const StructureMetrics* metrics = nullptr; // rows matched across both, see ComputeStructureMetrics
};

// Crosshair through the shared cursor voxel (ci, cj, ck), projected into
// this panel's (col, row) in-plane coordinates, plus an (i,j,k)/HU readout
// and, when ground truth and/or a prediction are loaded, a second line
// showing what each says is at the cursor and that structure's Dice score.
static void DrawCrosshairAndReadout(ImVec2 item_min, ImVec2 item_size, int tex_w, int tex_h, int col, int row, bool flipped,
    const SliceView& view, const Volume& vol, int ci, int cj, int ck, const StructureComparisonContext& cmp)
{
    int display_row = flipped ? (tex_h - 1 - row) : row;
    float fx = (((float)col + 0.5f) / tex_w - view.left) / view.Extent(); // whole-slice fraction -> displayed-image fraction
    float fy = (((float)display_row + 0.5f) / tex_h - view.top) / view.Extent();
    float sx = item_min.x + fx * item_size.x;
    float sy = item_min.y + fy * item_size.y;

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImU32 crosshair_color = IM_COL32(255, 230, 0, 180);
    if (fx >= 0.0f && fx <= 1.0f) // zoomed in, the cursor can be outside the visible region
        draw_list->AddLine(ImVec2(sx, item_min.y), ImVec2(sx, item_min.y + item_size.y), crosshair_color);
    if (fy >= 0.0f && fy <= 1.0f)
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
        int gt_row = -1, pred_row = -1; // index into cmp.metrics->rows; -1 = background
        if (cmp.masks_loaded)
        {
            uint8_t id = cmp.label_volume->At(ci, cj, ck);
            gt_name = id == 0 ? "background" : cmp.label_volume->labels[id - 1];
            gt_row = (id == 0 || !cmp.metrics) ? -1 : cmp.metrics->gt_id_to_row[id];
        }
        if (cmp.prediction_loaded)
        {
            uint8_t id = cmp.prediction_volume->At(ci, cj, ck);
            pred_name = id == 0 ? "background" : cmp.prediction_volume->labels[id - 1];
            pred_row = (id == 0 || !cmp.metrics) ? -1 : cmp.metrics->pred_id_to_row[id];
        }

        if (!cmp.masks_loaded)
            line2_color = IM_COL32(220, 220, 220, 255); // prediction only, nothing to compare against
        else if (!cmp.prediction_loaded)
            line2_color = IM_COL32(220, 220, 220, 255); // ground truth only, ditto
        else if (gt_row == -1 && pred_row == -1)
            line2_color = IM_COL32(180, 180, 180, 255); // both background - not interesting either way
        else if (gt_row == pred_row)
            line2_color = IM_COL32(90, 220, 130, 255);  // agree on a real structure
        else
            line2_color = IM_COL32(230, 90, 90, 255);   // disagree (false positive/negative here)

        int relevant_row = gt_row >= 0 ? gt_row : pred_row;
        float dice_val = relevant_row >= 0 ? cmp.metrics->rows[relevant_row].dice : std::numeric_limits<float>::quiet_NaN();

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

    if (view.zoom > 1.001f)
    {
        char zoom_text[64];
        snprintf(zoom_text, sizeof(zoom_text), "%.1fx  (right-drag pans, middle-click resets)", view.zoom);
        ImVec2 zoom_size = ImGui::CalcTextSize(zoom_text);
        ImVec2 zoom_pos(item_min.x + 4.0f, item_min.y + item_size.y - zoom_size.y - 8.0f);
        draw_list->AddRectFilled(zoom_pos, ImVec2(zoom_pos.x + zoom_size.x + 4.0f, zoom_pos.y + zoom_size.y + 4.0f), IM_COL32(0, 0, 0, 160));
        draw_list->AddText(ImVec2(zoom_pos.x + 2.0f, zoom_pos.y + 2.0f), IM_COL32(255, 255, 255, 255), zoom_text);
    }
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

// Drag-and-drop: what a dropped path looks like, by name only (no file reads
// on the UI thread -- source/gui_load.py validates the actual contents).
enum class DropKind { DicomSeries, Volume, SegNrrd, RtStruct, RtDose, Unsupported };

static const char* kVolumeSuffixes[] = { ".nii.gz", ".nii", ".nrrd", ".nhdr", ".mha", ".mhd" };

static std::string LowercaseAscii(std::string s)
{
    for (char& c : s)
        c = (char)std::tolower((unsigned char)c);
    return s;
}

static bool EndsWith(const std::string& s, const std::string& suffix)
{
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// A dropped path, ignoring any trailing separator (so filename() is the last component).
static std::filesystem::path NormalizedPath(const std::string& utf8_path)
{
    std::filesystem::path p = std::filesystem::path(Utf8ToWide(utf8_path)).lexically_normal();
    return p.has_filename() ? p : p.parent_path();
}

static std::string FileNameUtf8(const std::filesystem::path& p)
{
    return WideToUtf8(p.filename().wstring());
}

static DropKind ClassifyDroppedPath(const std::string& utf8_path)
{
    std::filesystem::path p = NormalizedPath(utf8_path);
    std::error_code ec;
    if (std::filesystem::is_directory(p, ec))
    {
        for (const auto& entry : std::filesystem::directory_iterator(p, ec))
            if (EndsWith(LowercaseAscii(FileNameUtf8(entry.path())), ".seg.nrrd"))
                return DropKind::SegNrrd;
        return DropKind::DicomSeries;
    }
    std::string name = LowercaseAscii(FileNameUtf8(p));
    if (EndsWith(name, ".seg.nrrd"))
        return DropKind::SegNrrd;
    if (EndsWith(name, ".dcm"))
    {
        if (name.rfind("rs.", 0) == 0 || name.find("rtstruct") != std::string::npos)
            return DropKind::RtStruct;
        if (name.rfind("rd.", 0) == 0 || name.find("rtdose") != std::string::npos)
            return DropKind::RtDose;
        return DropKind::DicomSeries;
    }
    for (const char* suffix : kVolumeSuffixes)
        if (EndsWith(name, suffix))
            return DropKind::Volume;
    return DropKind::Unsupported;
}

// The "user/<name>" case a dropped CT becomes. A series directory (or a .dcm
// slice's directory) is named "<parent>_<dir>", since exports often use
// generic folder names ("Original", "DICOM"); a volume file uses its stem.
static std::string CaseNameForDroppedCt(const std::string& utf8_path)
{
    std::filesystem::path p = NormalizedPath(utf8_path);
    std::error_code ec;
    std::string lower = LowercaseAscii(FileNameUtf8(p));
    std::string name;
    if (!std::filesystem::is_directory(p, ec) && EndsWith(lower, ".dcm"))
        p = p.parent_path();
    if (std::filesystem::is_directory(p, ec))
    {
        name = FileNameUtf8(p.parent_path()) + "_" + FileNameUtf8(p);
    }
    else
    {
        name = FileNameUtf8(p);
        for (const char* suffix : kVolumeSuffixes)
            if (EndsWith(lower, suffix))
            {
                name.resize(name.size() - std::string(suffix).size());
                break;
            }
    }

    std::string sanitized;
    for (unsigned char c : name)
        sanitized += (std::isalnum(c) || c == '-' || c == '.' || c == '_') ? (char)c : '_';
    if (sanitized.empty())
        sanitized = "case";
    if (sanitized[0] == '.')
        sanitized[0] = '_';
    return "user/" + sanitized;
}

// The "Load as" choice to preselect: 0 = CT (new case), 1 = ground truth, 2 = prediction, 3 = dose.
static int DefaultDropTarget(DropKind kind, const std::string& first_path, bool ct_loaded)
{
    if (kind == DropKind::RtDose)
        return ct_loaded ? 3 : 0;
    if (!ct_loaded || kind == DropKind::DicomSeries)
        return 0;
    std::string name = LowercaseAscii(FileNameUtf8(NormalizedPath(first_path)));
    if (name.find("pred") != std::string::npos)
        return 2;
    if (kind == DropKind::SegNrrd || kind == DropKind::RtStruct)
        return 1;
    for (const char* hint : { "seg", "mask", "label", "gt" })
        if (name.find(hint) != std::string::npos)
            return 1;
    for (const char* hint : { "_0000.", "ct", "img", "image" })
        if (name.find(hint) != std::string::npos)
            return 0;
    return 1; // a volume dropped onto an already-open case is most likely its masks
}

// Cases with an exported CT under `data_dir` (<case>/gui_export/ct.json, one
// or two folders deep: "phantom", "user/<name>", "hanseg_cases/case_03"),
// most recently exported first -- File > Recent cases.
static std::vector<std::string> FindRecentCases(const std::string& data_dir, size_t max_count = 15)
{
    namespace fs = std::filesystem;
    std::vector<std::pair<fs::file_time_type, std::string>> found;
    auto consider = [&](const fs::path& dir, const std::string& name)
    {
        std::error_code ec;
        fs::file_time_type t = fs::last_write_time(dir / L"gui_export" / L"ct.json", ec);
        if (!ec)
            found.emplace_back(t, name);
    };
    try
    {
        fs::path root = fs::path(Utf8ToWide(data_dir)).lexically_normal();
        for (const fs::directory_entry& level1 : fs::directory_iterator(root))
        {
            if (!level1.is_directory())
                continue;
            std::string name1 = FileNameUtf8(level1.path());
            consider(level1.path(), name1);
            for (const fs::directory_entry& level2 : fs::directory_iterator(level1.path()))
                if (level2.is_directory())
                    consider(level2.path(), name1 + "/" + FileNameUtf8(level2.path()));
        }
    }
    catch (const fs::filesystem_error&)
    {
        // An unreadable folder somewhere under data/: list what was found before it.
    }
    std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<std::string> cases;
    for (size_t i = 0; i < found.size() && i < max_count; ++i)
        cases.push_back(found[i].second);
    return cases;
}

// "20260910_143015", for export/batch output folder names.
static std::string LocalTimestamp()
{
    SYSTEMTIME t;
    ::GetLocalTime(&t);
    char stamp[32];
    snprintf(stamp, sizeof(stamp), "%04d%02d%02d_%02d%02d%02d", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond);
    return stamp;
}

static void OpenFolderInExplorer(const std::string& utf8_dir)
{
    std::wstring dir = std::filesystem::path(Utf8ToWide(utf8_dir)).lexically_normal().make_preferred().wstring();
    ::ShellExecuteW(nullptr, L"explore", dir.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

static std::string CsvField(const std::string& s)
{
    if (s.find_first_of(",\"\n") == std::string::npos)
        return s;
    std::string quoted = "\"";
    for (char c : s)
        quoted += (c == '"') ? std::string("\"\"") : std::string(1, c);
    return quoted + "\"";
}

// NVIDIA adapters (vendor 0x10DE) DXGI can see: the GPUs local nnU-Net (CUDA)
// could use, to cap Batch Evaluation's local parallelism. A proxy -- CUDA's
// own count needs the CUDA runtime -- but DXGI is already linked, and
// source/batch_evaluate.py re-checks with torch.cuda.device_count().
static int CountNvidiaGpus()
{
    IDXGIFactory1* factory = nullptr;
    if (FAILED(::CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory)))
        return 0;
    int count = 0;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i)
    {
        DXGI_ADAPTER_DESC1 desc;
        if (SUCCEEDED(adapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && desc.VendorId == 0x10DE)
            ++count;
        adapter->Release();
    }
    factory->Release();
    return count;
}

int main(int argc, char** argv)
{
    // Optional argv[1]: a case directory under <repo>/data, e.g. "phantom"
    // (default) or "hanseg_cases/case_03" - see source/export_gui_volume.py.
    // Any other data is opened at runtime by dragging it onto the window,
    // which converts it into a "user/<name>" case (source/gui_load.py) and
    // switches to it (LoadCase below).
    std::string case_dir = (argc > 1) ? argv[1] : "phantom";
    std::string case_data_dir;

    // Editable configs/<case_dir>/inference_config.json (source/inference_config.py's
    // schema), (re)loaded per case by LoadCase -- missing is fine, LoadInferenceConfig
    // leaves the current settings for a brand-new case so the user can fill in fields
    // and Save. `force_select_backend_tab` makes the Local/Cloud tab bar below open on
    // whichever backend the config actually has (ImGui otherwise always defaults to the first tab).
    InferenceConfigUI inference_cfg;
    bool force_select_backend_tab = true;

    WNDCLASSEXW wc = {
        sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L,
        GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr,
        L"ContourLensWindowClass", nullptr
    };
    ::RegisterClassExW(&wc);
    HWND hwnd = ::CreateWindowW(
        wc.lpszClassName, L"ContourLens", WS_OVERLAPPEDWINDOW,
        100, 100, 1280, 800, nullptr, nullptr, wc.hInstance, nullptr);

    if (!CreateDeviceD3D(hwnd))
    {
        CleanupDeviceD3D();
        ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    ::DragAcceptFiles(hwnd, TRUE); // Explorer drops arrive as WM_DROPFILES -> g_DroppedPaths (see WndProc)
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

    // The current case's CT, (re)loaded by LoadCase below: argv[1]'s case at
    // startup, then any CT dropped onto the window. Orthogonal MPR only:
    // slicing below assumes an axis-aligned volume (identity direction matrix
    // -- source/export_gui_volume.py's to_gui_orientation reorients
    // axis-aligned volumes on export) and ignores Volume::direction entirely.
    // Oblique reformatting would need a resample through that matrix - deferred.
    Volume ct_volume;
    bool ct_loaded = false;

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
    SliceView slice_views[3]; // zoom/pan per MPR panel, indexed kAxial/kCoronal/kSagittal (below)

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

    // One background job at a time: loading dropped data (source/gui_load.py)
    // or a "Run Inference" (source/run_inference_for_gui.py). Its progress,
    // log, and result show at the top of the Controls panel (DrawJobStatus).
    BackgroundJob job;

    // Per-structure Dice/volumes (ground truth vs. prediction), for the
    // Metrics table and the crosshair readout's accuracy line. Recomputed
    // (full-volume pass, so not every frame) whenever either is loaded/reloaded.
    StructureMetrics structure_metrics;
    std::vector<int> metrics_order; // Metrics table's sorted row order; cleared whenever the metrics change
    bool expand_metrics = false;    // opens the Metrics section once, when an inference run finishes

    // The last drop, awaiting the "Load dropped data" popup's choice
    // (classified once when dropped, not every frame the popup is open).
    std::vector<std::string> drop_paths;
    DropKind drop_kind = DropKind::Unsupported;
    bool drop_is_directory = false;
    std::string drop_ct_case_name;
    int drop_target = 0; // 0 = CT (new case), 1 = ground truth, 2 = prediction, 3 = dose

    // Error map (Metrics > Error map, or View > Error map): voxels where the
    // prediction disagrees with the ground truth (ComputeErrorVolume), drawn
    // as a third overlay in the 2D panels. Colors indexed by kErrorMissed-1 / kErrorExtra-1.
    LabelVolume error_volume;
    bool show_error_map = false;
    float error_map_alpha = 0.6f;
    const std::vector<std::array<uint8_t, 3>> error_colors = { { 255, 140, 0 }, { 0, 200, 255 } }; // missed = orange, extra = cyan
    int selected_metric_row = -1;  // last Metrics row clicked (jumped to)
    bool aliases_changed = false;  // set while drawing Metrics; applied after the Controls window, never mid-iteration

    std::vector<std::string> recent_cases; // File > Recent cases, rescanned whenever that menu opens

    // View > Batch Evaluation (source/batch_evaluate.py).
    bool show_batch = false;
    int batch_fold = 0;
    int batch_parallel = 2;       // concurrent Modal predictions
    nlohmann::json batch_results; // the last finished run's results.json (null until one finishes)
    std::string batch_results_dir;
    const int local_gpu_count = CountNvidiaGpus(); // caps Batch Evaluation's parallelism on the local backend

    // Raycast 3D panel.
    Raycaster raycaster;
    bool raycaster_ok = false;
    bool show_slice_planes = true; // axial/coronal/sagittal cut-plane outlines, at the shared cursor

    // DVH. "View > DVH" toggles the window's visibility. Prefers a real dose
    // export (source/export_gui_volume.py --dose <RTDOSE.dcm>) if present at
    // gui_export/dose(.bin/.json); falls back to a synthetic Gaussian blob
    // (see the session decision: no RTDOSE exists for the phantom case) so
    // the DVH plot still has something to show end-to-end.
    std::vector<DVHCurve> dvh_curves;      // ground-truth structures
    std::vector<DVHCurve> dvh_pred_curves; // predicted structures, from the same dose (so the same bins)
    bool dvh_show_prediction = true;
    bool show_dvh = false;
    bool dvh_needs_repositioning = true; // forces a sane pos/size the first time (and after toggled back on)
    bool dvh_dose_is_synthetic = true;

    // Fills structure_metrics' hd95_mm from source/gui_metrics.py's results
    // file, but only while it's newer than both label volumes (stale results
    // are ignored, not shown). Returns true when every scored structure has one.
    auto LoadHd95Results = [&]() -> bool
    {
        if (!masks_loaded || !prediction_loaded)
            return false;
        std::filesystem::path dir(Utf8ToWide(case_data_dir));
        std::error_code ec;
        auto results_time = std::filesystem::last_write_time(dir / L"metrics_hd95.json", ec);
        if (ec)
            return false;
        for (const wchar_t* volume : { L"masks.bin", L"prediction.bin" })
        {
            auto volume_time = std::filesystem::last_write_time(dir / volume, ec);
            if (ec || volume_time > results_time)
                return false;
        }

        std::ifstream f(dir / L"metrics_hd95.json");
        nlohmann::json j = nlohmann::json::parse(f, nullptr, false);
        if (j.is_discarded() || !j.is_object() || !j.contains("pairs") || !j["pairs"].is_object())
            return false;
        bool complete = true;
        for (StructureMetric& row : structure_metrics.rows)
        {
            if (!row.gt_id || !row.pred_id)
                continue;
            std::string key = std::to_string(row.gt_id) + ":" + std::to_string(row.pred_id);
            if (!j["pairs"].contains(key))
                complete = false;
            else if (j["pairs"][key].is_number())
                row.hd95_mm = j["pairs"][key].get<float>();
        }
        return complete;
    };

    auto RecomputeMetrics = [&]()
    {
        structure_metrics = ComputeStructureMetrics(masks_loaded ? &label_volume : nullptr,
            prediction_loaded ? &prediction_volume : nullptr, ct_volume.spacing);
        metrics_order.clear();
        selected_metric_row = -1;
        error_volume = (masks_loaded && prediction_loaded)
            ? ComputeErrorVolume(label_volume, prediction_volume, structure_metrics) : LabelVolume{};
        if (error_volume.data.empty())
            show_error_map = false;
        LoadHd95Results();
        k_dirty = j_dirty = i_dirty = true;
    };

    // DVH per ground-truth and predicted structure: a real dose export when
    // one matches the CT's grid, else the synthetic one.
    auto RecomputeDVH = [&]()
    {
        dvh_curves.clear();
        dvh_pred_curves.clear();
        if (!masks_loaded && !prediction_loaded)
            return;

        DoseVolume dose_volume;
        bool real_dose_loaded = LoadDoseVolume(case_data_dir + "/dose", dose_volume);
        if (real_dose_loaded && (dose_volume.nx != ct_volume.nx || dose_volume.ny != ct_volume.ny || dose_volume.nz != ct_volume.nz))
        {
            fprintf(stderr, "Dose volume shape %dx%dx%d doesn't match CT shape %dx%dx%d; ignoring it\n",
                dose_volume.nx, dose_volume.ny, dose_volume.nz, ct_volume.nx, ct_volume.ny, ct_volume.nz);
            real_dose_loaded = false;
        }

        dvh_dose_is_synthetic = !real_dose_loaded;
        std::vector<float> synthetic_dose;
        if (!real_dose_loaded)
            synthetic_dose = GenerateSyntheticDose(ct_volume);
        const std::vector<float>& dose = real_dose_loaded ? dose_volume.data : synthetic_dose;
        if (masks_loaded)
            dvh_curves = ComputeDVHCurves(dose, label_volume);
        if (prediction_loaded)
            dvh_pred_curves = ComputeDVHCurves(dose, prediction_volume);
    };

    // (Re)loads one overlay slot for the current case -- ground truth
    // (gui_export/masks) or prediction (gui_export/prediction) -- whichever
    // of startup, a dropped-mask job, or an inference run wrote it. A missing
    // or mismatched file just leaves that slot empty. `recompute` = false
    // when loading both slots in a row (LoadCase), to compute DVH/metrics once.
    enum class LabelSlot { GroundTruth, Prediction };
    auto LoadLabelSlot = [&](LabelSlot slot, bool recompute = true)
    {
        bool is_prediction = slot == LabelSlot::Prediction;
        LabelVolume& volume = is_prediction ? prediction_volume : label_volume;
        bool& loaded = is_prediction ? prediction_loaded : masks_loaded;
        std::vector<bool>& visible = is_prediction ? prediction_label_visible : label_visible;
        std::vector<std::array<uint8_t, 3>>& colors = is_prediction ? prediction_label_colors : label_colors;

        loaded = ct_loaded && LoadLabelVolume(case_data_dir + (is_prediction ? "/prediction" : "/masks"), volume);
        if (loaded && (volume.nx != ct_volume.nx || volume.ny != ct_volume.ny || volume.nz != ct_volume.nz))
        {
            fprintf(stderr, "%s volume shape %dx%dx%d doesn't match CT shape %dx%dx%d; ignoring it\n",
                is_prediction ? "Prediction" : "Mask", volume.nx, volume.ny, volume.nz, ct_volume.nx, ct_volume.ny, ct_volume.nz);
            loaded = false;
        }

        if (loaded)
        {
            visible.assign(volume.labels.size(), true);
            colors = BuildLabelColors(volume.labels);
            (is_prediction ? show_prediction_overlay : show_overlays) = true;
            if (raycaster_ok)
                UploadLabelVolume(g_pd3dDevice, raycaster, volume, colors, is_prediction);
        }
        else
        {
            volume = LabelVolume{};
            visible.clear();
            colors.clear();
            (is_prediction ? raycaster.has_pred_labels : raycaster.has_gt_labels) = false;
        }

        if (recompute)
        {
            RecomputeDVH();
            RecomputeMetrics();
        }
        k_dirty = j_dirty = i_dirty = true;
    };

    // After configs/structure_aliases.json changes: re-derive colors (an alias
    // can map a name onto a canonical structure's color) and re-pair metrics.
    auto ReapplyAliases = [&]()
    {
        if (masks_loaded)
        {
            label_colors = BuildLabelColors(label_volume.labels);
            if (raycaster_ok)
                UploadLabelVolume(g_pd3dDevice, raycaster, label_volume, label_colors, false);
        }
        if (prediction_loaded)
        {
            prediction_label_colors = BuildLabelColors(prediction_volume.labels);
            if (raycaster_ok)
                UploadLabelVolume(g_pd3dDevice, raycaster, prediction_volume, prediction_label_colors, true);
        }
        RecomputeMetrics();
    };

    // Metrics row click: put the cursor where that structure's disagreement is worst.
    auto JumpToStructure = [&](int row)
    {
        int i = 0, j = 0, k = 0;
        if (row < 0 || row >= (int)structure_metrics.rows.size() ||
            !FindStructureFocus(masks_loaded ? &label_volume : nullptr, prediction_loaded ? &prediction_volume : nullptr,
                structure_metrics.rows[row], &i, &j, &k))
            return;
        cursor_i = i;
        cursor_j = j;
        cursor_k = k;
        k_dirty = j_dirty = i_dirty = true;
        selected_metric_row = row;

        // Zoomed-in panels re-center on the jumped-to voxel (displayed-image
        // fractions; coronal/sagittal show superior on top, hence 1 - k).
        auto center = [](SliceView& view, float fx, float fy)
        {
            view.left = fx - 0.5f * view.Extent();
            view.top = fy - 0.5f * view.Extent();
            view.Clamp();
        };
        center(slice_views[kAxial], (i + 0.5f) / ct_volume.nx, (j + 0.5f) / ct_volume.ny);
        center(slice_views[kCoronal], (i + 0.5f) / ct_volume.nx, 1.0f - (k + 0.5f) / ct_volume.nz);
        center(slice_views[kSagittal], (j + 0.5f) / ct_volume.ny, 1.0f - (k + 0.5f) / ct_volume.nz);
        focused_panel = PanelId::Axial;
        ImGui::SetWindowFocus("Axial");
    };

    // Switches the whole viewer to `new_case_dir`: its CT (slice textures and
    // 3D raycaster rebuilt for the new size), both overlay slots, dose/DVH,
    // and its inference config. `dropped_source` is set for a case just
    // created from dragged-in data: it becomes that config's source_ct
    // (keeping the current model/backend settings when the case has no config
    // yet), so Run Inference works on it right away.
    auto LoadCase = [&](const std::string& new_case_dir, const std::string& dropped_source)
    {
        case_dir = new_case_dir;
        case_data_dir = std::string(CONTOURLENS_DATA_DIR) + "/" + case_dir + "/gui_export";
        ::SetWindowTextW(hwnd, (L"ContourLens - " + Utf8ToWide(case_dir)).c_str());

        LoadInferenceConfig(CONTOURLENS_REPO_DIR, case_dir, inference_cfg);
        if (!dropped_source.empty())
        {
            inference_cfg.source_ct = dropped_source;
            SaveInferenceConfig(CONTOURLENS_REPO_DIR, case_dir, inference_cfg);
        }
        force_select_backend_tab = true;

        for (ID3D11ShaderResourceView** srv : { &axial_srv, &coronal_srv, &sagittal_srv })
            if (*srv) { (*srv)->Release(); *srv = nullptr; }
        for (ID3D11Texture2D** tex : { &axial_tex, &coronal_tex, &sagittal_tex })
            if (*tex) { (*tex)->Release(); *tex = nullptr; }
        ReleaseRaycaster(raycaster); // null-safe; also resets the camera for the new volume
        raycaster_ok = false;

        ct_volume = Volume{};
        ct_loaded = LoadVolume(case_data_dir + "/ct", ct_volume);
        if (ct_loaded)
        {
            cursor_i = ct_volume.nx / 2;
            cursor_j = ct_volume.ny / 2;
            cursor_k = ct_volume.nz / 2;
            for (SliceView& view : slice_views)
                view = SliceView{}; // a new volume starts un-zoomed
            auto [min_it, max_it] = std::minmax_element(ct_volume.data.begin(), ct_volume.data.end());
            printf("Loaded CT volume (%s): %dx%dx%d spacing=(%.3f,%.3f,%.3f) HU range=[%d,%d]\n", case_dir.c_str(),
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
        }

        if (ct_loaded)
        {
            raycaster_ok = InitRaycaster(g_pd3dDevice, ct_volume, raycaster);
            if (!raycaster_ok)
                fprintf(stderr, "Failed to initialize the GPU raycaster; 3D panel will be unavailable\n");
        }
        else
        {
            fprintf(stderr, "No CT loaded for case '%s' (see LoadVolume errors above); drag a CT onto the window, "
                            "or run source/export_gui_volume.py\n", case_dir.c_str());
        }

        LoadLabelSlot(LabelSlot::GroundTruth, false);
        LoadLabelSlot(LabelSlot::Prediction, false); // e.g. from a previous "Run Inference" click
        RecomputeDVH();
        RecomputeMetrics();
    };

    auto StartLoadCtJob = [&](const std::string& path)
    {
        job.kind = JobKind::LoadCt;
        job.title = "Loading CT";
        job.case_name = CaseNameForDroppedCt(path);
        job.source = path;
        StartJob(job, CONTOURLENS_REPO_DIR, "source.gui_load", { "ct", path, "--case", job.case_name });
    };

    auto StartLoadMasksJob = [&](const std::vector<std::string>& paths, bool as_prediction)
    {
        job.kind = JobKind::LoadMasks;
        job.slot = as_prediction ? "prediction" : "gt";
        job.title = as_prediction ? "Loading prediction" : "Loading ground truth";
        job.case_name = case_dir;
        std::vector<std::string> args = { "masks" };
        args.insert(args.end(), paths.begin(), paths.end());
        args.insert(args.end(), { "--case", case_dir, "--slot", job.slot });
        StartJob(job, CONTOURLENS_REPO_DIR, "source.gui_load", args);
    };

    auto StartInference = [&]()
    {
        SaveInferenceConfig(CONTOURLENS_REPO_DIR, case_dir, inference_cfg);
        job.kind = JobKind::Inference;
        job.title = "Inference";
        job.case_name = case_dir;
        StartJob(job, CONTOURLENS_REPO_DIR, "source.run_inference_for_gui", { case_dir });
    };

    auto StartLoadDoseJob = [&](const std::string& path)
    {
        job.kind = JobKind::LoadDose;
        job.title = "Loading dose";
        job.case_name = case_dir;
        StartJob(job, CONTOURLENS_REPO_DIR, "source.gui_load", { "dose", path, "--case", case_dir });
    };

    // HD95 for every scored structure (source/gui_metrics.py). Passes this
    // GUI's own id pairs, so aliases and matching aren't re-derived in Python.
    auto StartHd95Job = [&]()
    {
        std::string pairs;
        for (const StructureMetric& row : structure_metrics.rows)
            if (row.gt_id && row.pred_id)
                pairs += (pairs.empty() ? "" : ",") + std::to_string(row.gt_id) + ":" + std::to_string(row.pred_id);
        if (pairs.empty())
            return;
        job.kind = JobKind::Hd95;
        job.title = "Computing HD95";
        job.case_name = case_dir;
        StartJob(job, CONTOURLENS_REPO_DIR, "source.gui_metrics", { "hd95", "--case", case_dir, "--pairs", pairs });
    };

    // metrics.csv + metrics.json from the Metrics table as shown; source/gui_export.py
    // reads the structure colors from metrics.json for the RTSTRUCT.
    auto WriteMetricsFiles = [&](const std::filesystem::path& out_dir)
    {
        auto number_or_null = [](double v) { return std::isnan(v) ? nlohmann::json(nullptr) : nlohmann::json(v); };
        nlohmann::json structures = nlohmann::json::array();
        std::ofstream csv(out_dir / L"metrics.csv");
        csv << "structure,gt_id,pred_id,dice,hd95_mm,gt_cc,pred_cc\n";
        for (const StructureMetric& row : structure_metrics.rows)
        {
            std::array<uint8_t, 3> c = ColorForStructureName(row.name);
            structures.push_back({
                { "name", row.name }, { "gt_id", row.gt_id }, { "pred_id", row.pred_id },
                { "dice", number_or_null(row.dice) }, { "hd95_mm", number_or_null(row.hd95_mm) },
                { "gt_cc", row.gt_cc }, { "pred_cc", row.pred_cc }, { "color", { c[0], c[1], c[2] } },
            });
            csv << CsvField(row.name) << ',' << row.gt_id << ',' << row.pred_id << ','
                << (std::isnan(row.dice) ? std::string() : std::to_string(row.dice)) << ','
                << (std::isnan(row.hd95_mm) ? std::string() : std::to_string(row.hd95_mm)) << ','
                << row.gt_cc << ',' << row.pred_cc << '\n';
        }
        nlohmann::json metrics = {
            { "case", case_dir }, { "mean_dice", number_or_null(structure_metrics.mean_dice) }, { "structures", structures },
        };
        std::ofstream(out_dir / L"metrics.json") << metrics.dump(2);
    };

    // Metrics > Export: metrics files from this table, then source/gui_export.py
    // for the prediction itself, all into data/<case>/exports/<timestamp>/.
    auto StartExportJob = [&]()
    {
        std::filesystem::path out_path = std::filesystem::path(
            Utf8ToWide(std::string(CONTOURLENS_DATA_DIR) + "/" + case_dir + "/exports/" + LocalTimestamp())).lexically_normal();
        job.kind = JobKind::Export;
        job.title = "Exporting";
        job.case_name = case_dir;
        job.output_dir = WideToUtf8(out_path.wstring());

        std::error_code ec;
        std::filesystem::create_directories(out_path, ec);
        if (ec)
        {
            job.cancelled = false;
            job.error_message = "Couldn't create " + job.output_dir + ": " + ec.message();
            return;
        }
        WriteMetricsFiles(out_path);
        StartJob(job, CONTOURLENS_REPO_DIR, "source.gui_export", { "--case", case_dir, "--out", job.output_dir });
    };

    auto LoadBatchResults = [&](const std::string& dir)
    {
        std::ifstream f(std::filesystem::path(Utf8ToWide(dir)) / L"results.json");
        nlohmann::json j = nlohmann::json::parse(f, nullptr, false);
        if (j.is_discarded() || !j.is_object())
            return;
        batch_results = std::move(j);
        batch_results_dir = dir;
    };

    auto StartBatchJob = [&]()
    {
        SaveInferenceConfig(CONTOURLENS_REPO_DIR, case_dir, inference_cfg); // the batch reads model/backend from this case's config
        std::filesystem::path out_path = std::filesystem::path(Utf8ToWide(
            std::string(CONTOURLENS_DATA_DIR) + "/batch/" + LocalTimestamp() + "_fold" + std::to_string(batch_fold))).lexically_normal();
        job.kind = JobKind::Batch;
        job.title = "Batch evaluation (fold " + std::to_string(batch_fold) + ")";
        job.case_name = case_dir;
        job.output_dir = WideToUtf8(out_path.wstring());
        StartJob(job, CONTOURLENS_REPO_DIR, "source.batch_evaluate", {
            "--case", case_dir, "--fold", std::to_string(batch_fold),
            "--parallel", std::to_string(batch_parallel), "--out", job.output_dir });
    };

    // The Batch Evaluation window's results: per-structure mean +/- std, then per case.
    auto DrawBatchResults = [&]()
    {
        if (!batch_results.is_object())
        {
            ImGui::TextDisabled("No results yet: run a batch above.");
            return;
        }
        try
        {
            const nlohmann::json& summary = batch_results.at("summary");
            const nlohmann::json& cases = batch_results.at("cases");
            ImGui::Text("Fold %d  |  backend %s  |  %d cases", batch_results.value("fold", -1),
                batch_results.value("backend", std::string("?")).c_str(), (int)cases.size());
            ImGui::SameLine();
            if (ImGui::SmallButton("Open folder##batch"))
                OpenFolderInExplorer(batch_results_dir);

            auto mean_std = [](const nlohmann::json& stats, const char* format) -> std::string
            {
                if (!stats.is_object() || !stats.contains("mean") || !stats.at("mean").is_number())
                    return "-";
                char buf[64];
                snprintf(buf, sizeof(buf), format, stats.at("mean").get<double>(), stats.at("std").get<double>(), stats.at("n").get<int>());
                return buf;
            };

            ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp;
            if (ImGui::BeginTable("##batch_structures", 3, flags, ImVec2(0.0f, ImGui::GetTextLineHeightWithSpacing() * 12.0f)))
            {
                ImGui::TableSetupScrollFreeze(0, 1);
                ImGui::TableSetupColumn("Structure", 0, 2.0f);
                ImGui::TableSetupColumn("Dice mean +/- std (n)", 0, 1.6f);
                ImGui::TableSetupColumn("HD95 mm mean +/- std (n)", 0, 1.6f);
                ImGui::TableHeadersRow();
                for (const auto& item : summary.at("structures").items())
                {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(item.key().c_str());
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(mean_std(item.value().at("dice"), "%.3f +/- %.3f (%d)").c_str());
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(mean_std(item.value().at("hd95_mm"), "%.1f +/- %.1f (%d)").c_str());
                }
                ImGui::EndTable();
            }

            if (ImGui::TreeNode("Per case"))
            {
                const nlohmann::json& case_summary = summary.at("cases");
                for (const auto& item : cases.items())
                {
                    if (item.value().contains("error"))
                    {
                        const nlohmann::json& error = item.value().at("error");
                        ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "%s: failed (%s)", item.key().c_str(),
                            error.is_string() ? error.get<std::string>().c_str() : error.dump().c_str());
                    }
                    else if (case_summary.contains(item.key()) && case_summary.at(item.key()).at("mean_dice").is_number())
                        ImGui::Text("%s: mean Dice %.3f", item.key().c_str(), case_summary.at(item.key()).at("mean_dice").get<double>());
                    else
                        ImGui::Text("%s: no scored structures", item.key().c_str());
                }
                ImGui::TreePop();
            }
        }
        catch (const nlohmann::json::exception& e)
        {
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "Couldn't read results.json: %s", e.what());
        }
    };

    // On the frame PollJob reports a finished job: reload whatever it wrote.
    auto OnJobFinished = [&]()
    {
        if (job.cancelled)
            return;
        if (job.exit_code != 0)
        {
            if (job.error_message.empty())
                job.error_message = "Exited with code " + std::to_string(job.exit_code) + "; see the log below.";
            return;
        }
        // After a job that can leave both label volumes loaded without HD95
        // results, compute them -- the job runner is free again at this point.
        auto maybe_start_hd95 = [&]()
        {
            if (!masks_loaded || !prediction_loaded || structure_metrics.scored == 0 || LoadHd95Results())
                return;
            std::vector<std::string> notices = job.notices; // keep e.g. "deployed the Modal app" visible past the follow-up
            StartHd95Job();
            job.notices.insert(job.notices.begin(), notices.begin(), notices.end());
        };

        switch (job.kind)
        {
        case JobKind::LoadCt:
            LoadCase(job.case_name, job.source);
            break;
        case JobKind::LoadMasks:
            if (job.case_name == case_dir)
            {
                LoadLabelSlot(job.slot == "prediction" ? LabelSlot::Prediction : LabelSlot::GroundTruth);
                maybe_start_hd95();
            }
            break;
        case JobKind::LoadDose:
            if (job.case_name == case_dir)
            {
                RecomputeDVH();
                show_dvh = dvh_needs_repositioning = true;
            }
            break;
        case JobKind::Inference:
            if (job.case_name == case_dir)
            {
                LoadLabelSlot(LabelSlot::Prediction);
                expand_metrics = true;
                maybe_start_hd95();
            }
            break;
        case JobKind::Hd95:
            if (job.case_name == case_dir)
            {
                LoadHd95Results();
                metrics_order.clear(); // re-sort, in case the table is sorted by HD95
            }
            break;
        case JobKind::Batch:
            LoadBatchResults(job.output_dir);
            show_batch = true;
            break;
        case JobKind::Export:
        case JobKind::None:
            break;
        }
    };

    // Top of the Controls panel: the current (or last) job's progress bar, status, and output log.
    auto DrawJobStatus = [&]()
    {
        if (job.kind == JobKind::None)
            return;

        ImGui::SeparatorText(job.title.c_str());
        DWORD end_ms = job.running ? GetTickCount() : job.end_tick_ms;
        float elapsed_s = (end_ms - job.start_tick_ms) / 1000.0f;
        if (job.running)
        {
            if (job.progress.fraction >= 0.0f)
            {
                char overlay[32];
                snprintf(overlay, sizeof(overlay), "%.0f%%", job.progress.fraction * 100.0f);
                ImGui::ProgressBar(job.progress.fraction, ImVec2(-FLT_MIN, 0.0f), overlay);
            }
            else
            {
                ImGui::ProgressBar(-1.0f * (float)ImGui::GetTime(), ImVec2(-FLT_MIN, 0.0f)); // negative = animated indeterminate bar
            }
            if (!job.progress.message.empty())
                ImGui::TextWrapped("%s", job.progress.message.c_str());
            ImGui::TextDisabled("%s%s%.0fs elapsed", job.progress.stage.c_str(), job.progress.stage.empty() ? "" : " | ", elapsed_s);
            if (ImGui::Button("Cancel##job"))
                CancelJob(job);
        }
        else if (job.cancelled)
        {
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "Cancelled after %.0fs.", elapsed_s);
        }
        else if (!job.error_message.empty())
        {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.45f, 1.0f));
            ImGui::TextWrapped("Failed: %s", job.error_message.c_str());
            ImGui::PopStyleColor();
        }
        else
        {
            ImGui::TextColored(ImVec4(0.45f, 0.9f, 0.55f, 1.0f), "Done in %.0fs.", elapsed_s);
            if (!job.output_dir.empty() && (job.kind == JobKind::Export || job.kind == JobKind::Batch))
            {
                ImGui::TextWrapped("Saved to %s", job.output_dir.c_str());
                if (ImGui::Button("Open folder##job"))
                    OpenFolderInExplorer(job.output_dir);
            }
        }

        for (const std::string& notice : job.notices)
        {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.78f, 1.0f, 1.0f));
            ImGui::TextWrapped("%s", notice.c_str());
            ImGui::PopStyleColor();
        }

        if (!job.log.empty() && ImGui::TreeNode("Log##job"))
        {
            ImGui::BeginChild("##joblog", ImVec2(0.0f, ImGui::GetTextLineHeightWithSpacing() * 10.0f),
                ImGuiChildFlags_Borders, ImGuiWindowFlags_HorizontalScrollbar);
            for (const std::string& line : job.log)
                ImGui::TextUnformatted(line.c_str());
            if (job.log_updated && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f)
                ImGui::SetScrollHereY(1.0f); // follow new output unless the user scrolled up
            job.log_updated = false;
            ImGui::EndChild();
            ImGui::TreePop();
        }
        ImGui::Separator();
    };

    // Per-structure Dice, HD95 and volumes (StructureMetrics), sortable; by
    // default worst Dice first, so what the model gets wrong is at the top.
    // Clicking a row jumps the cursor to where that structure disagrees most.
    auto DrawMetricsTable = [&]()
    {
        const StructureMetrics& m = structure_metrics;
        const ImVec4 amber(1.0f, 0.8f, 0.3f, 1.0f);
        const bool both = masks_loaded && prediction_loaded;
        if (!masks_loaded)
            ImGui::TextWrapped("Drag ground-truth masks onto the window to score this prediction. Volumes only for now:");
        else if (!prediction_loaded)
            ImGui::TextWrapped("Run inference, or drag in a prediction, to score it against the ground truth. Volumes only for now:");
        else if (m.scored > 0)
        {
            double hd95_sum = 0.0;
            int hd95_count = 0;
            for (const StructureMetric& row : m.rows)
                if (!std::isnan(row.hd95_mm)) { hd95_sum += row.hd95_mm; ++hd95_count; }
            if (hd95_count > 0)
                ImGui::Text("Mean Dice: %.3f   Mean HD95: %.1f mm   (%d scored)", m.mean_dice, hd95_sum / hd95_count, m.scored);
            else
                ImGui::Text("Mean Dice: %.3f  (%d structures scored)", m.mean_dice, m.scored);
        }
        else
            ImGui::TextColored(amber, "No structures pair up between ground truth and prediction yet -- pair them below.");
        if (m.matched_by_id)
            ImGui::TextColored(amber, "Matched by label id (one side has no structure names).");

        if (both)
        {
            if (ImGui::Checkbox("Error map", &show_error_map))
                k_dirty = j_dirty = i_dirty = true;
            auto legend = [&](const char* id, const std::array<uint8_t, 3>& c, const char* text)
            {
                ImGui::SameLine();
                ImGui::ColorButton(id, ImVec4(c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f, 1.0f),
                    ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoPicker, ImVec2(10, 10));
                ImGui::SameLine(0.0f, 4.0f);
                ImGui::TextUnformatted(text);
            };
            legend("##missed", error_colors[kErrorMissed - 1], "missed");
            legend("##extra", error_colors[kErrorExtra - 1], "extra");
            if (show_error_map)
            {
                ImGui::SetNextItemWidth(120.0f);
                if (ImGui::SliderFloat("Error map opacity", &error_map_alpha, 0.1f, 1.0f, "%.2f"))
                    k_dirty = j_dirty = i_dirty = true;
            }
        }

        ImGui::BeginDisabled(job.running || !both || m.scored == 0);
        if (ImGui::Button("Compute HD95"))
            StartHd95Job();
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(job.running || !prediction_loaded);
        if (ImGui::Button("Export"))
            StartExportJob();
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("Writes metrics.csv/.json, prediction.nii.gz, and an RTSTRUCT (when the CT came from DICOM)\nto data/<case>/exports/<timestamp>/");
        if (both)
            ImGui::TextDisabled("Click a row to jump to where that structure disagrees most.");

        enum { kColName, kColDice, kColHd95, kColGt, kColPred, kColDelta };
        ImGuiTableFlags flags = ImGuiTableFlags_Sortable | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
            ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp;
        float height = ImGui::GetTextLineHeightWithSpacing() * (float)(std::min<size_t>(m.rows.size(), 12) + 1) +
            ImGui::GetStyle().CellPadding.y * 4.0f;
        if (ImGui::BeginTable("##metrics", 6, flags, ImVec2(0.0f, height)))
        {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Structure", ImGuiTableColumnFlags_NoHide, 2.4f, kColName);
            ImGui::TableSetupColumn("Dice", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_PreferSortAscending, 1.0f, kColDice);
            ImGui::TableSetupColumn("HD95 mm", ImGuiTableColumnFlags_PreferSortDescending, 1.0f, kColHd95);
            ImGui::TableSetupColumn("GT cc", 0, 1.0f, kColGt);
            ImGui::TableSetupColumn("Pred cc", 0, 1.0f, kColPred);
            ImGui::TableSetupColumn("dVol", 0, 1.0f, kColDelta);
            ImGui::TableHeadersRow();

            ImGuiTableSortSpecs* sort_specs = ImGui::TableGetSortSpecs();
            if (metrics_order.size() != m.rows.size() || (sort_specs && sort_specs->SpecsDirty))
            {
                metrics_order.resize(m.rows.size());
                for (size_t i = 0; i < m.rows.size(); ++i)
                    metrics_order[i] = (int)i;
                if (sort_specs && sort_specs->SpecsCount > 0)
                {
                    const ImGuiTableColumnSortSpecs& spec = sort_specs->Specs[0];
                    bool ascending = spec.SortDirection == ImGuiSortDirection_Ascending;
                    const double nan = std::numeric_limits<double>::quiet_NaN();
                    auto value = [&](const StructureMetric& r) -> double
                    {
                        switch (spec.ColumnUserID)
                        {
                        case kColDice: return r.dice;
                        case kColHd95: return r.hd95_mm;
                        case kColGt: return r.gt_id ? r.gt_cc : nan;
                        case kColPred: return r.pred_id ? r.pred_cc : nan;
                        case kColDelta: return (r.gt_voxels > 0 && r.pred_id) ? (r.pred_cc - r.gt_cc) / r.gt_cc : nan;
                        default: return 0.0;
                        }
                    };
                    std::stable_sort(metrics_order.begin(), metrics_order.end(), [&](int a, int b)
                    {
                        const StructureMetric& ra = m.rows[a];
                        const StructureMetric& rb = m.rows[b];
                        if (spec.ColumnUserID == kColName)
                            return ascending ? ra.name < rb.name : rb.name < ra.name;
                        double va = value(ra), vb = value(rb);
                        if (std::isnan(va) || std::isnan(vb))
                            return !std::isnan(va) && std::isnan(vb); // unscored rows last in either direction
                        return ascending ? va < vb : vb < va;
                    });
                }
                if (sort_specs)
                    sort_specs->SpecsDirty = false;
            }

            for (int r : metrics_order)
            {
                const StructureMetric& row = m.rows[r];
                ImGui::TableNextRow();
                ImGui::PushID(r);

                ImGui::TableNextColumn();
                std::array<uint8_t, 3> c = ColorForStructureName(row.name);
                ImGui::ColorButton("##swatch", ImVec4(c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f, 1.0f),
                    ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoPicker, ImVec2(10, 10));
                ImGui::SameLine();
                if (ImGui::Selectable(row.name.c_str(), selected_metric_row == r, ImGuiSelectableFlags_SpanAllColumns))
                    JumpToStructure(r);

                ImGui::TableNextColumn();
                if (!std::isnan(row.dice))
                {
                    ImVec4 color = row.dice >= 0.8f ? ImVec4(0.45f, 0.9f, 0.55f, 1.0f)
                        : row.dice >= 0.5f ? amber : ImVec4(1.0f, 0.45f, 0.45f, 1.0f);
                    ImGui::TextColored(color, "%.3f", row.dice);
                }
                else if (both)
                    ImGui::TextDisabled("%s", row.gt_id ? "not predicted" : "no GT");
                else
                    ImGui::TextDisabled("-");

                ImGui::TableNextColumn();
                if (!std::isnan(row.hd95_mm))
                    ImGui::Text("%.1f", row.hd95_mm);
                else if (job.running && job.kind == JobKind::Hd95 && row.gt_id && row.pred_id)
                    ImGui::TextDisabled("...");
                else
                    ImGui::TextDisabled("-");

                ImGui::TableNextColumn();
                if (row.gt_id) ImGui::Text("%.2f", row.gt_cc); else ImGui::TextDisabled("-");
                ImGui::TableNextColumn();
                if (row.pred_id) ImGui::Text("%.2f", row.pred_cc); else ImGui::TextDisabled("-");
                ImGui::TableNextColumn();
                if (row.gt_voxels > 0 && row.pred_id) ImGui::Text("%+.0f%%", 100.0 * (row.pred_cc - row.gt_cc) / row.gt_cc); else ImGui::TextDisabled("-");
                ImGui::PopID();
            }
            ImGui::EndTable();
        }

        // Structures only one side has (e.g. HaN-Seg's "OpticNrv_L" vs the
        // model's "optic_nerve_l") can be paired here; that saves a global alias.
        if (both)
        {
            std::vector<int> gt_only, pred_only;
            for (size_t r = 0; r < m.rows.size(); ++r)
            {
                if (m.rows[r].gt_id && !m.rows[r].pred_id)
                    gt_only.push_back((int)r);
                else if (!m.rows[r].gt_id && m.rows[r].pred_id)
                    pred_only.push_back((int)r);
            }
            if (!gt_only.empty() && !pred_only.empty() && ImGui::TreeNode("Pair unmatched structures"))
            {
                ImGui::TextWrapped("Pairing saves a name alias to configs/structure_aliases.json, which applies to every case.");
                float combo_x = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x * 0.45f;
                for (int g : gt_only)
                {
                    ImGui::PushID(g + 10000);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted(m.rows[g].name.c_str());
                    ImGui::SameLine(combo_x);
                    ImGui::SetNextItemWidth(-FLT_MIN);
                    if (ImGui::BeginCombo("##pair", "Pair with..."))
                    {
                        for (int p : pred_only)
                            if (ImGui::Selectable(m.rows[p].name.c_str()))
                            {
                                const std::string& pred_name = m.rows[p].name;
                                std::string pred_key = StructureMatchKey(pred_name);
                                // Point at whatever the prediction's name itself resolves to, in case it's aliased too.
                                SaveStructureAlias(CONTOURLENS_REPO_DIR, m.rows[g].name,
                                    pred_key == NormalizeStructureName(pred_name) ? pred_name : pred_key);
                                aliases_changed = true;
                            }
                        ImGui::EndCombo();
                    }
                    ImGui::PopID();
                }
                ImGui::TreePop();
            }
        }

        if (ImGui::TreeNode("Structure aliases"))
        {
            const std::vector<std::pair<std::string, std::string>> entries = StructureAliasEntries(); // a copy: removing one reloads the table
            if (entries.empty())
                ImGui::TextDisabled("None yet.");
            for (size_t a = 0; a < entries.size(); ++a)
            {
                ImGui::PushID((int)a + 20000);
                if (ImGui::SmallButton("x"))
                {
                    RemoveStructureAlias(CONTOURLENS_REPO_DIR, entries[a].first);
                    aliases_changed = true;
                }
                ImGui::SameLine();
                ImGui::Text("%s -> %s", entries[a].first.c_str(), entries[a].second.c_str());
                ImGui::PopID();
            }
            ImGui::TreePop();
        }
    };

    LoadStructureAliases(CONTOURLENS_REPO_DIR); // before the first LoadCase: metrics and colors match names through it
    LoadCase(case_dir, "");

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

        // Before the occluded early-out below: a minimized window must keep
        // draining the job's output pipe, or the child blocks writing to it.
        if (PollJob(job))
            OnJobFinished();

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

        // Files dropped onto the window (WM_DROPFILES): ask what to load them as.
        if (!g_DroppedPaths.empty())
        {
            drop_paths = std::move(g_DroppedPaths);
            g_DroppedPaths.clear();
            std::sort(drop_paths.begin(), drop_paths.end());
            std::error_code ec;
            drop_kind = ClassifyDroppedPath(drop_paths[0]);
            drop_is_directory = std::filesystem::is_directory(NormalizedPath(drop_paths[0]), ec);
            drop_ct_case_name = CaseNameForDroppedCt(drop_paths[0]);
            drop_target = DefaultDropTarget(drop_kind, drop_paths[0], ct_loaded);
            ImGui::OpenPopup("Load dropped data");
        }

        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        if (ImGui::BeginPopupModal("Load dropped data", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            const size_t kMaxListed = 6;
            for (size_t i = 0; i < drop_paths.size() && i < kMaxListed; ++i)
                ImGui::BulletText("%s", drop_paths[i].c_str());
            if (drop_paths.size() > kMaxListed)
                ImGui::TextDisabled("... and %d more", (int)(drop_paths.size() - kMaxListed));

            bool can_load_ct = drop_kind == DropKind::DicomSeries || drop_kind == DropKind::Volume;
            // Any single-file DICOM may be an RTSTRUCT/RTDOSE regardless of its name; gui_load.py checks the Modality.
            bool is_dicom_file = !drop_is_directory &&
                (drop_kind == DropKind::DicomSeries || drop_kind == DropKind::RtStruct || drop_kind == DropKind::RtDose);
            bool can_load_masks = ct_loaded && drop_kind != DropKind::Unsupported && drop_kind != DropKind::RtDose &&
                !(drop_kind == DropKind::DicomSeries && drop_is_directory);
            bool can_load_dose = ct_loaded && is_dicom_file && drop_paths.size() == 1;
            bool can_close_only = job.running || drop_kind == DropKind::Unsupported;

            if (job.running)
            {
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "Busy: wait for \"%s\" to finish (or cancel it) first.", job.title.c_str());
            }
            else if (drop_kind == DropKind::Unsupported)
            {
                ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "Unsupported file type.");
                ImGui::TextUnformatted("Drop a DICOM folder or .dcm slice, a .nii/.nii.gz/.nrrd volume,\n.seg.nrrd masks, or an RTSTRUCT/RTDOSE .dcm.");
            }
            else
            {
                ImGui::SeparatorText("Load as");
                ImGui::BeginDisabled(!can_load_ct);
                ImGui::RadioButton("CT (opens a new case)", &drop_target, 0);
                ImGui::EndDisabled();
                ImGui::BeginDisabled(!can_load_masks);
                ImGui::RadioButton("Ground-truth masks", &drop_target, 1);
                ImGui::RadioButton("Prediction", &drop_target, 2);
                ImGui::EndDisabled();
                ImGui::BeginDisabled(!can_load_dose);
                ImGui::RadioButton("Dose (RTDOSE, for the DVH)", &drop_target, 3);
                ImGui::EndDisabled();

                if (drop_target == 0)
                    ImGui::TextDisabled("New case: %s", drop_ct_case_name.c_str());
                else if (ct_loaded)
                    ImGui::TextDisabled("Resampled onto the CT of case: %s", case_dir.c_str());
                if (!ct_loaded && drop_kind != DropKind::DicomSeries && drop_kind != DropKind::Volume)
                    ImGui::TextDisabled("Load a CT first: masks and dose are resampled onto the loaded CT.");

                bool valid = drop_target == 0 ? can_load_ct : drop_target == 3 ? can_load_dose : can_load_masks;
                ImGui::BeginDisabled(!valid);
                if (ImGui::Button("Load", ImVec2(120.0f, 0.0f)))
                {
                    if (drop_target == 0)
                        StartLoadCtJob(drop_paths[0]);
                    else if (drop_target == 3)
                        StartLoadDoseJob(drop_paths[0]);
                    else
                        StartLoadMasksJob(drop_paths, drop_target == 2);
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
            }
            if (ImGui::Button(can_close_only ? "Close" : "Cancel", ImVec2(120.0f, 0.0f)))
                ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        bool overlay_toggled_from_menu = false;
        if (ImGui::BeginMainMenuBar())
        {
            if (ImGui::BeginMenu("File"))
            {
                ImGui::MenuItem("Load data: drag a CT, masks, or dose onto the window", nullptr, false, false);
                if (ImGui::BeginMenu("Recent cases", !job.running))
                {
                    if (ImGui::IsWindowAppearing())
                        recent_cases = FindRecentCases(CONTOURLENS_DATA_DIR);
                    if (recent_cases.empty())
                        ImGui::MenuItem("(no exported cases under data/)", nullptr, false, false);
                    std::string chosen;
                    for (const std::string& recent : recent_cases)
                        if (ImGui::MenuItem(recent.c_str(), nullptr, recent == case_dir))
                            chosen = recent;
                    ImGui::EndMenu();
                    if (!chosen.empty() && chosen != case_dir)
                        LoadCase(chosen, "");
                }
                ImGui::Separator();
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
                if (ImGui::MenuItem("Error map", nullptr, &show_error_map, !error_volume.data.empty()))
                    overlay_toggled_from_menu = true;
                if (ImGui::MenuItem("DVH", nullptr, &show_dvh, masks_loaded || prediction_loaded) && show_dvh)
                    dvh_needs_repositioning = true; // snap it back into the main window, wherever it drifted to before
                ImGui::MenuItem("Batch Evaluation", nullptr, &show_batch);
                ImGui::Separator();
                if (ImGui::MenuItem("Reset Layout"))
                    dock_layout_built = false; // rebuilds Controls/Axial/Coronal/Sagittal/3D at their default proportions next frame
                ImGui::EndMenu();
            }

            // Right-aligned job status, visible even with the Controls panel scrolled/collapsed.
            if (job.running)
            {
                char status[128];
                if (job.progress.fraction >= 0.0f)
                    snprintf(status, sizeof(status), "%s: %.0f%%", job.title.c_str(), job.progress.fraction * 100.0f);
                else
                    snprintf(status, sizeof(status), "%s... %.0fs", job.title.c_str(), (GetTickCount() - job.start_tick_ms) / 1000.0f);
                ImGui::SetCursorPosX(ImGui::GetWindowWidth() - ImGui::CalcTextSize(status).x - ImGui::GetStyle().ItemSpacing.x * 2.0f);
                ImGui::TextUnformatted(status);
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
        // These are plain letter keys, so they're suspended while a text field
        // has keyboard focus (io.WantTextInput -- also set when a slider is
        // Ctrl+clicked into text entry) or any popup/menu is open; otherwise
        // typing "q" into Inference Config would yank focus to another panel.
        bool shortcuts_enabled = !io.WantTextInput &&
            !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
        if (ct_loaded && shortcuts_enabled)
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
        cmp.metrics = &structure_metrics;

        ImGui::Begin("Controls");
        DrawJobStatus();
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
                "shared-across-all-three and per-panel. Mouse wheel over a "
                "slice zooms in on that spot; right-drag pans, middle-click resets.");
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

            if (ImGui::CollapsingHeader("Inference Config", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::TextWrapped("Source CT: a NIfTI/NRRD file or a DICOM series directory, path relative to the repo root.");
                ImGui::InputText("Source CT##infcfg", &inference_cfg.source_ct);

                ImGui::Spacing();
                ImGui::TextUnformatted("Model");
                ImGui::InputText("Dataset ID##infcfg", &inference_cfg.model.dataset_id);
                ImGui::InputText("Configuration##infcfg", &inference_cfg.model.configuration);
                ImGui::InputText("Trainer##infcfg", &inference_cfg.model.trainer);
                ImGui::InputText("Plans##infcfg", &inference_cfg.model.plans);
                ImGui::TextUnformatted("Folds (nnU-Net 5-fold CV):");
                for (int i = 0; i < 5; ++i)
                {
                    if (i > 0) ImGui::SameLine();
                    ImGui::PushID(i);
                    bool checked = inference_cfg.model.fold_enabled[i];
                    if (ImGui::Checkbox(std::to_string(i).c_str(), &checked))
                    {
                        if ((::GetKeyState(VK_CONTROL) & 0x8000) != 0)
                            inference_cfg.model.fold_enabled[i] = checked; // Ctrl+click: just this fold, e.g. fold 2 alone
                        else
                            for (int f = 0; f < 5; ++f)
                                inference_cfg.model.fold_enabled[f] = f <= i; // the number of folds: 0..i
                    }
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Click: use folds 0-%d.  Ctrl+click: toggle only fold %d.", i, i);
                    ImGui::PopID();
                }
                ImGui::InputText("Results dir override (blank = default)##infcfg", &inference_cfg.model.results_dir);

                ImGui::Spacing();
                ImGui::TextUnformatted("Backend");
                if (ImGui::BeginTabBar("InferenceBackendTabs"))
                {
                    ImGuiTabItemFlags local_flags = ImGuiTabItemFlags_None;
                    ImGuiTabItemFlags modal_flags = ImGuiTabItemFlags_None;
                    if (force_select_backend_tab)
                    {
                        (inference_cfg.backend == "modal" ? modal_flags : local_flags) = ImGuiTabItemFlags_SetSelected;
                        force_select_backend_tab = false;
                    }

                    if (ImGui::BeginTabItem("Local", nullptr, local_flags))
                    {
                        inference_cfg.backend = "local";
                        ImGui::TextWrapped("Runs nnUNetv2_predict on this machine using the model settings above.");
                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem("Cloud (Modal)", nullptr, modal_flags))
                    {
                        inference_cfg.backend = "modal";
                        ImGui::TextWrapped("Calls the Modal function below with the CT's bytes -- no pre-staged data needed. If its app isn't deployed yet, Run Inference deploys source/modal_nnunet.py first (one-time) and tells you.");
                        ImGui::InputText("App name##infcfg", &inference_cfg.modal.app_name);
                        ImGui::InputText("Function name##infcfg", &inference_cfg.modal.function_name);
                        ImGui::EndTabItem();
                    }
                    ImGui::EndTabBar();
                }

                ImGui::Spacing();
                if (ImGui::Button("Save Config"))
                    SaveInferenceConfig(CONTOURLENS_REPO_DIR, case_dir, inference_cfg);
                ImGui::SameLine();
                if (ImGui::Button("Reload Config"))
                {
                    LoadInferenceConfig(CONTOURLENS_REPO_DIR, case_dir, inference_cfg);
                    force_select_backend_tab = true;
                }
            }

            if (ImGui::CollapsingHeader("Prediction"))
            {
                ImGui::BeginDisabled(job.running);
                if (ImGui::Button("Run Inference"))
                    StartInference();
                ImGui::EndDisabled();
                if (job.running)
                {
                    ImGui::SameLine();
                    ImGui::TextDisabled("(%s in progress, see the top of this panel)", job.title.c_str());
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

            if (masks_loaded || prediction_loaded)
            {
                if (expand_metrics)
                {
                    ImGui::SetNextItemOpen(true);
                    expand_metrics = false;
                }
                if (ImGui::CollapsingHeader("Metrics"))
                    DrawMetricsTable();
            }
        }
        else
        {
            ImGui::TextWrapped("No CT loaded for case '%s'.", case_dir.c_str());
            ImGui::Spacing();
            ImGui::TextWrapped("Drag a DICOM folder or .dcm slice, or a .nii/.nii.gz/.nrrd volume, onto the window to open it.");
        }
        ImGui::End();
        if (aliases_changed)
        {
            ReapplyAliases();
            aliases_changed = false;
        }

        ImGui::Begin("Axial", nullptr, ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoScrollbar); // the wheel zooms the slice instead
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
                    { [&](int i, int j) { return error_volume.data.empty() ? (uint8_t)0 : error_volume.At(i, j, cursor_k); },
                      show_error_map, error_map_alpha, nullptr, &error_colors },
                };
                UploadSlice(axial_tex, ct_volume.nx, ct_volume.ny, window_width[kAxial], window_level[kAxial],
                    [&](int i, int j) { return ct_volume.At(i, j, cursor_k); }, layers);
                k_dirty = false;
            }

            ImVec2 size = FitImageSize(ImGui::GetContentRegionAvail(), ct_volume.nx, ct_volume.ny, ct_volume.spacing[0], ct_volume.spacing[1]);
            SliceView& view = slice_views[kAxial];
            ImVec2 uv0, uv1;
            SliceViewUV(view, false, &uv0, &uv1);
            ImGui::Image((ImTextureID)(intptr_t)axial_srv, size, uv0, uv1);

            ImVec2 item_min = ImGui::GetItemRectMin();
            UpdateSliceView(view, item_min, size);
            int click_i, click_j;
            if (PanelClicked(item_min, size, ct_volume.nx, ct_volume.ny, false, view, &click_i, &click_j))
            {
                cursor_i = click_i; cursor_j = click_j;
                j_dirty = i_dirty = true; // coronal/sagittal now pass through a different point
            }
            DrawCrosshairAndReadout(item_min, size, ct_volume.nx, ct_volume.ny, cursor_i, cursor_j, false,
                view, ct_volume, cursor_i, cursor_j, cursor_k, cmp);
        }
        else
        {
            ImGui::TextUnformatted("Axial (transversal) - no CT loaded (drag one onto the window)");
        }
        ImGui::End();

        ImGui::Begin("Coronal", nullptr, ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoScrollbar);
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
                    { [&](int i, int k) { return error_volume.data.empty() ? (uint8_t)0 : error_volume.At(i, cursor_j, k); },
                      show_error_map, error_map_alpha, nullptr, &error_colors },
                };
                UploadSlice(coronal_tex, ct_volume.nx, ct_volume.nz, window_width[kCoronal], window_level[kCoronal],
                    [&](int i, int k) { return ct_volume.At(i, cursor_j, k); }, layers);
                j_dirty = false;
            }

            ImVec2 size = FitImageSize(ImGui::GetContentRegionAvail(), ct_volume.nx, ct_volume.nz, ct_volume.spacing[0], ct_volume.spacing[2]);
            SliceView& view = slice_views[kCoronal];
            ImVec2 uv0, uv1;
            SliceViewUV(view, true, &uv0, &uv1);
            ImGui::Image((ImTextureID)(intptr_t)coronal_srv, size, uv0, uv1);

            ImVec2 item_min = ImGui::GetItemRectMin();
            UpdateSliceView(view, item_min, size);
            int click_i, click_k;
            if (PanelClicked(item_min, size, ct_volume.nx, ct_volume.nz, true, view, &click_i, &click_k))
            {
                cursor_i = click_i; cursor_k = click_k;
                i_dirty = k_dirty = true;
            }
            DrawCrosshairAndReadout(item_min, size, ct_volume.nx, ct_volume.nz, cursor_i, cursor_k, true,
                view, ct_volume, cursor_i, cursor_j, cursor_k, cmp);
        }
        else
        {
            ImGui::TextUnformatted("Coronal (frontal) - no CT loaded (drag one onto the window)");
        }
        ImGui::End();

        ImGui::Begin("Sagittal", nullptr, ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoScrollbar);
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
                    { [&](int j, int k) { return error_volume.data.empty() ? (uint8_t)0 : error_volume.At(cursor_i, j, k); },
                      show_error_map, error_map_alpha, nullptr, &error_colors },
                };
                UploadSlice(sagittal_tex, ct_volume.ny, ct_volume.nz, window_width[kSagittal], window_level[kSagittal],
                    [&](int j, int k) { return ct_volume.At(cursor_i, j, k); }, layers);
                i_dirty = false;
            }

            ImVec2 size = FitImageSize(ImGui::GetContentRegionAvail(), ct_volume.ny, ct_volume.nz, ct_volume.spacing[1], ct_volume.spacing[2]);
            SliceView& view = slice_views[kSagittal];
            ImVec2 uv0, uv1;
            SliceViewUV(view, true, &uv0, &uv1);
            ImGui::Image((ImTextureID)(intptr_t)sagittal_srv, size, uv0, uv1);

            ImVec2 item_min = ImGui::GetItemRectMin();
            UpdateSliceView(view, item_min, size);
            int click_j, click_k;
            if (PanelClicked(item_min, size, ct_volume.ny, ct_volume.nz, true, view, &click_j, &click_k))
            {
                cursor_j = click_j; cursor_k = click_k;
                j_dirty = k_dirty = true;
            }
            DrawCrosshairAndReadout(item_min, size, ct_volume.ny, ct_volume.nz, cursor_j, cursor_k, true,
                view, ct_volume, cursor_i, cursor_j, cursor_k, cmp);
        }
        else
        {
            ImGui::TextUnformatted("Sagittal - no CT loaded (drag one onto the window)");
        }
        ImGui::End();

        ImGui::Begin("3D");
        if (ImGui::IsWindowFocused()) focused_panel = PanelId::ThreeD;
        if (ct_loaded && raycaster_ok)
        {
            // Space: full reset (also the way out of axis-lock mode).
            if (shortcuts_enabled && ImGui::IsWindowHovered() && ImGui::IsKeyPressed(ImGuiKey_Space, false))
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
            ImGui::TextUnformatted("3D - no CT loaded (drag one onto the window), or the raycaster failed to initialize (see console)");
        }
        ImGui::End();

        if (show_batch)
        {
            ImGui::SetNextWindowSize(ImVec2(640.0f, 480.0f), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Batch Evaluation", &show_batch))
            {
                ImGui::TextWrapped("Runs the model from case %s's Inference Config on one fold's validation cases "
                    "(nnU-Net's splits_final.json), using only that fold's weights, so every score is on data the model never trained on.",
                    case_dir.c_str());
                ImGui::TextDisabled("Backend: %s   Dataset %s   %s", inference_cfg.backend.c_str(),
                    inference_cfg.model.dataset_id.c_str(), inference_cfg.model.trainer.c_str());
                ImGui::SetNextItemWidth(80.0f);
                ImGui::Combo("Fold", &batch_fold, "0\0" "1\0" "2\0" "3\0" "4\0");
                // Modal: cloud GPUs (up to 8 at once); local: one case per NVIDIA GPU this machine actually has.
                const bool local_backend = inference_cfg.backend != "modal";
                const int max_parallel = local_backend ? std::max(1, local_gpu_count) : 8;
                batch_parallel = std::clamp(batch_parallel, 1, max_parallel);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(100.0f);
                ImGui::BeginDisabled(max_parallel == 1);
                if (ImGui::InputInt(local_backend ? "GPUs##batch" : "Parallel##batch", &batch_parallel))
                    batch_parallel = std::clamp(batch_parallel, 1, max_parallel);
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                {
                    if (local_backend)
                        ImGui::SetTooltip("Cases predicted at once, one per local NVIDIA GPU (%d detected).", local_gpu_count);
                    else
                        ImGui::SetTooltip("Cases predicted at once on Modal, one cloud GPU each (up to %d).", max_parallel);
                }
                if (local_backend)
                {
                    const ImVec4 amber(1.0f, 0.8f, 0.3f, 1.0f);
                    if (local_gpu_count == 0)
                        ImGui::TextColored(amber, "No NVIDIA GPU detected: local nnU-Net would run on the CPU, one case at a time (very slow).");
                    else
                        ImGui::TextDisabled("%d local NVIDIA GPU%s detected, so at most %d case%s at a time.",
                            local_gpu_count, local_gpu_count == 1 ? "" : "s", local_gpu_count, local_gpu_count == 1 ? "" : "s");
                    ImGui::TextColored(amber, "Local nnU-Net has run this machine out of RAM before; each concurrent case needs its own RAM.");
                }
                ImGui::BeginDisabled(job.running);
                if (ImGui::Button("Run batch"))
                    StartBatchJob();
                ImGui::EndDisabled();
                if (job.running && job.kind == JobKind::Batch)
                {
                    ImGui::SameLine();
                    ImGui::TextDisabled("%s", job.progress.message.c_str());
                }
                ImGui::Separator();
                DrawBatchResults();
            }
            ImGui::End();
        }

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
            if (!dvh_curves.empty() || !dvh_pred_curves.empty())
            {
                if (dvh_dose_is_synthetic)
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Synthetic dose (Gaussian blob) - NOT a real treatment plan. Drag an RTDOSE .dcm onto the window for a real one.");
                else
                    ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Dose: %s/dose (real RTDOSE export)", case_data_dir.c_str());

                // Structures on both sides (dvh_curves / dvh_pred_curves are indexed by label id - 1), for the stats table.
                std::vector<const StructureMetric*> paired;
                for (const StructureMetric& row : structure_metrics.rows)
                    if (row.gt_id && row.pred_id && row.gt_id <= (int)dvh_curves.size() && row.pred_id <= (int)dvh_pred_curves.size())
                        paired.push_back(&row);
                if (!dvh_curves.empty() && !dvh_pred_curves.empty())
                    ImGui::Checkbox("Show prediction (thin lines)", &dvh_show_prediction);

                const ImGuiStyle& style = ImGui::GetStyle();
                float table_height = paired.empty() ? 0.0f
                    : ImGui::GetTextLineHeightWithSpacing() * (float)(std::min<size_t>(paired.size(), 8) + 1) + style.CellPadding.y * 4.0f + style.ItemSpacing.y;
                if (ImPlot::BeginPlot("Dose-Volume Histogram", ImVec2(-1.0f, -1.0f - table_height)))
                {
                    ImPlot::SetupAxes("Dose (Gy)", "Volume (%)");
                    auto plot_curve = [](const DVHCurve& curve, const std::string& label, float weight, float alpha)
                    {
                        const std::array<uint8_t, 3> c = ColorForStructureName(curve.name);
                        ImPlotSpec spec;
                        spec.LineColor = ImVec4(c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f, alpha);
                        spec.LineWeight = weight;
                        ImPlot::PlotLine(label.c_str(), curve.dose_gy.data(), curve.volume_pct.data(), (int)curve.dose_gy.size(), spec);
                    };
                    for (const DVHCurve& curve : dvh_curves)
                        plot_curve(curve, curve.name, 2.0f, 1.0f);
                    if (dvh_show_prediction || dvh_curves.empty())
                        for (const DVHCurve& curve : dvh_pred_curves)
                            plot_curve(curve, curve.name + " (pred)", 1.0f, 0.6f);
                    ImPlot::EndPlot();
                }

                if (!paired.empty() && ImGui::BeginTable("##dvh_stats", 5,
                        ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp,
                        ImVec2(0.0f, table_height - style.ItemSpacing.y)))
                {
                    ImGui::TableSetupScrollFreeze(0, 1);
                    ImGui::TableSetupColumn("Structure", 0, 2.0f);
                    ImGui::TableSetupColumn("Dmean GT / pred (Gy)", 0, 1.6f);
                    ImGui::TableSetupColumn("dDmean", 0, 0.8f);
                    ImGui::TableSetupColumn("Dmax GT / pred (Gy)", 0, 1.6f);
                    ImGui::TableSetupColumn("dDmax", 0, 0.8f);
                    ImGui::TableHeadersRow();
                    for (const StructureMetric* row : paired)
                    {
                        const DVHCurve& gt_curve = dvh_curves[row->gt_id - 1];
                        const DVHCurve& pred_curve = dvh_pred_curves[row->pred_id - 1];
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted(row->name.c_str());
                        ImGui::TableNextColumn();
                        ImGui::Text("%.1f / %.1f", gt_curve.mean_gy, pred_curve.mean_gy);
                        ImGui::TableNextColumn();
                        ImGui::Text("%+.1f", pred_curve.mean_gy - gt_curve.mean_gy);
                        ImGui::TableNextColumn();
                        ImGui::Text("%.1f / %.1f", gt_curve.max_gy, pred_curve.max_gy);
                        ImGui::TableNextColumn();
                        ImGui::Text("%+.1f", pred_curve.max_gy - gt_curve.max_gy);
                    }
                    ImGui::EndTable();
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
    ReleaseJob(job); // let a still-running job finish on its own

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
    case WM_DROPFILES:
    {
        HDROP drop = (HDROP)wParam;
        UINT count = ::DragQueryFileW(drop, 0xFFFFFFFF, nullptr, 0);
        g_DroppedPaths.clear();
        for (UINT i = 0; i < count; ++i)
        {
            UINT length = ::DragQueryFileW(drop, i, nullptr, 0);
            std::vector<wchar_t> path(length + 1, L'\0');
            ::DragQueryFileW(drop, i, path.data(), length + 1);
            g_DroppedPaths.push_back(WideToUtf8(std::wstring(path.data(), length)));
        }
        ::DragFinish(drop);
        return 0;
    }
    case WM_DESTROY:
        ::PostQuitMessage(0);
        return 0;
    }
    return ::DefWindowProcW(hWnd, msg, wParam, lParam);
}
