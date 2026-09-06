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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <d3d11.h>
#include <tchar.h>

// Distinct colors for up to 9 overlay structures (wraps via modulo past that).
static const uint8_t kLabelColors[][3] = {
    { 230, 60, 60 },   { 60, 160, 230 },  { 250, 200, 60 },
    { 120, 220, 120 }, { 200, 100, 220 }, { 80, 220, 200 },
    { 240, 140, 60 },  { 160, 160, 250 }, { 250, 250, 120 },
};
static const int kNumLabelColors = sizeof(kLabelColors) / sizeof(kLabelColors[0]);

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

// Eclipse-style 4-panel MPR layout: axial | coronal
//                                   sagittal | 3D
// Built once (not every frame, or user drags get wiped). Layout is fixed for
// now (io.IniFilename == nullptr) rather than user-re-dockable + persisted.
static void BuildDockLayout(ImGuiID dockspace_id)
{
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

    ImGuiID top, bottom;
    ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Up, 0.5f, &top, &bottom);

    ImGuiID top_left, top_right;
    ImGui::DockBuilderSplitNode(top, ImGuiDir_Left, 0.5f, &top_left, &top_right);

    ImGuiID bottom_left, bottom_right;
    ImGui::DockBuilderSplitNode(bottom, ImGuiDir_Left, 0.5f, &bottom_left, &bottom_right);

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

// Windows HU to a grayscale byte: (hu - (wl - ww/2)) / ww, clamped to [0,1],
// then optionally tints it towards that voxel's overlay structure color.
// `get_voxel`/`get_label(col, row)` pick the plane; callers below index the
// volume differently for axial/coronal/sagittal but share this upload path.
template <typename GetVoxel, typename GetLabel>
static void UploadSlice(ID3D11Texture2D* tex, int width, int height, float window_width, float window_level,
    GetVoxel get_voxel, GetLabel get_label, bool overlay_enabled, float overlay_alpha, const std::vector<bool>& label_visible)
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
            uint8_t r = gray, g = gray, b = gray;

            if (overlay_enabled)
            {
                uint8_t label = get_label(col, row);
                size_t li = (size_t)label - 1;
                if (label != 0 && li < label_visible.size() && label_visible[li])
                {
                    const uint8_t* c = kLabelColors[li % kNumLabelColors];
                    r = (uint8_t)(gray * (1.0f - overlay_alpha) + c[0] * overlay_alpha);
                    g = (uint8_t)(gray * (1.0f - overlay_alpha) + c[1] * overlay_alpha);
                    b = (uint8_t)(gray * (1.0f - overlay_alpha) + c[2] * overlay_alpha);
                }
            }

            dst[col * 4 + 0] = r;
            dst[col * 4 + 1] = g;
            dst[col * 4 + 2] = b;
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

// Crosshair through the shared cursor voxel (ci, cj, ck), projected into
// this panel's (col, row) in-plane coordinates, plus an (i,j,k)/HU readout.
static void DrawCrosshairAndReadout(ImVec2 item_min, ImVec2 item_size, int tex_w, int tex_h, int col, int row, bool flipped,
    const Volume& vol, int ci, int cj, int ck)
{
    int display_row = flipped ? (tex_h - 1 - row) : row;
    float sx = item_min.x + ((float)col + 0.5f) / tex_w * item_size.x;
    float sy = item_min.y + ((float)display_row + 0.5f) / tex_h * item_size.y;

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImU32 color = IM_COL32(255, 230, 0, 180);
    draw_list->AddLine(ImVec2(sx, item_min.y), ImVec2(sx, item_min.y + item_size.y), color);
    draw_list->AddLine(ImVec2(item_min.x, sy), ImVec2(item_min.x + item_size.x, sy), color);

    // Drawn as an overlay on the image (not a separate ImGui::Text line) so it
    // can't push the panel's content past the dock quadrant's fixed height.
    char buf[64];
    snprintf(buf, sizeof(buf), "(i=%d, j=%d, k=%d)  HU=%d", ci, cj, ck, vol.At(ci, cj, ck));
    ImVec2 text_pos(item_min.x + 4.0f, item_min.y + 4.0f);
    ImVec2 text_size = ImGui::CalcTextSize(buf);
    draw_list->AddRectFilled(text_pos, ImVec2(text_pos.x + text_size.x + 4.0f, text_pos.y + text_size.y + 4.0f), IM_COL32(0, 0, 0, 160));
    draw_list->AddText(ImVec2(text_pos.x + 2.0f, text_pos.y + 2.0f), IM_COL32(255, 255, 255, 255), buf);
}

// Draws one slice plane's outline (+ a faint fill) into the 3D raycast
// image, projecting its 4 world-space corners through the same camera used
// to render that frame's raycast (see raycaster.h's CameraFrame) so the
// outline lines up with the volume exactly. Skips drawing if any corner
// falls behind the camera (cheap near-plane clip, fine for a UI overlay).
static void DrawSlicePlaneOutline(ImVec2 item_min, ImVec2 item_size, const CameraFrame& cam, const float corners_world[4][3], ImU32 color)
{
    ImVec2 screen[4];
    for (int c = 0; c < 4; ++c)
    {
        float ndc[2];
        if (!ProjectToNDC(cam, corners_world[c], ndc))
            return;
        screen[c].x = item_min.x + (ndc[0] * 0.5f + 0.5f) * item_size.x;
        screen[c].y = item_min.y + (1.0f - (ndc[1] * 0.5f + 0.5f)) * item_size.y;
    }

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddConvexPolyFilled(screen, 4, (color & 0x00FFFFFF) | 0x20000000);
    draw_list->AddPolyline(screen, 4, color, ImDrawFlags_Closed, 2.0f);
}

int main(int, char**)
{
    WNDCLASSEXW wc = {
        sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L,
        GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr,
        L"DicomRtGuiWindowClass", nullptr
    };
    ::RegisterClassExW(&wc);
    HWND hwnd = ::CreateWindowW(
        wc.lpszClassName, L"DICOM RT Viewer", WS_OVERLAPPEDWINDOW,
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
    ImVec4 clear_color = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);

    // Milestone 1: hardcode the phantom CT path (see source/export_gui_volume.py).
    // Orthogonal MPR only: slicing below assumes an axis-aligned volume
    // (identity direction matrix) and ignores Volume::direction entirely.
    // Oblique reformatting would need a resample through that matrix - deferred.
    Volume ct_volume;
    bool ct_loaded = LoadVolume(std::string(DICOM_RT_DATA_DIR) + "/phantom/gui_export/ct", ct_volume);

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
    float window_width = 400.0f, window_level = 40.0f; // soft tissue preset, shared across planes
    bool k_dirty = true, j_dirty = true, i_dirty = true;

    // OAR mask/contour overlay (source/export_gui_volume.py --export-masks).
    // Optional: the MPR panels just show plain CT if this fails to load.
    LabelVolume label_volume;
    bool masks_loaded = false;
    bool show_overlays = true;   // "View > Structure overlays" toggle
    float overlay_alpha = 0.45f;
    std::vector<bool> label_visible;

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

        masks_loaded = LoadLabelVolume(std::string(DICOM_RT_DATA_DIR) + "/phantom/gui_export/masks", label_volume);
        if (masks_loaded && (label_volume.nx != ct_volume.nx || label_volume.ny != ct_volume.ny || label_volume.nz != ct_volume.nz))
        {
            fprintf(stderr, "Mask volume shape %dx%dx%d doesn't match CT shape %dx%dx%d; disabling overlays\n",
                label_volume.nx, label_volume.ny, label_volume.nz, ct_volume.nx, ct_volume.ny, ct_volume.nz);
            masks_loaded = false;
        }
        if (masks_loaded)
        {
            label_visible.assign(label_volume.labels.size(), true);

            DoseVolume dose_volume;
            bool real_dose_loaded = LoadDoseVolume(std::string(DICOM_RT_DATA_DIR) + "/phantom/gui_export/dose", dose_volume);
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
                ImGui::MenuItem("DVH", nullptr, &show_dvh, masks_loaded);
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }
        if (overlay_toggled_from_menu)
            k_dirty = j_dirty = i_dirty = true;

        ImGui::Begin("Axial");
        if (ct_loaded)
        {
            k_dirty |= ImGui::SliderInt("Slice", &cursor_k, 0, ct_volume.nz - 1);

            bool wl_changed = false;
            wl_changed |= ImGui::SliderFloat("Width", &window_width, 1.0f, 4000.0f, "%.0f");
            wl_changed |= ImGui::SliderFloat("Level", &window_level, -1000.0f, 1000.0f, "%.0f");

            if (ImGui::Button("Soft 400/40")) { window_width = 400; window_level = 40; wl_changed = true; }
            ImGui::SameLine();
            if (ImGui::Button("Lung 1500/-600")) { window_width = 1500; window_level = -600; wl_changed = true; }
            ImGui::SameLine();
            if (ImGui::Button("Bone 2000/500")) { window_width = 2000; window_level = 500; wl_changed = true; }
            ImGui::SameLine();
            if (ImGui::Button("Brain 80/40")) { window_width = 80; window_level = 40; wl_changed = true; }

            if (ImGui::IsWindowHovered() && io.MouseWheel != 0.0f)
            {
                cursor_k = std::clamp(cursor_k - (int)io.MouseWheel, 0, ct_volume.nz - 1);
                k_dirty = true;
            }

            // W/L is shared across all views, so a change here dirties all of them.
            if (wl_changed)
                k_dirty = j_dirty = i_dirty = true;

            if (masks_loaded && ImGui::CollapsingHeader("Structures"))
            {
                bool overlay_ui_changed = ImGui::SliderFloat("Overlay opacity", &overlay_alpha, 0.0f, 1.0f, "%.2f");
                for (size_t li = 0; li < label_volume.labels.size(); ++li)
                {
                    ImGui::PushID((int)li);
                    const uint8_t* c = kLabelColors[li % kNumLabelColors];
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

            if (k_dirty)
            {
                UploadSlice(axial_tex, ct_volume.nx, ct_volume.ny, window_width, window_level,
                    [&](int i, int j) { return ct_volume.At(i, j, cursor_k); },
                    [&](int i, int j) { return masks_loaded ? label_volume.At(i, j, cursor_k) : (uint8_t)0; },
                    show_overlays, overlay_alpha, label_visible);
                k_dirty = false;
            }

            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImVec2 size = FitImageSize(avail, ct_volume.nx, ct_volume.ny, ct_volume.spacing[0], ct_volume.spacing[1]);
            ImGui::Image((ImTextureID)(intptr_t)axial_srv, size);

            ImVec2 item_min = ImGui::GetItemRectMin();
            int click_i, click_j;
            if (PanelClicked(item_min, size, ct_volume.nx, ct_volume.ny, false, &click_i, &click_j))
            {
                cursor_i = click_i; cursor_j = click_j;
                j_dirty = i_dirty = true; // coronal/sagittal now pass through a different point
            }
            DrawCrosshairAndReadout(item_min, size, ct_volume.nx, ct_volume.ny, cursor_i, cursor_j, false,
                ct_volume, cursor_i, cursor_j, cursor_k);
        }
        else
        {
            ImGui::TextUnformatted("Axial (transversal) - failed to load CT volume, see console");
        }
        ImGui::End();

        ImGui::Begin("Coronal");
        if (ct_loaded)
        {
            j_dirty |= ImGui::SliderInt("Slice", &cursor_j, 0, ct_volume.ny - 1);

            if (ImGui::IsWindowHovered() && io.MouseWheel != 0.0f)
            {
                cursor_j = std::clamp(cursor_j - (int)io.MouseWheel, 0, ct_volume.ny - 1);
                j_dirty = true;
            }

            if (j_dirty)
            {
                // width=x, height=z; row 0 = k=0 (inferior) -- flip via UV below so superior is on top.
                UploadSlice(coronal_tex, ct_volume.nx, ct_volume.nz, window_width, window_level,
                    [&](int i, int k) { return ct_volume.At(i, cursor_j, k); },
                    [&](int i, int k) { return masks_loaded ? label_volume.At(i, cursor_j, k) : (uint8_t)0; },
                    show_overlays, overlay_alpha, label_visible);
                j_dirty = false;
            }

            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImVec2 size = FitImageSize(avail, ct_volume.nx, ct_volume.nz, ct_volume.spacing[0], ct_volume.spacing[2]);
            ImGui::Image((ImTextureID)(intptr_t)coronal_srv, size, ImVec2(0, 1), ImVec2(1, 0));

            ImVec2 item_min = ImGui::GetItemRectMin();
            int click_i, click_k;
            if (PanelClicked(item_min, size, ct_volume.nx, ct_volume.nz, true, &click_i, &click_k))
            {
                cursor_i = click_i; cursor_k = click_k;
                i_dirty = k_dirty = true;
            }
            DrawCrosshairAndReadout(item_min, size, ct_volume.nx, ct_volume.nz, cursor_i, cursor_k, true,
                ct_volume, cursor_i, cursor_j, cursor_k);
        }
        else
        {
            ImGui::TextUnformatted("Coronal (frontal) - failed to load CT volume, see console");
        }
        ImGui::End();

        ImGui::Begin("Sagittal");
        if (ct_loaded)
        {
            i_dirty |= ImGui::SliderInt("Slice", &cursor_i, 0, ct_volume.nx - 1);

            if (ImGui::IsWindowHovered() && io.MouseWheel != 0.0f)
            {
                cursor_i = std::clamp(cursor_i - (int)io.MouseWheel, 0, ct_volume.nx - 1);
                i_dirty = true;
            }

            if (i_dirty)
            {
                // width=y, height=z; same vertical flip as coronal (superior on top).
                UploadSlice(sagittal_tex, ct_volume.ny, ct_volume.nz, window_width, window_level,
                    [&](int j, int k) { return ct_volume.At(cursor_i, j, k); },
                    [&](int j, int k) { return masks_loaded ? label_volume.At(cursor_i, j, k) : (uint8_t)0; },
                    show_overlays, overlay_alpha, label_visible);
                i_dirty = false;
            }

            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImVec2 size = FitImageSize(avail, ct_volume.ny, ct_volume.nz, ct_volume.spacing[1], ct_volume.spacing[2]);
            ImGui::Image((ImTextureID)(intptr_t)sagittal_srv, size, ImVec2(0, 1), ImVec2(1, 0));

            ImVec2 item_min = ImGui::GetItemRectMin();
            int click_j, click_k;
            if (PanelClicked(item_min, size, ct_volume.ny, ct_volume.nz, true, &click_j, &click_k))
            {
                cursor_j = click_j; cursor_k = click_k;
                j_dirty = k_dirty = true;
            }
            DrawCrosshairAndReadout(item_min, size, ct_volume.ny, ct_volume.nz, cursor_j, cursor_k, true,
                ct_volume, cursor_i, cursor_j, cursor_k);
        }
        else
        {
            ImGui::TextUnformatted("Sagittal - failed to load CT volume, see console");
        }
        ImGui::End();

        ImGui::Begin("3D");
        if (ct_loaded && raycaster_ok)
        {
            if (ImGui::IsWindowHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Left))
            {
                raycaster.azimuth += io.MouseDelta.x * 0.01f;
                raycaster.elevation = std::clamp(raycaster.elevation - io.MouseDelta.y * 0.01f, -1.5f, 1.5f);
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

            RenderRaycast(g_pd3dDeviceContext, raycaster, ct_volume, window_width, window_level);

            ImGui::Checkbox("Show slice planes", &show_slice_planes);
            ImGui::SameLine();
            ImGui::TextUnformatted("(drag to orbit, right-drag to pan, wheel to zoom)");
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
                float axial_corners[4][3] = { { xmin, ymin, zk }, { xmax, ymin, zk }, { xmax, ymax, zk }, { xmin, ymax, zk } };
                DrawSlicePlaneOutline(item_min, size, cam, axial_corners, IM_COL32(230, 60, 60, 255));

                float coronal_corners[4][3] = { { xmin, yk, zmin }, { xmax, yk, zmin }, { xmax, yk, zmax }, { xmin, yk, zmax } };
                DrawSlicePlaneOutline(item_min, size, cam, coronal_corners, IM_COL32(230, 210, 60, 255));

                float sagittal_corners[4][3] = { { xk, ymin, zmin }, { xk, ymax, zmin }, { xk, ymax, zmax }, { xk, ymin, zmax } };
                DrawSlicePlaneOutline(item_min, size, cam, sagittal_corners, IM_COL32(60, 220, 120, 255));
            }
        }
        else
        {
            ImGui::TextUnformatted("3D - failed to load CT volume or initialize the raycaster, see console");
        }
        ImGui::End();

        if (show_dvh)
        {
            ImGui::Begin("DVH", &show_dvh);
            if (!dvh_curves.empty())
            {
                if (dvh_dose_is_synthetic)
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Synthetic dose (Gaussian blob) - NOT a real treatment plan.");
                else
                    ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Dose: data/phantom/gui_export/dose (real RTDOSE export)");
                if (ImPlot::BeginPlot("Dose-Volume Histogram", ImVec2(-1, -1)))
                {
                    ImPlot::SetupAxes("Dose (Gy)", "Volume (%)");
                    for (size_t li = 0; li < dvh_curves.size(); ++li)
                    {
                        const uint8_t* c = kLabelColors[li % kNumLabelColors];
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
