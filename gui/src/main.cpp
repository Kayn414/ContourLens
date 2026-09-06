// Minimal Dear ImGui + Win32 + DirectX 11 bootstrap.
// Modeled closely on imgui/examples/example_win32_directx11/main.cpp (docking branch).

#define NOMINMAX // windows.h's min/max macros shadow std::min/std::max otherwise
#include "imgui.h"
#include "imgui_internal.h" // DockBuilder* - forced 2x2 layout, built once on first frame
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include "volume.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <d3d11.h>
#include <tchar.h>

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

// Windows HU to a grayscale byte: (hu - (wl - ww/2)) / ww, clamped to [0,1].
// `get_voxel(col, row)` picks the plane; callers below index the volume
// differently for axial/coronal/sagittal but share this upload path.
template <typename GetVoxel>
static void UploadSlice(ID3D11Texture2D* tex, int width, int height, float window_width, float window_level, GetVoxel get_voxel)
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
            dst[col * 4 + 0] = gray;
            dst[col * 4 + 1] = gray;
            dst[col * 4 + 2] = gray;
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

    int axial_slice = 0;    // k (z index)
    int coronal_slice = 0;  // j (y index)
    int sagittal_slice = 0; // i (x index)
    float window_width = 400.0f, window_level = 40.0f; // soft tissue preset, shared across planes
    bool axial_dirty = true, coronal_dirty = true, sagittal_dirty = true;

    if (ct_loaded)
    {
        axial_slice = ct_volume.nz / 2;
        coronal_slice = ct_volume.ny / 2;
        sagittal_slice = ct_volume.nx / 2;
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

        if (ImGui::BeginMainMenuBar())
        {
            if (ImGui::BeginMenu("File"))
            {
                if (ImGui::MenuItem("Exit"))
                    done = true;
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }

        ImGui::Begin("Axial");
        if (ct_loaded)
        {
            bool slice_changed = axial_dirty;
            slice_changed |= ImGui::SliderInt("Slice", &axial_slice, 0, ct_volume.nz - 1);

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
                axial_slice = std::clamp(axial_slice - (int)io.MouseWheel, 0, ct_volume.nz - 1);
                slice_changed = true;
            }

            // W/L is shared across all three planes, so a change here dirties them too.
            if (wl_changed)
                coronal_dirty = sagittal_dirty = true;

            if (slice_changed || wl_changed)
            {
                UploadSlice(axial_tex, ct_volume.nx, ct_volume.ny, window_width, window_level,
                    [&](int i, int j) { return ct_volume.At(i, j, axial_slice); });
                axial_dirty = false;
            }

            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImVec2 size = FitImageSize(avail, ct_volume.nx, ct_volume.ny, ct_volume.spacing[0], ct_volume.spacing[1]);
            ImGui::Image((ImTextureID)(intptr_t)axial_srv, size);
        }
        else
        {
            ImGui::TextUnformatted("Axial (transversal) - failed to load CT volume, see console");
        }
        ImGui::End();

        ImGui::Begin("Coronal");
        if (ct_loaded)
        {
            bool changed = coronal_dirty;
            changed |= ImGui::SliderInt("Slice", &coronal_slice, 0, ct_volume.ny - 1);

            if (ImGui::IsWindowHovered() && io.MouseWheel != 0.0f)
            {
                coronal_slice = std::clamp(coronal_slice - (int)io.MouseWheel, 0, ct_volume.ny - 1);
                changed = true;
            }

            if (changed)
            {
                // width=x, height=z; row 0 = k=0 (inferior) -- flip via UV below so superior is on top.
                UploadSlice(coronal_tex, ct_volume.nx, ct_volume.nz, window_width, window_level,
                    [&](int i, int k) { return ct_volume.At(i, coronal_slice, k); });
                coronal_dirty = false;
            }

            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImVec2 size = FitImageSize(avail, ct_volume.nx, ct_volume.nz, ct_volume.spacing[0], ct_volume.spacing[2]);
            ImGui::Image((ImTextureID)(intptr_t)coronal_srv, size, ImVec2(0, 1), ImVec2(1, 0));
        }
        else
        {
            ImGui::TextUnformatted("Coronal (frontal) - failed to load CT volume, see console");
        }
        ImGui::End();

        ImGui::Begin("Sagittal");
        if (ct_loaded)
        {
            bool changed = sagittal_dirty;
            changed |= ImGui::SliderInt("Slice", &sagittal_slice, 0, ct_volume.nx - 1);

            if (ImGui::IsWindowHovered() && io.MouseWheel != 0.0f)
            {
                sagittal_slice = std::clamp(sagittal_slice - (int)io.MouseWheel, 0, ct_volume.nx - 1);
                changed = true;
            }

            if (changed)
            {
                // width=y, height=z; same vertical flip as coronal (superior on top).
                UploadSlice(sagittal_tex, ct_volume.ny, ct_volume.nz, window_width, window_level,
                    [&](int j, int k) { return ct_volume.At(sagittal_slice, j, k); });
                sagittal_dirty = false;
            }

            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImVec2 size = FitImageSize(avail, ct_volume.ny, ct_volume.nz, ct_volume.spacing[1], ct_volume.spacing[2]);
            ImGui::Image((ImTextureID)(intptr_t)sagittal_srv, size, ImVec2(0, 1), ImVec2(1, 0));
        }
        else
        {
            ImGui::TextUnformatted("Sagittal - failed to load CT volume, see console");
        }
        ImGui::End();

        ImGui::Begin("3D");
        ImGui::TextUnformatted("3D render - placeholder");
        ImGui::End();

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

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
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
