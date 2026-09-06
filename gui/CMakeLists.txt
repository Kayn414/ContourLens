cmake_minimum_required(VERSION 3.21)
project(dicom_rt_gui LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.92.9b-docking
)
FetchContent_MakeAvailable(imgui)

set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(json)

# ImPlot ships no CMakeLists.txt (same as imgui) -- FetchContent_MakeAvailable
# just populates the source dir without add_subdirectory-ing it; compile its
# .cpp files directly like the imgui ones below.
FetchContent_Declare(
  implot
  GIT_REPOSITORY https://github.com/epezent/implot.git
  GIT_TAG        v1.0
)
FetchContent_MakeAvailable(implot)

add_executable(dicom_rt_gui
  src/main.cpp
  src/volume.cpp
  src/raycaster.cpp
  src/dvh.cpp
  src/inference.cpp
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_dx11.cpp
  ${implot_SOURCE_DIR}/implot.cpp
  ${implot_SOURCE_DIR}/implot_items.cpp
)

target_include_directories(dicom_rt_gui PRIVATE
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
  ${implot_SOURCE_DIR}
)

target_link_libraries(dicom_rt_gui PRIVATE d3d11.lib dxgi.lib d3dcompiler.lib nlohmann_json::nlohmann_json)

# Volume exports (source/export_gui_volume.py) live under <repo>/data, and
# "Run Inference" (gui/src/inference.cpp) shells out to `uv run python -m
# source...` from the repo root -- resolve both relative to the repo
# regardless of the exe's own working directory.
target_compile_definitions(dicom_rt_gui PRIVATE
  DICOM_RT_DATA_DIR="${CMAKE_SOURCE_DIR}/../data"
  DICOM_RT_REPO_DIR="${CMAKE_SOURCE_DIR}/.."
)

set_target_properties(dicom_rt_gui PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
