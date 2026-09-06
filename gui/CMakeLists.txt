cmake_minimum_required(VERSION 3.21)
project(dicom_rt_gui LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.92.9b-docking
)
FetchContent_MakeAvailable(imgui)

add_executable(dicom_rt_gui
  src/main.cpp
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_dx11.cpp
)

target_include_directories(dicom_rt_gui PRIVATE
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
)

target_link_libraries(dicom_rt_gui PRIVATE d3d11.lib dxgi.lib d3dcompiler.lib)

set_target_properties(dicom_rt_gui PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
