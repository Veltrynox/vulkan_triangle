cmake_minimum_required(VERSION 3.25)

project(Triangle)

# Setup C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==============================================================================
# VULKAN SETUP
# ==============================================================================

set(VULKAN_SDK_ROOT "/Volumes/Work/Gamedev/systems/Vulkan/macOS")

# Manually define where the headers and libs are
set(Vulkan_INCLUDE_DIR "${VULKAN_SDK_ROOT}/include")
set(Vulkan_LIBRARY     "${VULKAN_SDK_ROOT}/lib/libvulkan.dylib")

# Check if they actually exist (Sanity check)
if(NOT EXISTS "${Vulkan_INCLUDE_DIR}")
    message(FATAL_ERROR "Could not find Vulkan headers at: ${Vulkan_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${Vulkan_LIBRARY}")
    # Fallback: sometimes it's .1.dylib on Mac
    set(Vulkan_LIBRARY "${VULKAN_SDK_ROOT}/lib/libvulkan.1.dylib")
    if(NOT EXISTS "${Vulkan_LIBRARY}")
        message(FATAL_ERROR "Could not find Vulkan library at: ${Vulkan_SDK_ROOT}/lib/libvulkan.dylib")
    endif()
endif()

# Create the "Vulkan::Vulkan" target manually so we can link against it
add_library(Vulkan::Vulkan SHARED IMPORTED)
set_target_properties(Vulkan::Vulkan PROPERTIES
    IMPORTED_LOCATION "${Vulkan_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIR}"
)

message(STATUS "Vulkan Found Manually: ${Vulkan_LIBRARY}")

# ==============================================================================
# OPENUSD SETUP 
# ==============================================================================
set(USD_ROOT "/Volumes/Work/Gamedev/systems/OpenUSD_Install")
list(APPEND CMAKE_PREFIX_PATH "${USD_ROOT}")
find_package(pxr REQUIRED)

set(USD_MANUAL_LIBS
    "${USD_ROOT}/lib/libusd_usdGeom.dylib"
    "${USD_ROOT}/lib/libusd_usd.dylib"
    "${USD_ROOT}/lib/libusd_pcp.dylib"
    "${USD_ROOT}/lib/libusd_sdf.dylib"
    "${USD_ROOT}/lib/libusd_vt.dylib"
    "${USD_ROOT}/lib/libusd_gf.dylib"
    "${USD_ROOT}/lib/libusd_tf.dylib"
    "${USD_ROOT}/lib/libusd_plug.dylib"
    "${USD_ROOT}/lib/libusd_work.dylib"
    "${USD_ROOT}/lib/libusd_arch.dylib"
    "${USD_ROOT}/lib/libusd_ar.dylib"
    "${USD_ROOT}/lib/libtbb.dylib"
)

message(STATUS "OpenUSD Found at: ${pxr_DIR}")

# ==============================================================================
# GLFW SETUP
# ==============================================================================
include(FetchContent)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.3.8
)
FetchContent_MakeAvailable(glfw)

# ==============================================================================
# IMGUI SETUP
# ==============================================================================
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
)
FetchContent_MakeAvailable(imgui)
set(IMGUI_DIR ${imgui_SOURCE_DIR})
add_library(imgui_backend STATIC
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui_backend PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(imgui_backend PUBLIC glfw Vulkan::Vulkan)

# ==============================================================================
# C++20 MODULE SETUP
# ==============================================================================
option(ENABLE_CPP20_MODULE "Enable Vulkan C++20 Modules" OFF)

if(ENABLE_CPP20_MODULE)
    add_library(VulkanCppModule)
    add_library(Vulkan::cppm ALIAS VulkanCppModule)

    target_compile_definitions(VulkanCppModule PUBLIC
            VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
            VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    )

    target_include_directories(VulkanCppModule PRIVATE "${Vulkan_INCLUDE_DIR}")
    target_link_libraries(VulkanCppModule PUBLIC Vulkan::Vulkan)
    set_target_properties(VulkanCppModule PROPERTIES CXX_STANDARD 20)

    target_sources(VulkanCppModule
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES
            BASE_DIRS "${Vulkan_INCLUDE_DIR}"
            FILES "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm"
    )
endif()

# ==============================================================================
# MAIN EXECUTABLE
# ==============================================================================
add_executable(Triangle 
    main.cpp 
    Texture.cpp
    Model.cpp
    UsdBridge.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    GuiManager.cpp
)

target_include_directories(Triangle PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PXR_INCLUDE_DIRS}
    ${IMGUI_DIR}
)

target_compile_definitions(Triangle PRIVATE ${PXR_CXX_DEFINITIONS})

if(ENABLE_CPP20_MODULE)
    target_link_libraries(Triangle PRIVATE 
        Vulkan::cppm 
        glfw 
        imgui_backend 
        ${USD_MANUAL_LIBS}
    )
else()
    target_link_libraries(Triangle PRIVATE 
        Vulkan::Vulkan 
        glfw 
        imgui_backend 
        ${USD_MANUAL_LIBS}
    )
endif()

# macOS RPATH fix
set_target_properties(Triangle PROPERTIES
    BUILD_RPATH "${VULKAN_SDK_ROOT}/lib;${USD_ROOT}/lib"
    INSTALL_RPATH "${VULKAN_SDK_ROOT}/lib;${USD_ROOT}/lib"
)