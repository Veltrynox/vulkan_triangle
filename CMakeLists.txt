cmake_minimum_required(VERSION 3.25)

project(Triangle)

# 1. Setup C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==============================================================================
# MANUAL VULKAN SETUP
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
# C++20 MODULE SETUP (From your guide)
# ==============================================================================
option(ENABLE_CPP20_MODULE "Enable Vulkan C++20 Modules" OFF)

if(ENABLE_CPP20_MODULE)
    # We don't need find_package(Vulkan) here because we did it manually above
    
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
add_executable(Triangle main.cpp)

if(ENABLE_CPP20_MODULE)
    target_link_libraries(Triangle PRIVATE Vulkan::cppm glfw)
else()
    target_link_libraries(Triangle PRIVATE Vulkan::Vulkan glfw)
endif()

# macOS RPATH fix so it finds the dylib at runtime
set_target_properties(Triangle PROPERTIES
    BUILD_RPATH "${VULKAN_SDK_ROOT}/lib"
    INSTALL_RPATH "${VULKAN_SDK_ROOT}/lib"
)