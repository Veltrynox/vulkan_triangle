cmake_minimum_required(VERSION 3.25)

project(Triangle)

# ==============================================================================
# C++ SETUP
# ==============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==============================================================================
# VULKAN SETUP
# ==============================================================================
set(VULKAN_SDK_ROOT "/Volumes/Work/Gamedev/systems/Vulkan/macOS")
set(Vulkan_INCLUDE_DIR "${VULKAN_SDK_ROOT}/include")
set(Vulkan_LIBRARY     "${VULKAN_SDK_ROOT}/lib/libvulkan.dylib")

if(NOT EXISTS "${Vulkan_LIBRARY}")
    set(Vulkan_LIBRARY "${VULKAN_SDK_ROOT}/lib/libvulkan.1.dylib")
endif()

add_library(Vulkan::Vulkan SHARED IMPORTED)
set_target_properties(Vulkan::Vulkan PROPERTIES
    IMPORTED_LOCATION "${Vulkan_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIR}"
)

# ==============================================================================
# EXTERNAL DEPENDENCIES (FetchContent)
# ==============================================================================
include(FetchContent)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.3.8
)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        master
)

FetchContent_MakeAvailable(glfw imgui)

# ==============================================================================
# IMGUI BACKEND LIBRARY
# ==============================================================================
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
    GuiManager.cpp
    Splat.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
)

target_include_directories(Triangle PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${IMGUI_DIR}
)

if(ENABLE_CPP20_MODULE)
    target_link_libraries(Triangle PRIVATE 
        Vulkan::cppm 
        glfw 
        imgui_backend
    )
else()
    target_link_libraries(Triangle PRIVATE 
        Vulkan::Vulkan 
        glfw 
        imgui_backend
    )
endif()

# ==============================================================================
# macOS RPATH SETUP
# ==============================================================================
set_target_properties(Triangle PROPERTIES
    BUILD_RPATH "${VULKAN_SDK_ROOT}/lib"
    INSTALL_RPATH "${VULKAN_SDK_ROOT}/lib"
)