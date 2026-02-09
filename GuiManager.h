#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
    #include <vulkan/vulkan_raii.hpp>
#else
    import vulkan_hpp;
#endif

#include <GLFW/glfw3.h>

class GuiManager {
public:
    GuiManager(GLFWwindow* window,
               const vk::raii::Instance& instance,
               const vk::raii::PhysicalDevice& physicalDevice,
               const vk::raii::Device& device,
               const vk::raii::Queue& graphicsQueue,
               const vk::raii::RenderPass& renderPass,
               uint32_t queueFamily,
               uint32_t imageCount);
    
    ~GuiManager();

    void NewFrame();

    void UpdateUI();

    void Render(const vk::raii::CommandBuffer& commandBuffer);

private:
    vk::raii::DescriptorPool imguiPool = nullptr;
    
    void createDescriptorPool(const vk::raii::Device& device);
};