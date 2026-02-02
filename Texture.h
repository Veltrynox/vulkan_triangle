#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
    #include <vulkan/vulkan_raii.hpp>
#else
    import vulkan_hpp;
#endif

#include <string>

class Texture {
public:
    Texture(const vk::raii::Device& device, 
            const vk::raii::PhysicalDevice& physicalDevice, 
            const vk::raii::CommandPool& commandPool, 
            const vk::raii::Queue& queue, 
            const std::string& path);

    // Getters for the main application to use in Descriptor Sets
    const vk::raii::Image& getImage() const { return image; }
    const vk::raii::DeviceMemory& getMemory() const { return imageMemory; }

    const vk::raii::ImageView& getView() const { return imageView; }
    const vk::raii::Sampler& getSampler() const { return sampler; }

private:
    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory imageMemory = nullptr;
    vk::raii::ImageView imageView = nullptr;
    vk::raii::Sampler sampler = nullptr;

    // Internal Helpers
    uint32_t findMemoryType(const vk::raii::PhysicalDevice& physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties);
    
    void transitionImageLayout(const vk::raii::Device& device, const vk::raii::CommandPool& commandPool, 
                               const vk::raii::Queue& queue, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
    
    void copyBufferToImage(const vk::raii::Device& device, const vk::raii::CommandPool& commandPool, 
                           const vk::raii::Queue& queue, vk::Buffer buffer, uint32_t width, uint32_t height);
};