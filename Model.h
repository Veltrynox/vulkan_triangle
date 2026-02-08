#pragma once

#define GLFW_INCLUDE_VULKAN
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
    #include <vulkan/vulkan_raii.hpp>
#else
    import vulkan_hpp;
#endif

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <array>

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription() {
        return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
    }

    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions{};
        attributeDescriptions[0] = {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)};
        attributeDescriptions[1] = {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)};
        attributeDescriptions[2] = {2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord)};
        return attributeDescriptions;
    }
};

class Model {
public:
    Model(const vk::raii::Device& device, const vk::raii::PhysicalDevice& physicalDevice, 
          const vk::raii::CommandPool& commandPool, const vk::raii::Queue& queue, 
          const std::string& path);

    const vk::raii::Buffer& getVertexBuffer() const { return vertexBuffer; }
    const vk::raii::Buffer& getIndexBuffer() const { return indexBuffer; }
    uint32_t getIndexCount() const { return static_cast<uint32_t>(indices.size()); }

private:
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    vk::raii::Buffer vertexBuffer = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory = nullptr;
    vk::raii::Buffer indexBuffer = nullptr;
    vk::raii::DeviceMemory indexBufferMemory = nullptr;

    void loadModel(const std::string& path);
    void createVertexBuffers(const vk::raii::Device& device, const vk::raii::PhysicalDevice& physicalDevice, const vk::raii::CommandPool& commandPool, const vk::raii::Queue& queue);
    void createIndexBuffers(const vk::raii::Device& device, const vk::raii::PhysicalDevice& physicalDevice, const vk::raii::CommandPool& commandPool, const vk::raii::Queue& queue);

    void createBuffer(const vk::raii::Device& device, const vk::raii::PhysicalDevice& physicalDevice, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory);
    void copyBuffer(const vk::raii::Device& device, const vk::raii::CommandPool& commandPool, const vk::raii::Queue& queue, vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size);
    uint32_t findMemoryType(const vk::raii::PhysicalDevice& physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties);
};