#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>

// Texture support
#include "Texture.h"
#include <memory>

// Vulkan RAII and Standard Headers
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
    #include <vulkan/vulkan_raii.hpp>
#else
    import vulkan_hpp;
#endif

#include <iostream>
#include <stdexcept>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <fstream>
#include <limits>
#include <array>
#include <cstddef>
#include <glm/glm.hpp>
#include <vector>
#include <array>

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    // --- 1. CONFIGURATION ---
    const uint32_t WIDTH = 800;
    const uint32_t HEIGHT = 600;

    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        "VK_KHR_portability_subset"
    };

    
    // --- 2. STRUCTS ---
    struct SwapChainSupportDetails {
        vk::SurfaceCapabilitiesKHR capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR> presentModes;
    };
    
    struct Vertex {
        glm::vec2 pos;
        glm::vec3 color;
        glm::vec2 texCoord; // Add this

        static vk::VertexInputBindingDescription getBindingDescription() {
            return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
        }

        static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
            std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions{};

            attributeDescriptions[0] = {0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, pos)};
            attributeDescriptions[1] = {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)};
            attributeDescriptions[2] = {2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord)};

            return attributeDescriptions;
        }
    };

    const std::vector<Vertex> vertices = {
        {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
        {{ 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        {{ 0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
        {{-0.5f,  0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}
    };

    const std::vector<uint16_t> indices = {
        0, 3, 2,
        2, 1, 0
    };

    struct UniformBufferObject {
        alignas(16) glm::mat4 model;
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
    };

    // --- 3. CLASS MEMBERS ---
    GLFWwindow* window;
    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;
    
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;
    vk::raii::Queue graphicsQueue = nullptr;
    vk::raii::Queue presentQueue = nullptr;

    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    vk::raii::RenderPass renderPass = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;
    std::vector<vk::raii::Framebuffer> swapChainFramebuffers;

    vk::raii::CommandPool commandPool = nullptr;
    vk::raii::CommandBuffers commandBuffers = nullptr;

    vk::raii::Semaphore imageAvailableSemaphore = nullptr;
    vk::raii::Semaphore renderFinishedSemaphore = nullptr;
    vk::raii::Fence inFlightFence = nullptr;

    vk::raii::Buffer vertexBuffer = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory = nullptr;
    vk::raii::Buffer indexBuffer = nullptr;
    vk::raii::DeviceMemory indexBufferMemory = nullptr;

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    vk::raii::DescriptorSets descriptorSets = nullptr;

    vk::raii::Buffer uniformBuffer = nullptr;
    vk::raii::DeviceMemory uniformBufferMemory = nullptr;
    void* uniformBufferMapped = nullptr;

    std::unique_ptr<Texture> texture;
    
    uint32_t graphicsFamilyIndex = 0;
    uint32_t presentFamilyIndex = 0;

    // --- 4. INITIALIZATION FUNCTIONS ---

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        texture = std::make_unique<Texture>(device, physicalDevice, commandPool, graphicsQueue, "textures/texture.jpg");
        createCommandBuffer();
        createSyncObjects();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffer();
        createDescriptorPool();
        createDescriptorSets();
    }

    void createInstance() {
        vk::ApplicationInfo appInfo("Hello Triangle", VK_MAKE_VERSION(1, 0, 0), "No Engine", VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_3);

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

        vk::InstanceCreateInfo createInfo(
            vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR, &appInfo, 0, nullptr,
            static_cast<uint32_t>(extensions.size()), extensions.data()
        );

        instance = vk::raii::Instance(context, createInfo);
    }

    void createSurface() {
        VkSurfaceKHR rawSurface;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &rawSurface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = vk::raii::SurfaceKHR(instance, rawSurface);
    }

    void pickPhysicalDevice() {
        vk::raii::PhysicalDevices devices(instance);
        for (const auto& dev : devices) {
            if (checkDeviceExtensionSupport(dev)) {
                SwapChainSupportDetails swapChainSupport = querySwapChainSupport(dev);
                if (!swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty()) {
                    physicalDevice = dev;
                    break;
                }
            }
        }
        if (*physicalDevice == nullptr) throw std::runtime_error("No suitable GPU!");
        
        auto props = physicalDevice.getProperties();
        std::cout << "Selected GPU: " << props.deviceName << std::endl;
    }

    void createLogicalDevice() {
        auto queueFamilies = physicalDevice.getQueueFamilyProperties();
        bool graphicsFound = false, presentFound = false;

        for (uint32_t i = 0; i < queueFamilies.size(); i++) {
            if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                graphicsFamilyIndex = i; graphicsFound = true;
            }
            if (physicalDevice.getSurfaceSupportKHR(i, *surface)) {
                presentFamilyIndex = i; presentFound = true;
            }
            if (graphicsFound && presentFound) break;
        }

        std::vector<vk::DeviceQueueCreateInfo> queueInfos;
        std::set<uint32_t> uniqueFamilies = {graphicsFamilyIndex, presentFamilyIndex};
        float priority = 1.0f;
        for (uint32_t family : uniqueFamilies) {
            queueInfos.push_back({{}, family, 1, &priority});
        }

        vk::PhysicalDeviceFeatures features{};
        vk::DeviceCreateInfo createInfo({}, queueInfos, {}, deviceExtensions, &features);

        device = vk::raii::Device(physicalDevice, createInfo);
        graphicsQueue = vk::raii::Queue(device, graphicsFamilyIndex, 0);
        presentQueue = vk::raii::Queue(device, presentFamilyIndex, 0);
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
        vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        vk::SwapchainCreateInfoKHR createInfo({}, *surface, imageCount, surfaceFormat.format, surfaceFormat.colorSpace, extent, 1, vk::ImageUsageFlagBits::eColorAttachment);

        uint32_t queueFamilyIndices[] = {graphicsFamilyIndex, presentFamilyIndex};
        if (graphicsFamilyIndex != presentFamilyIndex) {
            createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = vk::SharingMode::eExclusive;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        swapChain = vk::raii::SwapchainKHR(device, createInfo);
        swapChainImages = swapChain.getImages();
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
        
        std::cout << "Swapchain created (" << swapChainExtent.width << "x" << swapChainExtent.height << ")" << std::endl;
    }

    void createImageViews() {
        swapChainImageViews.clear();
        for (const auto& image : swapChainImages) {
            vk::ImageViewCreateInfo createInfo({}, image, vk::ImageViewType::e2D, swapChainImageFormat, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
            swapChainImageViews.emplace_back(device, createInfo);
        }
    }

    void createRenderPass() {
        vk::AttachmentDescription colorAttachment({}, swapChainImageFormat, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR);
        vk::AttachmentReference colorRef(0, vk::ImageLayout::eColorAttachmentOptimal);
        vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &colorRef);
        vk::SubpassDependency dependency(VK_SUBPASS_EXTERNAL, 0, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, vk::AccessFlagBits::eColorAttachmentWrite);
        
        vk::RenderPassCreateInfo renderPassInfo({}, 1, &colorAttachment, 1, &subpass, 1, &dependency);
        renderPass = vk::raii::RenderPass(device, renderPassInfo);
    }

    void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;

        vk::DescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};
        vk::DescriptorSetLayoutCreateInfo layoutInfo({}, static_cast<uint32_t>(bindings.size()), bindings.data());
        
        descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
    }

    void createDescriptorPool() {
        std::array<vk::DescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
        poolSizes[0].descriptorCount = 1;
        poolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
        poolSizes[1].descriptorCount = 1;

        vk::DescriptorPoolCreateInfo poolInfo({}, 1, static_cast<uint32_t>(poolSizes.size()), poolSizes.data());
        descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
    }

    void createDescriptorSets() {
        vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool, *descriptorSetLayout);
        descriptorSets = vk::raii::DescriptorSets(device, allocInfo);

        vk::DescriptorBufferInfo bufferInfo(*uniformBuffer, 0, sizeof(UniformBufferObject));
        vk::DescriptorImageInfo imageInfo(*texture->getSampler(), *texture->getView(), vk::ImageLayout::eShaderReadOnlyOptimal);
        std::array<vk::WriteDescriptorSet, 2> descriptorWrites{};

        descriptorWrites[0].dstSet = *descriptorSets[0];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        descriptorWrites[1].dstSet = *descriptorSets[0];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;

        device.updateDescriptorSets(descriptorWrites, nullptr);
    }

    void createGraphicsPipeline() {
        auto vertCode = readFile("shaders/vert.spv");
        auto fragCode = readFile("shaders/frag.spv");

        vk::raii::ShaderModule vertModule = createShaderModule(vertCode);
        vk::raii::ShaderModule fragModule = createShaderModule(fragCode);

        vk::PipelineShaderStageCreateInfo shaderStages[] = {
            {{}, vk::ShaderStageFlagBits::eVertex, *vertModule, "main"},
            {{}, vk::ShaderStageFlagBits::eFragment, *fragModule, "main"}
        };

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList, VK_FALSE);
        vk::Viewport viewport(0.0f, 0.0f, (float)swapChainExtent.width, (float)swapChainExtent.height, 0.0f, 1.0f);
        vk::Rect2D scissor({0, 0}, swapChainExtent);
        vk::PipelineViewportStateCreateInfo viewportState({}, 1, &viewport, 1, &scissor);
        vk::PipelineRasterizationStateCreateInfo rasterizer({}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise, VK_FALSE, 0.0f, 0.0f, 0.0f, 1.0f);
        vk::PipelineMultisampleStateCreateInfo multisampling({}, vk::SampleCountFlagBits::e1, VK_FALSE);
        vk::PipelineColorBlendAttachmentState colorBlendAttachment(VK_FALSE, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
        vk::PipelineColorBlendStateCreateInfo colorBlending({}, VK_FALSE, vk::LogicOp::eCopy, 1, &colorBlendAttachment);
        
        vk::PipelineLayoutCreateInfo layoutInfo({}, 1, &*descriptorSetLayout, 0, nullptr);
        pipelineLayout = vk::raii::PipelineLayout(device, layoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo({}, 2, shaderStages, &vertexInputInfo, &inputAssembly, nullptr, &viewportState, &rasterizer, &multisampling, nullptr, &colorBlending, nullptr, *pipelineLayout, *renderPass, 0);
        graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    }

    void createFramebuffers() {
        swapChainFramebuffers.clear();
        for (const auto& view : swapChainImageViews) {
            vk::ImageView attachments[] = {*view};
            vk::FramebufferCreateInfo fbInfo({}, *renderPass, 1, attachments, swapChainExtent.width, swapChainExtent.height, 1);
            swapChainFramebuffers.emplace_back(device, fbInfo);
        }
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo poolInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, graphicsFamilyIndex);
        commandPool = vk::raii::CommandPool(device, poolInfo);
    }

    void createCommandBuffer() {
        vk::CommandBufferAllocateInfo allocInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 1);
        commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
    }

    void createSyncObjects() {
        vk::SemaphoreCreateInfo semaphoreInfo{};
        vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);
        imageAvailableSemaphore = vk::raii::Semaphore(device, semaphoreInfo);
        renderFinishedSemaphore = vk::raii::Semaphore(device, semaphoreInfo);
        inFlightFence = vk::raii::Fence(device, fenceInfo);
    }

    // --- 5. RUNTIME LOOP ---

    void drawFrame() {
        (void)device.waitForFences(*inFlightFence, VK_TRUE, UINT64_MAX);
        device.resetFences(*inFlightFence);

        auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *imageAvailableSemaphore);
        
        const auto& commandBuffer = commandBuffers[0];
        
        updateUniformBuffer();
        
        commandBuffer.reset();
        commandBuffer.begin(vk::CommandBufferBeginInfo{});

        vk::ClearValue clearColor(vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}));
        vk::RenderPassBeginInfo renderPassInfo(*renderPass, *swapChainFramebuffers[imageIndex], vk::Rect2D({0, 0}, swapChainExtent), 1, &clearColor);

        commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);

        vk::Buffer vertexBuffers[] = {*vertexBuffer};
        vk::DeviceSize offsets[] = {0};
        
        commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
        commandBuffer.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint16);

        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, *descriptorSets[0], nullptr);
        commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
        commandBuffer.endRenderPass();
        commandBuffer.end();

        vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
        vk::SubmitInfo submitInfo(*imageAvailableSemaphore, waitStages, *commandBuffer, *renderFinishedSemaphore);
        graphicsQueue.submit(submitInfo, *inFlightFence);

        vk::PresentInfoKHR presentInfo(*renderFinishedSemaphore, *swapChain, imageIndex);
        (void)presentQueue.presentKHR(presentInfo);
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }
        device.waitIdle();
    }

    void cleanup() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    // --- 6. HELPERS ---

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("failed to open file: " + filename);
        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();
        return buffer;
    }

    vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) {
        return vk::raii::ShaderModule(device, vk::ShaderModuleCreateInfo({}, code.size(), reinterpret_cast<const uint32_t*>(code.data())));
    }

    bool checkDeviceExtensionSupport(const vk::raii::PhysicalDevice& dev) {
        auto availableExtensions = dev.enumerateDeviceExtensionProperties();
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
        for (const auto& ext : availableExtensions) requiredExtensions.erase(ext.extensionName);
        return requiredExtensions.empty();
    }

    SwapChainSupportDetails querySwapChainSupport(const vk::raii::PhysicalDevice& dev) {
        return { dev.getSurfaceCapabilitiesKHR(*surface), dev.getSurfaceFormatsKHR(*surface), dev.getSurfacePresentModesKHR(*surface) };
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        for (const auto& format : availableFormats) {
            if (format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) return format;
        }
        return availableFormats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availableModes) {
        for (const auto& mode : availableModes) if (mode == vk::PresentModeKHR::eMailbox) return mode;
        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) return capabilities.currentExtent;
        int w, h; glfwGetFramebufferSize(window, &w, &h);
        vk::Extent2D actual = {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
        actual.width = std::clamp(actual.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actual.height = std::clamp(actual.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        return actual;
    }

    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, 
                    vk::MemoryPropertyFlags properties, 
                    vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory) {
        
        vk::BufferCreateInfo bufferInfo({}, size, usage, vk::SharingMode::eExclusive);
        buffer = vk::raii::Buffer(device, bufferInfo);

        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();

        vk::MemoryAllocateInfo allocInfo(memRequirements.size, 
                                        findMemoryType(memRequirements.memoryTypeBits, properties));

        bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
        buffer.bindMemory(*bufferMemory, 0);
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && 
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createVertexBuffer() {
        vk::DeviceSize bufferSize = sizeof(Vertex) * vertices.size();

        // Staging Buffer
        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingMemory = nullptr;
        createBuffer(bufferSize, 
                    vk::BufferUsageFlagBits::eTransferSrc, 
                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, 
                    stagingBuffer, stagingMemory);

        // Copy
        void* data = stagingMemory.mapMemory(0, bufferSize);
        memcpy(data, vertices.data(), (size_t)bufferSize);
        stagingMemory.unmapMemory();

        // Vertex Buffer
        createBuffer(bufferSize, 
                    vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, 
                    vk::MemoryPropertyFlagBits::eDeviceLocal, 
                    vertexBuffer, vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
    }

    void createIndexBuffer() {
        vk::DeviceSize bufferSize = sizeof(uint16_t) * indices.size();

        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingMemory = nullptr;
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, 
                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, 
                    stagingBuffer, stagingMemory);

        void* data = stagingMemory.mapMemory(0, bufferSize);
        memcpy(data, indices.data(), (size_t)bufferSize);
        stagingMemory.unmapMemory();

        createBuffer(bufferSize, 
                    vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, 
                    vk::MemoryPropertyFlagBits::eDeviceLocal, 
                    indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);
    }

    void createUniformBuffer() {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

        createBuffer(bufferSize, 
            vk::BufferUsageFlagBits::eUniformBuffer, 
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, 
            uniformBuffer, uniformBufferMemory);

        uniformBufferMapped = uniformBufferMemory.mapMemory(0, bufferSize);
    }

    void updateUniformBuffer() {
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        
        ubo.view = glm::lookAt(glm::vec3(0.0f, 0.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        
        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);
        
        ubo.proj[1][1] *= -1;

        memcpy(uniformBufferMapped, &ubo, sizeof(ubo));
    }

    void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size) {
        vk::CommandBufferAllocateInfo allocInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 1);
        vk::raii::CommandBuffer copyCmd = std::move(device.allocateCommandBuffers(allocInfo).front());

        copyCmd.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

        vk::BufferCopy copyRegion(0, 0, size);
        copyCmd.copyBuffer(*srcBuffer, *dstBuffer, copyRegion);

        copyCmd.end();
        
        vk::SubmitInfo submitInfo({}, {}, *copyCmd, {});
        graphicsQueue.submit(submitInfo, nullptr);
        graphicsQueue.waitIdle();
    }
};

int main() {
    HelloTriangleApplication app;
    try { app.run(); } 
    catch (const std::exception& e) { std::cerr << e.what() << std::endl; return EXIT_FAILURE; }
    return EXIT_SUCCESS;
}