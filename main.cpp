#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>

#include "Camera.h"
#include "GuiManager.h"
#include "DataProvider.h"
#include "Splat.h"

// Vulkan RAII and Standard Headers
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
    #include <vulkan/vulkan_raii.hpp>
#else
    import vulkan_hpp;
#endif

#include <memory>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <fstream>
#include <limits>
#include <array>

// --- GLOBALS ---
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = 400.0f, lastY = 300.0f;
bool firstMouse = true;
float deltaTime = 0.0f, lastFrame = 0.0f;

// --- INPUT CALLBACK ---
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }

    float xoffset = xpos - lastX;
    float yoffset = ypos - lastY;
    lastX = xpos; lastY = ypos;

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
        camera.ProcessMouseMovement(xoffset, yoffset);
    }
}

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    // --- CONSTANTS ---
    const uint32_t WIDTH = 800;
    const uint32_t HEIGHT = 600;
    const std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME, "VK_KHR_portability_subset" };

    // --- STRUCTS ---
    struct SwapChainSupportDetails {
        vk::SurfaceCapabilitiesKHR capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR> presentModes;
    };

    struct UniformBufferObject {
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
        alignas(16) glm::vec4 cameraPos;
        alignas(16) glm::vec4 viewport_focal;
    };

    struct SortProxy {
        uint32_t index;
        float depth;
    };

    // --- VULKAN HANDLES ---
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

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    vk::raii::DescriptorSets descriptorSets = nullptr;

    // Buffers
    vk::raii::Buffer uniformBuffer = nullptr;
    vk::raii::DeviceMemory uniformBufferMemory = nullptr;
    void* uniformBufferMapped = nullptr;

    vk::raii::Buffer splatBuffer = nullptr;
    vk::raii::DeviceMemory splatBufferMemory = nullptr;

    vk::raii::Buffer sortBuffer = nullptr;
    vk::raii::DeviceMemory sortBufferMemory = nullptr;
    std::vector<uint32_t> hostSortIndices;
    std::vector<SortProxy> sortProxies;

    // Resources
    vk::raii::Image depthImage = nullptr;
    vk::raii::DeviceMemory depthImageMemory = nullptr;
    vk::raii::ImageView depthImageView = nullptr;
    vk::Format depthFormat;

    vk::raii::Image colorImage = nullptr;
    vk::raii::DeviceMemory colorImageMemory = nullptr;
    vk::raii::ImageView colorImageView = nullptr;
    vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e4;

    std::unique_ptr<GuiManager> gui;
    std::unique_ptr<IDataProvider> dataProvider;
    
    // SH
    vk::raii::Buffer shBuffer = nullptr;
    vk::raii::DeviceMemory shBufferMemory = nullptr;

    uint32_t graphicsFamilyIndex = 0;
    uint32_t presentFamilyIndex = 0;

    // --- HELPER DECLARATIONS ---
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

    // --- INITIALIZATION ---
    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Splatting", nullptr, nullptr);
        glfwSetCursorPosCallback(window, mouse_callback);
    }

    void init_imgui() {
        gui = std::make_unique<GuiManager>(window, instance, physicalDevice, device, graphicsQueue, renderPass, graphicsFamilyIndex, swapChainImages.size());
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createColorResources();
        createDepthResources();
        createRenderPass();
        
        dataProvider = std::make_unique<SplatCloud>();
        if(!dynamic_cast<SplatCloud*>(dataProvider.get())->loadPly("models/Splat/input_image2.ply")) {
             throw std::runtime_error("Failed to load PLY!");
        }
        std::cout << "Loaded: " << dataProvider->GetElementCount() << std::endl;

        createDescriptorSetLayout();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        
        // Create Buffers
        createDataBuffers();
        createSortBuffer(); 
        createUniformBuffer();
        
        createDescriptorPool();
        createDescriptorSets();
        
        createCommandBuffer();
        createSyncObjects();
        init_imgui();
    }

    void createInstance() {
        vk::ApplicationInfo appInfo("SplatRenderer", 1, "No Engine", 1, VK_API_VERSION_1_3);
        uint32_t glfwCount = 0;
        const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwCount);
        std::vector<const char*> extensions(glfwExts, glfwExts + glfwCount);
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        vk::InstanceCreateInfo createInfo(vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR, &appInfo, 0, nullptr, (uint32_t)extensions.size(), extensions.data());
        instance = vk::raii::Instance(context, createInfo);
    }

    void createSurface() {
        VkSurfaceKHR s;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &s) != VK_SUCCESS) throw std::runtime_error("failed to create window surface!");
        surface = vk::raii::SurfaceKHR(instance, s);
    }

    void pickPhysicalDevice() {
        vk::raii::PhysicalDevices devices(instance);
        for (const auto& dev : devices) {
            if (checkDeviceExtensionSupport(dev)) {
                auto swapChainSupport = querySwapChainSupport(dev);
                if (!swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty()) {
                    physicalDevice = dev;
                    break;
                }
            }
        }
        if (*physicalDevice == nullptr) throw std::runtime_error("No suitable GPU!");
        std::cout << "Selected GPU: " << physicalDevice.getProperties().deviceName << std::endl;
    }

    void createLogicalDevice() {
        auto indices = findQueueFamilies(physicalDevice);
        std::vector<vk::DeviceQueueCreateInfo> queueInfos;
        std::set<uint32_t> uniqueFamilies = {indices[0], indices[1]};
        float priority = 1.0f;
        for (uint32_t family : uniqueFamilies) queueInfos.push_back({{}, family, 1, &priority});
        vk::PhysicalDeviceFeatures features{};
        vk::DeviceCreateInfo createInfo({}, queueInfos, {}, deviceExtensions, &features);
        device = vk::raii::Device(physicalDevice, createInfo);
        graphicsQueue = vk::raii::Queue(device, indices[0], 0);
        presentQueue = vk::raii::Queue(device, indices[1], 0);
        graphicsFamilyIndex = indices[0];
        presentFamilyIndex = indices[1];
    }

    void createSwapChain() {
        auto swapChainSupport = querySwapChainSupport(physicalDevice);
        auto surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        auto presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        auto extent = chooseSwapExtent(swapChainSupport.capabilities);
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
            imageCount = swapChainSupport.capabilities.maxImageCount;

        vk::SwapchainCreateInfoKHR createInfo({}, *surface, imageCount, surfaceFormat.format, surfaceFormat.colorSpace, extent, 1, vk::ImageUsageFlagBits::eColorAttachment);
        uint32_t indices[] = {graphicsFamilyIndex, presentFamilyIndex};
        if (graphicsFamilyIndex != presentFamilyIndex) {
            createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = indices;
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
        createImageViews();
    }

    void createImageViews() {
        swapChainImageViews.clear();
        for (const auto& image : swapChainImages) {
            vk::ImageViewCreateInfo createInfo({}, image, vk::ImageViewType::e2D, swapChainImageFormat, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
            swapChainImageViews.emplace_back(device, createInfo);
        }
    }

    void createRenderPass() {
        vk::AttachmentDescription colorAttachment({}, swapChainImageFormat, msaaSamples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
        vk::AttachmentDescription depthAttachment({}, depthFormat, msaaSamples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);
        vk::AttachmentDescription resolveAttachment({}, swapChainImageFormat, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR);

        vk::AttachmentReference colorRef(0, vk::ImageLayout::eColorAttachmentOptimal);
        vk::AttachmentReference depthRef(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);
        vk::AttachmentReference resolveRef(2, vk::ImageLayout::eColorAttachmentOptimal);

        vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &colorRef, &resolveRef, &depthRef);
        
        vk::SubpassDependency dependency(VK_SUBPASS_EXTERNAL, 0, 
            vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests, 
            vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests, 
            {}, vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite);

        std::array<vk::AttachmentDescription, 3> attachments = {colorAttachment, depthAttachment, resolveAttachment};
        vk::RenderPassCreateInfo renderPassInfo({}, (uint32_t)attachments.size(), attachments.data(), 1, &subpass, 1, &dependency);
        renderPass = vk::raii::RenderPass(device, renderPassInfo);
    }

    void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboBind{0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex};
        vk::DescriptorSetLayoutBinding splatBind{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex};
        vk::DescriptorSetLayoutBinding sortBind{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex};
        vk::DescriptorSetLayoutBinding shBind{3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex};

        std::array<vk::DescriptorSetLayoutBinding, 4> bindings = {uboBind, splatBind, sortBind, shBind};
        vk::DescriptorSetLayoutCreateInfo layoutInfo({}, (uint32_t)bindings.size(), bindings.data());
        descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
    }

    void createGraphicsPipeline() {
        auto vertCode = readFile("shaders/vert.spv");
        auto fragCode = readFile("shaders/frag.spv");
        vk::raii::ShaderModule vertModule = createShaderModule(vertCode);
        vk::raii::ShaderModule fragModule = createShaderModule(fragCode);

        vk::PipelineShaderStageCreateInfo stages[] = {
            {{}, vk::ShaderStageFlagBits::eVertex, *vertModule, "main"},
            {{}, vk::ShaderStageFlagBits::eFragment, *fragModule, "main"}
        };

        vk::PipelineVertexInputStateCreateInfo vertexInput{};
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleStrip, VK_FALSE);
        vk::Viewport viewport(0.0f, 0.0f, (float)swapChainExtent.width, (float)swapChainExtent.height, 0.0f, 1.0f);
        vk::Rect2D scissor({0, 0}, swapChainExtent);
        vk::PipelineViewportStateCreateInfo viewportState({}, 1, &viewport, 1, &scissor);
        vk::PipelineRasterizationStateCreateInfo rasterizer({}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise, VK_FALSE, 0.0f, 0.0f, 0.0f, 1.0f);
        vk::PipelineMultisampleStateCreateInfo multisampling({}, msaaSamples, VK_FALSE);
        vk::PipelineDepthStencilStateCreateInfo depthStencil({}, VK_TRUE, VK_TRUE, vk::CompareOp::eLess, VK_FALSE, VK_FALSE);

        // --- BLENDING FIX ---
        vk::PipelineColorBlendAttachmentState colorBlend{};
        colorBlend.blendEnable = VK_TRUE;
        colorBlend.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
        colorBlend.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
        colorBlend.colorBlendOp = vk::BlendOp::eAdd;
        colorBlend.srcAlphaBlendFactor = vk::BlendFactor::eOne;
        colorBlend.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
        colorBlend.alphaBlendOp = vk::BlendOp::eAdd;
        colorBlend.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

        vk::PipelineColorBlendStateCreateInfo colorBlending({}, VK_FALSE, vk::LogicOp::eCopy, 1, &colorBlend);
        vk::PipelineLayoutCreateInfo layoutInfo({}, 1, &*descriptorSetLayout, 0, nullptr);
        pipelineLayout = vk::raii::PipelineLayout(device, layoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo({}, 2, stages, &vertexInput, &inputAssembly, nullptr, &viewportState, &rasterizer, &multisampling, &depthStencil, &colorBlending, nullptr, *pipelineLayout, *renderPass, 0);
        graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    }

    void createFramebuffers() {
        swapChainFramebuffers.clear();
        for (const auto& view : swapChainImageViews) {
            std::array<vk::ImageView, 3> attachments = {*colorImageView, *depthImageView, *view};
            vk::FramebufferCreateInfo fbInfo({}, *renderPass, (uint32_t)attachments.size(), attachments.data(), swapChainExtent.width, swapChainExtent.height, 1);
            swapChainFramebuffers.emplace_back(device, fbInfo);
        }
    }

    // --- BUFFERS ---
    void createDataBuffers() {
        vk::DeviceSize size = dataProvider->GetMainDataSize();
        if (size == 0) return;

        vk::raii::Buffer stagingBuf = nullptr; vk::raii::DeviceMemory stagingMem = nullptr;
        createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuf, stagingMem);

        void* data = stagingMem.mapMemory(0, size);
        memcpy(data, dataProvider->GetMainData(), (size_t)size);
        stagingMem.unmapMemory();

        createBuffer(size, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, splatBuffer, splatBufferMemory);
        copyBuffer(device, commandPool, graphicsQueue, *stagingBuf, *splatBuffer, size);

        if (dataProvider->HasAuxData()) {
            vk::DeviceSize auxSize = dataProvider->GetAuxDataSize();
            vk::raii::Buffer auxStagingBuf = nullptr; vk::raii::DeviceMemory auxStagingMem = nullptr;
            createBuffer(auxSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, auxStagingBuf, auxStagingMem);

            void* auxData = auxStagingMem.mapMemory(0, auxSize);
            memcpy(auxData, dataProvider->GetAuxData(), (size_t)auxSize);
            auxStagingMem.unmapMemory();

            createBuffer(auxSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, shBuffer, shBufferMemory);
            copyBuffer(device, commandPool, graphicsQueue, *auxStagingBuf, *shBuffer, auxSize);
        }
    }

    void createSortBuffer() {
        size_t count = dataProvider->GetElementCount();
        vk::DeviceSize size = sizeof(uint32_t) * count;
        hostSortIndices.resize(count);
        sortProxies.resize(count);
        for(uint32_t i=0; i<count; ++i) { hostSortIndices[i] = i; sortProxies[i].index = i; }

        createBuffer(size, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, sortBuffer, sortBufferMemory);
    }

    void createDescriptorPool() {
        vk::DescriptorPoolSize poolSizes[] = {
            {vk::DescriptorType::eUniformBuffer, 1},
            {vk::DescriptorType::eStorageBuffer, 3}
        };
        vk::DescriptorPoolCreateInfo poolInfo({}, 1, 2, poolSizes);
        descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
    }

    void createDescriptorSets() {
        vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool, *descriptorSetLayout);
        descriptorSets = vk::raii::DescriptorSets(device, allocInfo);

        vk::DescriptorBufferInfo uboInfo(*uniformBuffer, 0, sizeof(UniformBufferObject));
        vk::DescriptorBufferInfo splatInfo(*splatBuffer, 0, VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo sortInfo(*sortBuffer, 0, VK_WHOLE_SIZE);

        std::vector<vk::WriteDescriptorSet> writes;
        
        writes.push_back({*descriptorSets[0], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uboInfo});
        writes.push_back({*descriptorSets[0], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &splatInfo});
        writes.push_back({*descriptorSets[0], 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &sortInfo});

        if (dataProvider->HasAuxData()) {
            vk::DescriptorBufferInfo shInfo(*shBuffer, 0, VK_WHOLE_SIZE);
            writes.push_back({*descriptorSets[0], 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &shInfo});
        }

        device.updateDescriptorSets(writes, nullptr);
    }

    void createUniformBuffer() {
        vk::DeviceSize size = sizeof(UniformBufferObject);
        createBuffer(size, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, uniformBuffer, uniformBufferMemory);
        uniformBufferMapped = uniformBufferMemory.mapMemory(0, size);
    }

    // --- LOGIC ---
    void sortSplats() {
        SplatCloud* cloud = dynamic_cast<SplatCloud*>(dataProvider.get());
        if (!cloud) return;

        glm::vec3 camDir = camera.Front;
        const auto& splats = cloud->GetSplats();
        size_t count = splats.size();

        for (size_t i = 0; i < count; i++) {
            sortProxies[i].index = (uint32_t)i;
            sortProxies[i].depth = glm::dot(glm::vec3(splats[i].pos_opacity), camDir);
        }

        std::sort(sortProxies.begin(), sortProxies.end(), [](const SortProxy& a, const SortProxy& b) {
            return a.depth > b.depth;
        });

        for(size_t i=0; i<count; ++i) hostSortIndices[i] = sortProxies[i].index;

        void* data = sortBufferMemory.mapMemory(0, sizeof(uint32_t) * count);
        memcpy(data, hostSortIndices.data(), sizeof(uint32_t) * count);
        sortBufferMemory.unmapMemory();
    }

    void updateUniformBuffer() {
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
        deltaTime = time - lastFrame; lastFrame = time;

        processInput(window, deltaTime);

        static int sortCounter = 0;
        static int frameCounter = 0;
        sortCounter++; frameCounter++;
        if (sortCounter > 30 || frameCounter < 5) {
            sortSplats();
            sortCounter = 0;
        }

        float fovRad = glm::radians(camera.Zoom);
        float tanHalfFov = tan(fovRad * 0.5f);
        float focal_y = (float)swapChainExtent.height / (2.0f * tanHalfFov);
        float focal_x = (float)swapChainExtent.width / (2.0f * tanHalfFov);

        static bool printed = false;
        if (!printed) {
            std::cout << "[DEBUG] Focal X: " << focal_x << " Focal Y: " << focal_y << std::endl;
            printed = true;
        }

        UniformBufferObject ubo{};
        ubo.view = camera.GetViewMatrix();
        ubo.proj = glm::perspective(fovRad, (float)swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 100.0f);
        ubo.cameraPos = glm::vec4(camera.Position, 1.0f);
        
        ubo.viewport_focal = glm::vec4(
            (float)swapChainExtent.width, 
            (float)swapChainExtent.height, 
            focal_x, 
            focal_y
        );

        memcpy(uniformBufferMapped, &ubo, sizeof(ubo));
    }

    void processInput(GLFWwindow* window, float dt) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);
        float speed = camera.MovementSpeed;
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) camera.MovementSpeed *= 5.0f;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camera.ProcessKeyboard(FORWARD, dt);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camera.ProcessKeyboard(BACKWARD, dt);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camera.ProcessKeyboard(LEFT, dt);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camera.ProcessKeyboard(RIGHT, dt);
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) camera.ProcessKeyboard(DOWN, dt);
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) camera.ProcessKeyboard(UP, dt);
        camera.MovementSpeed = speed;
    }
    
    // --- BOILERPLATE HELPERS ---
    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }
        device.waitIdle();
    }
    void drawFrame() {
        (void)device.waitForFences({*inFlightFence}, VK_TRUE, UINT64_MAX);
        device.resetFences({*inFlightFence});
        auto [res, imgIdx] = swapChain.acquireNextImage(UINT64_MAX, *imageAvailableSemaphore);
        updateUniformBuffer();
        
        gui->NewFrame(); gui->UpdateUI();
        commandBuffers[0].reset();
        commandBuffers[0].begin({});
        
        std::array<vk::ClearValue, 3> clears = { 
            vk::ClearColorValue(std::array<float,4>{0,0,0,1}), 
            vk::ClearDepthStencilValue(1.0f, 0),
            vk::ClearColorValue(std::array<float,4>{0,0,0,1}) 
        };
        vk::RenderPassBeginInfo passInfo(*renderPass, *swapChainFramebuffers[imgIdx], {{0,0}, swapChainExtent}, (uint32_t)clears.size(), clears.data());
        
        commandBuffers[0].beginRenderPass(passInfo, vk::SubpassContents::eInline);
        commandBuffers[0].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        
        // Draw Splats
        commandBuffers[0].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, {*descriptorSets[0]}, nullptr);
        commandBuffers[0].draw(4, dataProvider->GetElementCount(), 0, 0);

        gui->Render(commandBuffers[0]);
        commandBuffers[0].endRenderPass();
        commandBuffers[0].end();

        vk::PipelineStageFlags waitMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        vk::SubmitInfo submitInfo(*imageAvailableSemaphore, waitMask, *commandBuffers[0], *renderFinishedSemaphore);
        graphicsQueue.submit(submitInfo, *inFlightFence);
        vk::PresentInfoKHR presentInfo(*renderFinishedSemaphore, *swapChain, imgIdx);
        (void)presentQueue.presentKHR(presentInfo);
    }
    std::vector<uint32_t> findQueueFamilies(const vk::raii::PhysicalDevice& device) { 
        auto props = device.getQueueFamilyProperties();
        std::vector<uint32_t> indices(2);
        for(uint32_t i=0; i<props.size(); ++i) {
            if(props[i].queueFlags & vk::QueueFlagBits::eGraphics) indices[0] = i;
            if(device.getSurfaceSupportKHR(i, *surface)) indices[1] = i;
        }
        return indices; 
    } 
    bool checkDeviceExtensionSupport(const vk::raii::PhysicalDevice& device) { 
        auto exts = device.enumerateDeviceExtensionProperties();
        std::set<std::string> required(deviceExtensions.begin(), deviceExtensions.end());
        for(const auto& ext : exts) required.erase(ext.extensionName);
        return required.empty();
    } 
    SwapChainSupportDetails querySwapChainSupport(const vk::raii::PhysicalDevice& device) { return {device.getSurfaceCapabilitiesKHR(*surface), device.getSurfaceFormatsKHR(*surface), device.getSurfacePresentModesKHR(*surface)}; }
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& formats) { 
        for(auto& f : formats) if(f.format == vk::Format::eB8G8R8A8Srgb && f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) return f;
        return formats[0]; 
    }
    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& modes) { 
        for(auto& m : modes) if(m == vk::PresentModeKHR::eMailbox) return m;
        return vk::PresentModeKHR::eFifo; 
    }
    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& caps) { 
        if(caps.currentExtent.width != UINT32_MAX) return caps.currentExtent;
        int w, h; glfwGetFramebufferSize(window, &w, &h);
        return { std::clamp((uint32_t)w, caps.minImageExtent.width, caps.maxImageExtent.width), std::clamp((uint32_t)h, caps.minImageExtent.height, caps.maxImageExtent.height) };
    }
    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& memory) {
        vk::BufferCreateInfo info({}, size, usage);
        buffer = vk::raii::Buffer(device, info);
        vk::MemoryRequirements req = buffer.getMemoryRequirements();
        vk::PhysicalDeviceMemoryProperties memProps = physicalDevice.getMemoryProperties();
        uint32_t typeIndex = 0;
        for(uint32_t i=0; i<memProps.memoryTypeCount; ++i) {
            if((req.memoryTypeBits & (1<<i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) { typeIndex = i; break; }
        }
        vk::MemoryAllocateInfo allocInfo(req.size, typeIndex);
        memory = vk::raii::DeviceMemory(device, allocInfo);
        buffer.bindMemory(*memory, 0);
    }
    void copyBuffer(const vk::raii::Device&, const vk::raii::CommandPool&, const vk::raii::Queue& q, vk::Buffer src, vk::Buffer dst, vk::DeviceSize size) {
        vk::CommandBufferAllocateInfo allocInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 1);
        auto cmd = std::move(device.allocateCommandBuffers(allocInfo)[0]);
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        vk::BufferCopy region(0, 0, size);
        cmd.copyBuffer(src, dst, region);
        cmd.end();
        vk::SubmitInfo submit({}, {}, *cmd, {});
        q.submit(submit, nullptr);
        q.waitIdle();
    }
    vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) {
        return vk::raii::ShaderModule(device, vk::ShaderModuleCreateInfo({}, code.size(), (uint32_t*)code.data()));
    }
    void createDepthResources() {
        depthFormat = vk::Format::eD32Sfloat;
        vk::ImageCreateInfo info({}, vk::ImageType::e2D, depthFormat, {swapChainExtent.width, swapChainExtent.height, 1}, 1, 1, msaaSamples, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment);
        depthImage = vk::raii::Image(device, info);
        
        vk::MemoryRequirements req = depthImage.getMemoryRequirements();
        vk::PhysicalDeviceMemoryProperties memProps = physicalDevice.getMemoryProperties();
        uint32_t typeIndex = 0; 
        for(uint32_t i=0; i<memProps.memoryTypeCount; ++i) { if((req.memoryTypeBits & (1<<i)) && (memProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) == vk::MemoryPropertyFlagBits::eDeviceLocal) { typeIndex = i; break; } }
        depthImageMemory = vk::raii::DeviceMemory(device, {req.size, typeIndex});
        depthImage.bindMemory(*depthImageMemory, 0);
        
        vk::ImageViewCreateInfo viewInfo({}, *depthImage, vk::ImageViewType::e2D, depthFormat, {}, {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1});
        depthImageView = vk::raii::ImageView(device, viewInfo);
    }
    void createColorResources() {
        vk::Format colorFormat = swapChainImageFormat;
        vk::ImageCreateInfo info({}, vk::ImageType::e2D, colorFormat, {swapChainExtent.width, swapChainExtent.height, 1}, 1, 1, msaaSamples, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment);
        colorImage = vk::raii::Image(device, info);
        
        vk::MemoryRequirements req = colorImage.getMemoryRequirements();
        vk::PhysicalDeviceMemoryProperties memProps = physicalDevice.getMemoryProperties();
        uint32_t typeIndex = 0; 
        for(uint32_t i=0; i<memProps.memoryTypeCount; ++i) { if((req.memoryTypeBits & (1<<i)) && (memProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) == vk::MemoryPropertyFlagBits::eDeviceLocal) { typeIndex = i; break; } }
        colorImageMemory = vk::raii::DeviceMemory(device, {req.size, typeIndex});
        colorImage.bindMemory(*colorImageMemory, 0);
        
        vk::ImageViewCreateInfo viewInfo({}, *colorImage, vk::ImageViewType::e2D, colorFormat, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
        colorImageView = vk::raii::ImageView(device, viewInfo);
    }
    void createCommandPool() {
        commandPool = vk::raii::CommandPool(device, {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, graphicsFamilyIndex});
    }
    void createCommandBuffer() {
        commandBuffers = vk::raii::CommandBuffers(device, {*commandPool, vk::CommandBufferLevel::ePrimary, 1});
    }
    void createSyncObjects() {
        imageAvailableSemaphore = vk::raii::Semaphore(device, {});
        renderFinishedSemaphore = vk::raii::Semaphore(device, {});
        inFlightFence = vk::raii::Fence(device, {vk::FenceCreateFlagBits::eSignaled});
    }
    void cleanup() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
};

int main() {
    HelloTriangleApplication app;
    try { app.run(); } 
    catch (const std::exception& e) { std::cerr << e.what() << std::endl; return EXIT_FAILURE; }
    return EXIT_SUCCESS;
}