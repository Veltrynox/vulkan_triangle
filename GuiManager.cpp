#include "GuiManager.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <stdexcept>

GuiManager::GuiManager(GLFWwindow* window, 
                       const vk::raii::Instance& instance, 
                       const vk::raii::PhysicalDevice& physicalDevice, 
                       const vk::raii::Device& device, 
                       const vk::raii::Queue& graphicsQueue, 
                       const vk::raii::RenderPass& renderPass,
                       uint32_t queueFamily,
                       uint32_t imageCount) {
    
    createDescriptorPool(device);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = *instance;
    init_info.PhysicalDevice = *physicalDevice;
    init_info.Device = *device;
    init_info.QueueFamily = queueFamily;
    init_info.Queue = *graphicsQueue;
    init_info.PipelineCache = nullptr;
    init_info.DescriptorPool = *imguiPool;

    init_info.PipelineInfoMain.RenderPass = *renderPass;
    init_info.PipelineInfoMain.Subpass = 0;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_4_BIT; 
    
    init_info.MinImageCount = 2;
    init_info.ImageCount = imageCount;
    init_info.Allocator = nullptr;
    init_info.CheckVkResultFn = [](VkResult err) {
        if (err == 0) return;
        fprintf(stderr, "[ImGui] Vulkan Error: %d\n", err);
    };
    ImGui_ImplVulkan_Init(&init_info);
}

GuiManager::~GuiManager() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void GuiManager::createDescriptorPool(const vk::raii::Device& device) {
    std::vector<vk::DescriptorPoolSize> poolSizes = {
        {vk::DescriptorType::eSampler, 1000},
        {vk::DescriptorType::eCombinedImageSampler, 1000},
        {vk::DescriptorType::eSampledImage, 1000},
        {vk::DescriptorType::eStorageImage, 1000},
        {vk::DescriptorType::eUniformTexelBuffer, 1000},
        {vk::DescriptorType::eStorageTexelBuffer, 1000},
        {vk::DescriptorType::eUniformBuffer, 1000},
        {vk::DescriptorType::eStorageBuffer, 1000},
        {vk::DescriptorType::eUniformBufferDynamic, 1000},
        {vk::DescriptorType::eStorageBufferDynamic, 1000},
        {vk::DescriptorType::eInputAttachment, 1000}
    };

    vk::DescriptorPoolCreateInfo poolInfo({}, 1000, poolSizes);
    imguiPool = vk::raii::DescriptorPool(device, poolInfo);
}

void GuiManager::NewFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void GuiManager::UpdateUI() {
    ImGui::Begin("Engine Settings");
    static float rotationSpeed = 45.0f;
    ImGui::SliderFloat("Rotation Speed", &rotationSpeed, 0.0f, 360.0f);
    ImGui::End();
}

void GuiManager::Render(const vk::raii::CommandBuffer& commandBuffer) {
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *commandBuffer);
}