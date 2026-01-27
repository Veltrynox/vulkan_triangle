#include <GLFW/glfw3.h>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
    #include <vulkan/vulkan_raii.hpp>
#else
    import vulkan_hpp;
#endif

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    const uint32_t WIDTH = 800;
    const uint32_t HEIGHT = 600;

    GLFWwindow* window;
    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;
    vk::raii::Queue graphicsQueue = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;

    void initVulkan() {
        createInstance();
        pickPhysicalDevice();
        createLogicalDevice();
    }

    void createInstance() {
        vk::ApplicationInfo appInfo(
            "Hello Triangle",
            VK_MAKE_VERSION(1, 0, 0),
            "No Engine",
            VK_MAKE_VERSION(1, 0, 0),
            VK_API_VERSION_1_3
        );

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

        vk::InstanceCreateInfo createInfo(
            vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR,
            &appInfo,
            0, nullptr,
            static_cast<uint32_t>(extensions.size()),
            extensions.data()
        );

        try {
            instance = vk::raii::Instance(context, createInfo);
            std::cout << "Vulkan Instance successfully created!" << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("failed to create instance: ") + e.what());
        }
    }

    void pickPhysicalDevice() {
        vk::raii::PhysicalDevices devices(instance);
        
        if (devices.empty()) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }
        
        physicalDevice = devices[0];
        
        auto properties = physicalDevice.getProperties();
        std::cout << "Selected GPU: " << properties.deviceName << std::endl;
    }

    void createLogicalDevice() {
        auto queueFamilies = physicalDevice.getQueueFamilyProperties();
        
        int graphicsFamilyIndex = -1;
        for (int i = 0; i < queueFamilies.size(); i++) {
            if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                graphicsFamilyIndex = i;
                break;
            }
        }

        if (graphicsFamilyIndex == -1) {
            throw std::runtime_error("failed to find a graphics queue family!");
        }

        float queuePriority = 1.0f;
        vk::DeviceQueueCreateInfo queueCreateInfo(
            {},
            static_cast<uint32_t>(graphicsFamilyIndex),
            1,
            &queuePriority
        );

        vk::PhysicalDeviceFeatures deviceFeatures{};

        std::vector<const char*> deviceExtensions;
        deviceExtensions.push_back("VK_KHR_portability_subset");

        vk::DeviceCreateInfo createInfo(
            {},
            queueCreateInfo,
            {},
            deviceExtensions,
            &deviceFeatures
        );

        try {
            device = vk::raii::Device(physicalDevice, createInfo);
            std::cout << "Logical Device successfully created!" << std::endl;

            graphicsQueue = vk::raii::Queue(device, graphicsFamilyIndex, 0);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("failed to create logical device: ") + e.what());
        }
    }


    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}