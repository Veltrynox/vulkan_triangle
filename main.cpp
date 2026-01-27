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
        // Говорим GLFW не создавать OpenGL контекст
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    vk::raii::PhysicalDevice physicalDevice = nullptr;

    void initVulkan() {
        createInstance();
        pickPhysicalDevice();
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

    void createInstance() {
        // 1. Описание приложения
        vk::ApplicationInfo appInfo(
            "Hello Triangle",
            VK_MAKE_VERSION(1, 0, 0),
            "No Engine",
            VK_MAKE_VERSION(1, 0, 0),
            VK_API_VERSION_1_3
        );

        // 2. Получаем расширения, нужные для работы GLFW
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        // ВАЖНО ДЛЯ MACOS: Чтобы Vulkan работал через MoltenVK
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        // Это расширение нужно, чтобы корректно работали слои отладки в будущем
        extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

        // 3. Настройка создания Instance
        vk::InstanceCreateInfo createInfo(
            vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR, // Флаг специально для Mac
            &appInfo,
            0, nullptr, // Слои (пока пусто)
            static_cast<uint32_t>(extensions.size()),
            extensions.data()
        );

        // 4. Создание
        try {
            instance = vk::raii::Instance(context, createInfo);
            std::cout << "Vulkan Instance successfully created!" << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("failed to create instance: ") + e.what());
        }
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        // Благодаря RAII (vk::raii::Instance), инстанс удалится сам.
        // Нам нужно закрыть только GLFW.
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