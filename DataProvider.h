#pragma once

#include <vector>
#include <vulkan/vulkan_raii.hpp>

class IDataProvider {
public:
    virtual ~IDataProvider() = default;

    virtual const void* GetMainData() const = 0;
    virtual vk::DeviceSize GetMainDataSize() const = 0;
    virtual uint32_t GetElementCount() const = 0;

    // --- Optional Buffers ---

    virtual const void* GetAuxData() const { return nullptr; }
    virtual vk::DeviceSize GetAuxDataSize() const = 0;
    virtual bool HasAuxData() const { return GetAuxData() != nullptr; }
};
