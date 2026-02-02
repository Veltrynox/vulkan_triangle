#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Texture.h"
#include <stdexcept>

Texture::Texture(const vk::raii::Device& device, 
                 const vk::raii::PhysicalDevice& physicalDevice, 
                 const vk::raii::CommandPool& commandPool, 
                 const vk::raii::Queue& queue, 
                 const std::string& path) {

    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    vk::DeviceSize imageSize = texWidth * texHeight * 4;

    if (!pixels) {
        throw std::runtime_error("failed to load texture image: " + path);
    }

    // 1. Create Staging Buffer
    vk::BufferCreateInfo stagingBufferInfo({}, imageSize, vk::BufferUsageFlagBits::eTransferSrc);
    vk::raii::Buffer stagingBuffer(device, stagingBufferInfo);

    vk::MemoryRequirements stagingMemReq = stagingBuffer.getMemoryRequirements();
    vk::MemoryAllocateInfo stagingAllocInfo(stagingMemReq.size, 
        findMemoryType(physicalDevice, stagingMemReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
    
    vk::raii::DeviceMemory stagingBufferMemory(device, stagingAllocInfo);
    stagingBuffer.bindMemory(*stagingBufferMemory, 0);

    // Copy pixels to staging buffer
    void* data = stagingBufferMemory.mapMemory(0, imageSize);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    stagingBufferMemory.unmapMemory();
    stbi_image_free(pixels);

    // 2. Create GPU Image
    vk::ImageCreateInfo imageInfo({}, vk::ImageType::e2D, vk::Format::eR8G8B8A8Srgb, 
        {static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1}, 
        1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, 
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);

    image = vk::raii::Image(device, imageInfo);

    vk::MemoryRequirements memReq = image.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo(memReq.size, 
        findMemoryType(physicalDevice, memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal));

    imageMemory = vk::raii::DeviceMemory(device, allocInfo);
    image.bindMemory(*imageMemory, 0);

    // 3. Layout Transitions and Copy
    transitionImageLayout(device, commandPool, queue, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
    copyBufferToImage(device, commandPool, queue, *stagingBuffer, texWidth, texHeight);
    transitionImageLayout(device, commandPool, queue, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    // Create Image View
    vk::ImageViewCreateInfo viewInfo({}, *image, vk::ImageViewType::e2D, vk::Format::eR8G8B8A8Srgb, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    imageView = vk::raii::ImageView(device, viewInfo);

    // Create Sampler
    vk::SamplerCreateInfo samplerInfo({}, vk::Filter::eLinear, vk::Filter::eLinear, 
        vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eRepeat, 
        vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, 
        0.0f, VK_FALSE, 1.0f, VK_FALSE, vk::CompareOp::eAlways, 0.0f, 0.0f, 
        vk::BorderColor::eIntOpaqueBlack, VK_FALSE);

    sampler = vk::raii::Sampler(device, samplerInfo);
}

void Texture::transitionImageLayout(const vk::raii::Device& device, const vk::raii::CommandPool& commandPool, const vk::raii::Queue& queue, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
    vk::CommandBufferAllocateInfo allocInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 1);
    vk::raii::CommandBuffers cb(device, allocInfo);
    vk::raii::CommandBuffer commandBuffer = std::move(cb[0]);

    commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    vk::ImageMemoryBarrier barrier({}, {}, oldLayout, newLayout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, *image, 
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

    vk::PipelineStageFlags sourceStage, destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    }

    commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, barrier);
    commandBuffer.end();

    vk::SubmitInfo submitInfo({}, {}, *commandBuffer, {});
    queue.submit(submitInfo, nullptr);
    queue.waitIdle();
}

void Texture::copyBufferToImage(const vk::raii::Device& device, const vk::raii::CommandPool& commandPool, const vk::raii::Queue& queue, vk::Buffer buffer, uint32_t width, uint32_t height) {
    vk::CommandBufferAllocateInfo allocInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 1);
    vk::raii::CommandBuffers cb(device, allocInfo);
    vk::raii::CommandBuffer commandBuffer = std::move(cb[0]);

    commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    vk::BufferImageCopy region(0, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {0, 0, 0}, {width, height, 1});
    commandBuffer.copyBufferToImage(buffer, *image, vk::ImageLayout::eTransferDstOptimal, region);
    commandBuffer.end();

    vk::SubmitInfo submitInfo({}, {}, *commandBuffer, {});
    queue.submit(submitInfo, nullptr);
    queue.waitIdle();
}

uint32_t Texture::findMemoryType(const vk::raii::PhysicalDevice& physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}