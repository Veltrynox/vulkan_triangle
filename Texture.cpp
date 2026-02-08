#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Texture.h"
#include <cmath>
#include <algorithm>
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

    mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

    vk::BufferCreateInfo stagingBufferInfo({}, imageSize, vk::BufferUsageFlagBits::eTransferSrc);
    vk::raii::Buffer stagingBuffer(device, stagingBufferInfo);

    vk::MemoryRequirements stagingMemReq = stagingBuffer.getMemoryRequirements();
    vk::MemoryAllocateInfo stagingAllocInfo(stagingMemReq.size, 
        findMemoryType(physicalDevice, stagingMemReq.memoryTypeBits, 
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
    
    vk::raii::DeviceMemory stagingBufferMemory(device, stagingAllocInfo);
    stagingBuffer.bindMemory(*stagingBufferMemory, 0);

    void* data = stagingBufferMemory.mapMemory(0, imageSize);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    stagingBufferMemory.unmapMemory();
    stbi_image_free(pixels);

    vk::ImageCreateInfo imageInfo({}, vk::ImageType::e2D, vk::Format::eR8G8B8A8Srgb, 
        {static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1}, 
        mipLevels, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, 
        vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::SharingMode::eExclusive);

    image = vk::raii::Image(device, imageInfo);

    vk::MemoryRequirements memReq = image.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo(memReq.size, 
        findMemoryType(physicalDevice, memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal));

    imageMemory = vk::raii::DeviceMemory(device, allocInfo);
    image.bindMemory(*imageMemory, 0);

    transitionImageLayout(device, commandPool, queue, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

    copyBufferToImage(device, commandPool, queue, *stagingBuffer, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
    
    generateMipmaps(device, commandPool, queue, texWidth, texHeight);

    vk::ImageSubresourceRange subresourceRange(vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 1);
    vk::ImageViewCreateInfo viewInfo({}, *image, vk::ImageViewType::e2D, vk::Format::eR8G8B8A8Srgb, {}, subresourceRange);
    imageView = vk::raii::ImageView(device, viewInfo);

    vk::SamplerCreateInfo samplerInfo({}, 
        vk::Filter::eLinear, vk::Filter::eLinear, 
        vk::SamplerMipmapMode::eLinear, 
        vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, 
        0.0f, VK_FALSE, 1.0f, VK_FALSE, vk::CompareOp::eAlways, 
        0.0f, static_cast<float>(mipLevels),
        vk::BorderColor::eIntOpaqueBlack, VK_FALSE);

    sampler = vk::raii::Sampler(device, samplerInfo);
}

void Texture::generateMipmaps(const vk::raii::Device& device, const vk::raii::CommandPool& commandPool, 
                              const vk::raii::Queue& queue, int32_t width, int32_t height) {
    
    vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);

    vk::ImageMemoryBarrier barrier{};
    barrier.image = *image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mipWidth = width;
    int32_t mipHeight = height;

    for (uint32_t i = 1; i < mipLevels; i++) {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, nullptr, nullptr, barrier);

        vk::ImageBlit blit{};
        blit.srcOffsets[1] = vk::Offset3D{mipWidth, mipHeight, 1};
        blit.srcSubresource = {vk::ImageAspectFlagBits::eColor, i - 1, 0, 1};
        blit.dstOffsets[1] = vk::Offset3D{mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1};
        blit.dstSubresource = {vk::ImageAspectFlagBits::eColor, i, 0, 1};

        commandBuffer.blitImage(*image, vk::ImageLayout::eTransferSrcOptimal, *image, vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);

        barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, nullptr, nullptr, barrier);

        if (mipWidth > 1) mipWidth /= 2;
        if (mipHeight > 1) mipHeight /= 2;
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, nullptr, nullptr, barrier);

    endSingleTimeCommands(std::move(commandBuffer), queue);
}

vk::raii::CommandBuffer Texture::beginSingleTimeCommands(const vk::raii::Device& device, const vk::raii::CommandPool& commandPool) {
    vk::CommandBufferAllocateInfo allocInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 1);
    vk::raii::CommandBuffers cb(device, allocInfo);
    vk::raii::CommandBuffer commandBuffer = std::move(cb[0]);
    commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    return commandBuffer;
}

void Texture::endSingleTimeCommands(vk::raii::CommandBuffer cb, const vk::raii::Queue& queue) {
    cb.end();
    vk::SubmitInfo submitInfo({}, {}, *cb, {});
    queue.submit(submitInfo, nullptr);
    queue.waitIdle();
}

void Texture::transitionImageLayout(const vk::raii::Device& device, const vk::raii::CommandPool& commandPool, 
                                     const vk::raii::Queue& queue, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
    auto commandBuffer = beginSingleTimeCommands(device, commandPool);

    vk::ImageMemoryBarrier barrier({}, {}, oldLayout, newLayout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, *image, 
        {vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 1});

    vk::PipelineStageFlags sourceStage, destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, barrier);
    endSingleTimeCommands(std::move(commandBuffer), queue);
}

void Texture::copyBufferToImage(const vk::raii::Device& device, const vk::raii::CommandPool& commandPool, 
                                const vk::raii::Queue& queue, vk::Buffer buffer, uint32_t width, uint32_t height) {
    auto commandBuffer = beginSingleTimeCommands(device, commandPool);
    vk::BufferImageCopy region(0, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {0, 0, 0}, {width, height, 1});
    commandBuffer.copyBufferToImage(buffer, *image, vk::ImageLayout::eTransferDstOptimal, region);
    endSingleTimeCommands(std::move(commandBuffer), queue);
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