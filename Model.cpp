#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE

#include "stb_image.h"

#include "gltf/tiny_gltf.h"
#include "Model.h"
#include <iostream>

Model::Model(const vk::raii::Device& device, const vk::raii::PhysicalDevice& physicalDevice, 
             const vk::raii::CommandPool& commandPool, const vk::raii::Queue& queue, 
             const std::string& path) {
    loadModel(path);
    createVertexBuffers(device, physicalDevice, commandPool, queue);
    createIndexBuffers(device, physicalDevice, commandPool, queue);
}

void Model::loadModel(const std::string& path) {
    tinygltf::Model gltfModel;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    bool ret = false;

    if (path.find(".glb") != std::string::npos) {
        ret = loader.LoadBinaryFromFile(&gltfModel, &err, &warn, path);
    } else {
        ret = loader.LoadASCIIFromFile(&gltfModel, &err, &warn, path);
    }

    if (!warn.empty()) std::cout << "GLTF Warn: " << warn << std::endl;
    if (!err.empty()) std::cerr << "GLTF Err: " << err << std::endl;
    if (!ret) throw std::runtime_error("Failed to parse glTF: " + path);

    const tinygltf::Mesh& mesh = gltfModel.meshes[0];
    const tinygltf::Primitive& primitive = mesh.primitives[0];

    const tinygltf::Accessor& indexAccessor = gltfModel.accessors[primitive.indices];
    const tinygltf::BufferView& indexBufferView = gltfModel.bufferViews[indexAccessor.bufferView];
    const tinygltf::Buffer& indexGltfBuffer = gltfModel.buffers[indexBufferView.buffer];
    const void* dataPtr = &(indexGltfBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]);

    if (indexAccessor.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT) {
        const uint32_t* buf = static_cast<const uint32_t*>(dataPtr);
        for (size_t i = 0; i < indexAccessor.count; i++) indices.push_back(buf[i]);
    } else if (indexAccessor.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT) {
        const uint16_t* buf = static_cast<const uint16_t*>(dataPtr);
        for (size_t i = 0; i < indexAccessor.count; i++) indices.push_back(buf[i]);
    }

    const float* positionData = nullptr;
    const float* texCoordData = nullptr;

    if (primitive.attributes.find("POSITION") != primitive.attributes.end()) {
        const tinygltf::Accessor& posAccessor = gltfModel.accessors[primitive.attributes.at("POSITION")];
        const tinygltf::BufferView& posView = gltfModel.bufferViews[posAccessor.bufferView];
        positionData = reinterpret_cast<const float*>(&gltfModel.buffers[posView.buffer].data[posView.byteOffset + posAccessor.byteOffset]);
        vertices.resize(posAccessor.count);
    }

    if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
        const tinygltf::Accessor& texAccessor = gltfModel.accessors[primitive.attributes.at("TEXCOORD_0")];
        const tinygltf::BufferView& texView = gltfModel.bufferViews[texAccessor.bufferView];
        texCoordData = reinterpret_cast<const float*>(&gltfModel.buffers[texView.buffer].data[texView.byteOffset + texAccessor.byteOffset]);
    }

    for (size_t i = 0; i < vertices.size(); ++i) {
        vertices[i].pos = glm::vec3(positionData[i * 3 + 0], positionData[i * 3 + 1], positionData[i * 3 + 2]);
        vertices[i].color = glm::vec3(1.0f);
        if (texCoordData) vertices[i].texCoord = glm::vec2(texCoordData[i * 2 + 0], texCoordData[i * 2 + 1]);
    }
}

void Model::createVertexBuffers(const vk::raii::Device& device, const vk::raii::PhysicalDevice& physicalDevice, const vk::raii::CommandPool& commandPool, const vk::raii::Queue& queue) {
    vk::DeviceSize bufferSize = sizeof(Vertex) * vertices.size();
    vk::raii::Buffer stagingBuffer = nullptr;
    vk::raii::DeviceMemory stagingMemory = nullptr;

    createBuffer(device, physicalDevice, bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingMemory);

    void* data = stagingMemory.mapMemory(0, bufferSize);
    memcpy(data, vertices.data(), (size_t)bufferSize);
    stagingMemory.unmapMemory();

    createBuffer(device, physicalDevice, bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);
    copyBuffer(device, commandPool, queue, *stagingBuffer, *vertexBuffer, bufferSize);
}

void Model::createIndexBuffers(const vk::raii::Device& device, const vk::raii::PhysicalDevice& physicalDevice, const vk::raii::CommandPool& commandPool, const vk::raii::Queue& queue) {
    vk::DeviceSize bufferSize = sizeof(uint32_t) * indices.size();
    vk::raii::Buffer stagingBuffer = nullptr;
    vk::raii::DeviceMemory stagingMemory = nullptr;

    createBuffer(device, physicalDevice, bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingMemory);

    void* data = stagingMemory.mapMemory(0, bufferSize);
    memcpy(data, indices.data(), (size_t)bufferSize);
    stagingMemory.unmapMemory();

    createBuffer(device, physicalDevice, bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);
    copyBuffer(device, commandPool, queue, *stagingBuffer, *indexBuffer, bufferSize);
}

void Model::createBuffer(const vk::raii::Device& device, const vk::raii::PhysicalDevice& physicalDevice, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory) {
    vk::BufferCreateInfo bufferInfo({}, size, usage, vk::SharingMode::eExclusive);
    buffer = vk::raii::Buffer(device, bufferInfo);
    vk::MemoryRequirements memReq = buffer.getMemoryRequirements();
    bufferMemory = vk::raii::DeviceMemory(device, {memReq.size, findMemoryType(physicalDevice, memReq.memoryTypeBits, properties)});
    buffer.bindMemory(*bufferMemory, 0);
}

void Model::copyBuffer(const vk::raii::Device& device, const vk::raii::CommandPool& commandPool, const vk::raii::Queue& queue, vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) {
    vk::CommandBufferAllocateInfo allocInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 1);
    vk::raii::CommandBuffer copyCmd = std::move(device.allocateCommandBuffers(allocInfo).front());
    copyCmd.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    copyCmd.copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy(0, 0, size));
    copyCmd.end();
    queue.submit(vk::SubmitInfo({}, {}, *copyCmd, {}), nullptr);
    queue.waitIdle();
}

uint32_t Model::findMemoryType(const vk::raii::PhysicalDevice& physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) return i;
    }
    throw std::runtime_error("failed to find suitable memory type!");
}