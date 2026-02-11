#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <string>

#include "DataProvider.h"

struct GaussianData {
    alignas(16) glm::vec4 pos_opacity;
    alignas(16) glm::vec4 scale;
    alignas(16) glm::vec4 color;
    alignas(16) glm::vec4 rot;
};

class SplatCloud : public IDataProvider {
public:
    bool loadPly(const std::string& filename);

    const std::vector<GaussianData>& GetSplats() const { return splats; }
    uint32_t GetCount() const { return static_cast<uint32_t>(splats.size()); }
    const std::vector<float>& GetSHData() const { return shData; }

    // --- IDataProvider Implementation ---
    const void* GetMainData() const override { return splats.data(); }
    vk::DeviceSize GetMainDataSize() const override { return sizeof(GaussianData) * splats.size(); }
    uint32_t GetElementCount() const override { return GetCount(); }

    const void* GetAuxData() const override { return shData.empty() ? nullptr : shData.data(); }
    vk::DeviceSize GetAuxDataSize() const override { return sizeof(float) * shData.size(); }
    bool HasAuxData() const override { return !shData.empty(); }


private:
    std::vector<GaussianData> splats;
    std::vector<float> shData;
    const float SH_C0 = 0.28209479177f;
};