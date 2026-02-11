#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <string>

struct GaussianData {
    alignas(16) glm::vec4 pos_opacity;
    alignas(16) glm::vec4 scale;
    alignas(16) glm::vec4 color;
    alignas(16) glm::vec4 rot;
};

class SplatCloud {
public:
    bool loadPly(const std::string& filename);
    const std::vector<GaussianData>& GetSplats() const { return splats; }
    uint32_t GetCount() const { return static_cast<uint32_t>(splats.size()); }
    const std::vector<float>& GetSHData() const { return shData; }

private:
    std::vector<GaussianData> splats;
    std::vector<float> shData;
    const float SH_C0 = 0.28209479177f;
};