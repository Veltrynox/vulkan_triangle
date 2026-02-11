#include "Splat.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>
#include <cstring> // для memcpy

// Вспомогательная структура для свойства
struct PlyProperty {
    std::string name;
    std::string type;
    int sizeBytes;
    int offset; // Смещение в байтах от начала вершины
};

// Функция для определения размера типа PLY
int getPlyTypeSize(const std::string& type) {
    if (type == "float" || type == "float32") return 4;
    if (type == "uchar" || type == "uint8") return 1;
    if (type == "double" || type == "float64") return 8;
    if (type == "int" || type == "int32") return 4;
    if (type == "short" || type == "int16") return 2;
    return 4; // Fallback (лучше перебдеть)
}

// Вспомогательная функция для чтения значения из буфера с учетом типа
float readAsFloat(const uint8_t* ptr, const std::string& type) {
    if (type == "float" || type == "float32") return *(float*)ptr;
    if (type == "uchar" || type == "uint8") return (float)(*(uint8_t*)ptr) / 255.0f; // Нормализуем цвет 0..255 -> 0..1
    if (type == "double" || type == "float64") return (float)(*(double*)ptr);
    return 0.0f;
}

bool SplatCloud::loadPly(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    // 1. Парсим заголовок
    std::string line;
    uint32_t vertexCount = 0;
    std::vector<PlyProperty> properties;
    int currentOffset = 0;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        ss >> token;
        
        if (token == "end_header") break;
        
        if (token == "element") {
            ss >> token;
            if (token == "vertex") {
                ss >> vertexCount;
            }
        }
        else if (token == "property") {
            std::string type, name;
            ss >> type >> name;
            
            PlyProperty prop;
            prop.name = name;
            prop.type = type;
            prop.sizeBytes = getPlyTypeSize(type);
            prop.offset = currentOffset;
            
            properties.push_back(prop);
            currentOffset += prop.sizeBytes;
        }
    }

    if (vertexCount == 0) return false;

    size_t vertexStride = currentOffset; // Реальный размер одной вершины в байтах
    
    std::cout << "Detected PLY Layout: " << properties.size() << " props, " 
              << vertexStride << " bytes per vertex." << std::endl;
    std::cout << "Loading " << vertexCount << " splats..." << std::endl;

    // Проверка размера файла
    auto currentPos = file.tellg();
    file.seekg(0, std::ios::end);
    size_t fileSize = static_cast<size_t>(file.tellg()) - static_cast<size_t>(currentPos);
    file.seekg(currentPos);

    if (fileSize < vertexCount * vertexStride) {
        std::cerr << "File too small! Expected " << (vertexCount * vertexStride) 
                  << " bytes, found " << fileSize << std::endl;
        return false;
    }

    // 2. Читаем ВСЕ данные в сырой буфер
    std::vector<uint8_t> rawBytes(vertexCount * vertexStride);
    file.read(reinterpret_cast<char*>(rawBytes.data()), rawBytes.size());

    // 3. Создаем карту для быстрого поиска свойств
    std::map<std::string, PlyProperty> layout;
    for(const auto& p : properties) layout[p.name] = p;

    // 4. Подготавливаем GPU данные
    splats.resize(vertexCount);
    
    // Проверяем наличие SH
    bool hasSH = layout.count("f_rest_0");
    if (hasSH) {
        shData.resize(vertexCount * 45);
    } else {
        shData.clear();
    }

    const float SH_C0 = 0.28209479177f;

    // Кэшируем свойства для скорости (чтобы не искать в map в цикле)
    auto getProp = [&](const std::string& name) -> PlyProperty* {
        if (layout.count(name)) return &layout[name];
        return nullptr;
    };

    PlyProperty* p_x = getProp("x");
    PlyProperty* p_y = getProp("y");
    PlyProperty* p_z = getProp("z");
    
    PlyProperty* p_s0 = getProp("scale_0");
    PlyProperty* p_s1 = getProp("scale_1");
    PlyProperty* p_s2 = getProp("scale_2");

    PlyProperty* p_r0 = getProp("rot_0");
    PlyProperty* p_r1 = getProp("rot_1");
    PlyProperty* p_r2 = getProp("rot_2");
    PlyProperty* p_r3 = getProp("rot_3");

    PlyProperty* p_op = getProp("opacity");

    // Colors (могут быть f_dc или red/green/blue)
    PlyProperty* p_dc0 = getProp("f_dc_0"); if(!p_dc0) p_dc0 = getProp("red");
    PlyProperty* p_dc1 = getProp("f_dc_1"); if(!p_dc1) p_dc1 = getProp("green");
    PlyProperty* p_dc2 = getProp("f_dc_2"); if(!p_dc2) p_dc2 = getProp("blue");

    // SH start
    PlyProperty* p_sh0 = getProp("f_rest_0");

    // Главный цикл парсинга
    for (size_t i = 0; i < vertexCount; i++) {
        const uint8_t* vPtr = rawBytes.data() + i * vertexStride;

        // Position
        splats[i].pos_opacity.x = readAsFloat(vPtr + p_x->offset, p_x->type);
        splats[i].pos_opacity.y = readAsFloat(vPtr + p_y->offset, p_y->type);
        splats[i].pos_opacity.z = readAsFloat(vPtr + p_z->offset, p_z->type);

        // Scale (Exp)
        splats[i].scale.x = std::exp(readAsFloat(vPtr + p_s0->offset, p_s0->type));
        splats[i].scale.y = std::exp(readAsFloat(vPtr + p_s1->offset, p_s1->type));
        splats[i].scale.z = std::exp(readAsFloat(vPtr + p_s2->offset, p_s2->type));
        splats[i].scale.w = 0.0f;

        // Rotation (Normalize)
        // В файле rot_0 обычно первый (Real/W), но движок ждет порядок XYZW
        float r0 = readAsFloat(vPtr + p_r0->offset, p_r0->type);
        float r1 = readAsFloat(vPtr + p_r1->offset, p_r1->type);
        float r2 = readAsFloat(vPtr + p_r2->offset, p_r2->type);
        float r3 = readAsFloat(vPtr + p_r3->offset, p_r3->type);
        
        // Стандарт 3DGS: [0]=r, [1]=x, [2]=y, [3]=z
        // GPU ожидает: x, y, z, w
        splats[i].rot = glm::vec4(r1, r2, r3, r0);
        float len = glm::length(splats[i].rot);
        if (len > 0) splats[i].rot /= len;

        // Opacity (Sigmoid)
        float op_val = readAsFloat(vPtr + p_op->offset, p_op->type);
        splats[i].pos_opacity.w = 1.0f / (1.0f + std::exp(-op_val));

        // Color
        if (p_dc0) {
            float c0 = readAsFloat(vPtr + p_dc0->offset, p_dc0->type);
            float c1 = readAsFloat(vPtr + p_dc1->offset, p_dc1->type);
            float c2 = readAsFloat(vPtr + p_dc2->offset, p_dc2->type);

            // Если тип был uchar (0..255), readAsFloat уже вернул 0..1
            // Если тип float и это SH DC, конвертируем
            if (p_dc0->type == "float" || p_dc0->type == "float32") {
                splats[i].color.x = 0.5f + SH_C0 * c0;
                splats[i].color.y = 0.5f + SH_C0 * c1;
                splats[i].color.z = 0.5f + SH_C0 * c2;
            } else {
                // Если это обычный uchar цвет
                splats[i].color.x = c0;
                splats[i].color.y = c1;
                splats[i].color.z = c2;
            }
        } else {
            splats[i].color = glm::vec4(1,0,1,1);
        }
        splats[i].color.w = 1.0f;

        // SH
        if (hasSH) {
            // SH обычно всегда float, но на всякий случай используем stride
            // Мы предполагаем, что SH идут подряд: f_rest_0, f_rest_1...
            // Чтобы не искать каждое свойство, вычислим начало и шаг
            int sh_start_offset = p_sh0->offset;
            int sh_step = p_sh0->sizeBytes; // Предполагаем одинаковый тип для всех SH
            
            for (int j = 0; j < 45; j++) {
                // Простая защита от выхода за пределы вершины
                if (sh_start_offset + j * sh_step < vertexStride) {
                    // ВАЖНО: Тут мы хардкодим тип float, т.к. SH в uchar не хранят
                    // Если вдруг хранят - нужно добавить проверку
                    shData[i * 45 + j] = *(float*)(vPtr + sh_start_offset + j * sh_step);
                } else {
                    shData[i * 45 + j] = 0.0f;
                }
            }
        }
    }

    std::cout << "Successfully loaded " << vertexCount << " splats." << std::endl;
    return true;
}