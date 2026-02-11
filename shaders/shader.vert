// SH Constants
static const float SH_C1 = 0.4886025119f;
static const float SH_C2[] = { 1.09254843059f, -1.09254843059f, 0.31539156525f, -1.09254843059f, 0.54627421529f };
static const float SH_C3[] = { -0.59004358992f, 2.89061144264f, -0.45704579946f, 0.37317633259f, -0.45704579946f, 1.44530572132f, -0.59004358992f };

struct GaussianData {
    float4 pos_opacity;
    float4 scale;
    float4 color;
    float4 rot;
};

struct UBO {
    float4x4 view;
    float4x4 proj;
    float4 cameraPos;
    float4 viewport_focal;
};

[[vk::binding(0)]] ConstantBuffer<UBO> ubo : register(b0);
[[vk::binding(1)]] StructuredBuffer<GaussianData> splats : register(t0);
[[vk::binding(2)]] StructuredBuffer<uint> splatIndices : register(t1);
[[vk::binding(3)]] StructuredBuffer<float> shBuffer : register(t2);

struct VSOutput {
    float4 pos : SV_POSITION;
    [[vk::location(0)]] float4 color : COLOR;
    [[vk::location(1)]] float2 uv : TEXCOORD0;
    [[vk::location(2)]] float3 conic : TEXCOORD1;
};

static const float2 quadOffsets[4] = {
    float2(-1.0, -1.0), float2(1.0, -1.0),
    float2(-1.0,  1.0), float2(1.0,  1.0)
};

// --- Helper Functions ---

float3x3 BuildRotation(float4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    return float3x3(
        1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z),       2.0*(x*z + w*y),
        2.0*(x*y + w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x),
        2.0*(x*z - w*y),       2.0*(y*z + w*x),       1.0 - 2.0*(x*x + y*y)
    );
}

float3x3 ComputeCov3D(float3 scale, float4 rot) {
    float3x3 R = BuildRotation(rot);
    
    float3x3 M = float3x3(
        R[0][0] * scale.x, R[0][1] * scale.y, R[0][2] * scale.z,
        R[1][0] * scale.x, R[1][1] * scale.y, R[1][2] * scale.z,
        R[2][0] * scale.x, R[2][1] * scale.y, R[2][2] * scale.z
    );

    return mul(M, transpose(M));
}

float3 ComputeCov2D(float3 center_cam, float2 focal, float2 viewport, float3x3 cov3d, float4x4 viewMat) {
    float3x3 W = (float3x3)viewMat;
    float3x3 T = mul(mul(W, cov3d), transpose(W));

    float z = -center_cam.z;
    
    float f_x = focal.x;
    float f_y = focal.y;
    
    float3x3 J = float3x3(
        f_x / z,   0.0f,      -(f_x * center_cam.x) / (z * z),
        0.0f,      f_y / z,   -(f_y * center_cam.y) / (z * z),
        0.0f,      0.0f,      0.0f
    );

    float3x3 cov_final = mul(mul(J, T), transpose(J));
    
    return float3(cov_final[0][0], cov_final[0][1], cov_final[1][1]);
}

float3 ComputeColorFromSH(float3 pos, float3 camPos, float3 baseColor, uint shIndex) {
    float3 dir = normalize(pos - camPos);
    float3 result = baseColor - 0.5f;
    
    float x = dir.x; float y = dir.y; float z = dir.z;
    
    result.r += SH_C1 * (-y * shBuffer[shIndex + 0] + z * shBuffer[shIndex + 1] - x * shBuffer[shIndex + 2]);
    result.g += SH_C1 * (-y * shBuffer[shIndex + 15] + z * shBuffer[shIndex + 16] - x * shBuffer[shIndex + 17]);
    result.b += SH_C1 * (-y * shBuffer[shIndex + 30] + z * shBuffer[shIndex + 31] - x * shBuffer[shIndex + 32]);

    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, yz = y*z, xz = x*z;
    
    result += 0.5f;
    return max(0.0f, result);
}

// --- Main ---
VSOutput main(uint vID : SV_VertexID, uint instanceID : SV_InstanceID) {
    VSOutput output;

    uint realIndex = splatIndices[instanceID];
    GaussianData s = splats[realIndex];

    float3 centerWorld = s.pos_opacity.xyz;
    float4 centerView4 = mul(ubo.view, float4(centerWorld, 1.0));

    float3 viewDir = normalize(centerWorld - ubo.cameraPos.xyz);
    float3 color = s.color.rgb;
    uint shBase = realIndex * 45;
    float x = viewDir.x; float y = viewDir.y; float z = viewDir.z;

    color.r += SH_C1 * (-y * shBuffer[shBase + 0] + z * shBuffer[shBase + 3] - x * shBuffer[shBase + 6]);
    color.g += SH_C1 * (-y * shBuffer[shBase + 1] + z * shBuffer[shBase + 4] - x * shBuffer[shBase + 7]);
    color.b += SH_C1 * (-y * shBuffer[shBase + 2] + z * shBuffer[shBase + 5] - x * shBuffer[shBase + 8]);

    if (centerView4.z > -0.1) {
        output.pos = float4(0,0,0,0);
        return output;
    }

    float2 viewport = ubo.viewport_focal.xy;
    float2 focal    = ubo.viewport_focal.zw;

    float3x3 cov3d = ComputeCov3D(s.scale.xyz, s.rot);
    float3 cov2d = ComputeCov2D(centerView4.xyz, focal, viewport, cov3d, ubo.view);

    cov2d.x += 0.3f;
    cov2d.z += 0.3f;

    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det <= 1e-6) { output.pos = float4(0,0,0,0); return output; }
    
    float det_inv = 1.0 / det;
    float3 conic = float3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);

    float mid = 0.5 * (cov2d.x + cov2d.z);
    float term = sqrt(max(0.1, mid * mid - det));
    float lambda1 = mid + term;
    float lambda2 = mid - term;
    float radius = 3.0 * sqrt(max(0.1, lambda1));

    radius = min(radius, 1024.0); 

    float2 pixelOffset = quadOffsets[vID] * radius;
    float2 ndcOffset = pixelOffset / (viewport * 0.5);

    output.pos = mul(ubo.proj, centerView4);
    output.pos.xy += ndcOffset * output.pos.w; 

    output.color = float4(color, s.color.a);
    output.uv = pixelOffset; 
    output.conic = conic;

    return output;
}