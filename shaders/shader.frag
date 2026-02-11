struct PSInput {
    [[vk::location(0)]] float4 color : COLOR;
    [[vk::location(1)]] float2 uv : TEXCOORD0;
    [[vk::location(2)]] float3 conic : TEXCOORD1;
};

float4 main(PSInput input) : SV_TARGET {
    float2 d = input.uv;
    
    float power = -0.5 * (d.x * d.x * input.conic.x + 
                      d.y * d.y * input.conic.z + 
                      2.0 * d.x * d.y * input.conic.y);

    if (power > 0.0) discard;
    
    float alpha = exp(power);
    
    float finalAlpha = input.color.a * alpha;
    
    if (finalAlpha < 1.0/255.0) discard;

    return float4(input.color.rgb, finalAlpha);
}