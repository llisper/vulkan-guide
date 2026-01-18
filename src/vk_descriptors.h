#pragma once

#include <vk_types.h>

struct DescriptorLayoutBuilder
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    
    void add_binding(uint32_t binding, VkDescriptorType type);
    void clear();
    
    VkDescriptorSetLayout build(
        VkDevice device,
        VkShaderStageFlags shaderStages,
        void* pNext = nullptr,
        VkDescriptorSetLayoutCreateFlags flags = 0); 
    
};


struct DescriptorAllocator
{
    struct PoolSizeRatio
    {
        VkDescriptorType type;
        float ratio;
    };
    
    VkDescriptorPool pool;    
    
    void init_pool(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios);
    void clear_descriptors(VkDevice device);
    void destroy_pool(VkDevice device);
    
    VkDescriptorSet allocate(VkDevice device, VkDescriptorSetLayout layout);
};


struct DescriptorAllocatorGrowable
{
    struct PoolSizeRatio
    {
        VkDescriptorType type;
        float ratio;
    };
    
    void init(VkDevice device, uint32_t initialSets, std::span<PoolSizeRatio> poolRatios);
    void clear_pools(VkDevice device);
    void destroy_pools(VkDevice device);
    
    VkDescriptorSet allocate(VkDevice device, VkDescriptorSetLayout layout, void* pNext=nullptr);
    
private:
    VkDescriptorPool get_pool(VkDevice device);
    VkDescriptorPool create_pool(VkDevice device, uint32_t setCount, std::span<PoolSizeRatio> poolRatios);
    
    std::vector<PoolSizeRatio> ratios;
    std::vector<VkDescriptorPool> readyPools;
    std::vector<VkDescriptorPool> fullPools;
    uint32_t setsPerPool;
};


struct DescriptorWriter
{
    std::deque<VkDescriptorImageInfo> imageInfos;
    std::deque<VkDescriptorBufferInfo> bufferInfos;
    std::vector<VkWriteDescriptorSet> writes;
    
    void write_image(int32_t binding, VkImageView image, VkSampler sampler, VkImageLayout layout, VkDescriptorType type);
    void write_buffer(int32_t binding, VkBuffer buffer, uint32_t size, uint32_t offset, VkDescriptorType type);
    
    void clear();
    void update_set(VkDevice device, VkDescriptorSet set);
};
