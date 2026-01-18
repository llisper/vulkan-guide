#include <vk_descriptors.h>

// DescriptorLayoutBuilder
void DescriptorLayoutBuilder::add_binding(uint32_t binding, VkDescriptorType type)
{
    VkDescriptorSetLayoutBinding dsl_binding = {};
    dsl_binding.binding = binding;
    dsl_binding.descriptorType = type;
    dsl_binding.descriptorCount = 1;
    bindings.push_back(dsl_binding);
}

void DescriptorLayoutBuilder::clear()
{
    bindings.clear();
}

VkDescriptorSetLayout DescriptorLayoutBuilder::build(
    VkDevice device,
    VkShaderStageFlags shaderStages,
    void* pNext,
    VkDescriptorSetLayoutCreateFlags flags)
{
    for (auto& b : bindings)
    {
        b.stageFlags |= shaderStages;
    }
    
    VkDescriptorSetLayoutCreateInfo info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    info.pNext = nullptr;
    info.pBindings = bindings.data();
    info.bindingCount = bindings.size();
    info.flags = flags;
    
    VkDescriptorSetLayout set;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &set));
    
    return set;
}

// DescriptorAllocator
void DescriptorAllocator::init_pool(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios)
{
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (auto& p : poolRatios)
    {
        poolSizes.push_back(VkDescriptorPoolSize{
            .type = p.type,
            .descriptorCount = static_cast<uint32_t>(p.ratio * maxSets) 
        });
    }
    
    VkDescriptorPoolCreateInfo info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    info.flags = 0;
    info.pPoolSizes = poolSizes.data();
    info.poolSizeCount = poolSizes.size();
    info.maxSets = maxSets;
    
    vkCreateDescriptorPool(device, &info, nullptr, &pool);
}

void DescriptorAllocator::clear_descriptors(VkDevice device)
{
    vkResetDescriptorPool(device, pool, 0);
}

void DescriptorAllocator::destroy_pool(VkDevice device)
{
    vkDestroyDescriptorPool(device, pool, nullptr);
}

VkDescriptorSet DescriptorAllocator::allocate(VkDevice device, VkDescriptorSetLayout layout)
{
    VkDescriptorSetAllocateInfo alloc_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    alloc_info.pNext = nullptr;
    alloc_info.descriptorPool = pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &layout;
    
    VkDescriptorSet set;
    VK_CHECK(vkAllocateDescriptorSets(device, &alloc_info, &set));
    return set;
}

// DescriptorAllocatorGrowable
void DescriptorAllocatorGrowable::init(VkDevice device, uint32_t initialSets, std::span<PoolSizeRatio> poolRatios)
{
    ratios.clear();
    for (auto& p : poolRatios)
    {
        ratios.push_back(p);
    }
    
    VkDescriptorPool pool = create_pool(device, initialSets, ratios);
    setsPerPool = std::min(4096, (int32_t)(initialSets * 1.5));
    readyPools.push_back(pool);
}

void DescriptorAllocatorGrowable::clear_pools(VkDevice device)
{
    for (auto& p : readyPools)
    {
        vkResetDescriptorPool(device, p, 0);
    }
    for (auto& p : fullPools)
    {
        vkResetDescriptorPool(device, p, 0);
        readyPools.push_back(p);
    }
    fullPools.clear();
}

void DescriptorAllocatorGrowable::destroy_pools(VkDevice device)
{
    for (auto& p : readyPools)
    {
        vkDestroyDescriptorPool(device, p, nullptr);
    }
    readyPools.clear();
    
    for (auto& p : fullPools)
    {
        vkDestroyDescriptorPool(device, p, nullptr);
    }
    fullPools.clear();
}

VkDescriptorPool DescriptorAllocatorGrowable::get_pool(VkDevice device)
{
    if (!readyPools.empty())
    {
        VkDescriptorPool pool = readyPools.back();
        readyPools.pop_back();
        return pool;
    }
    else
    {
        VkDescriptorPool pool = create_pool(device, setsPerPool, ratios);
        setsPerPool = std::min(4096, (int32_t)(setsPerPool * 1.5));
        return pool;
    }
}

VkDescriptorPool DescriptorAllocatorGrowable::create_pool(VkDevice device, uint32_t setCount, std::span<PoolSizeRatio> poolRatios)
{
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (auto& p : poolRatios)
    {
        poolSizes.push_back(VkDescriptorPoolSize{
            .type = p.type,
            .descriptorCount = static_cast<uint32_t>(p.ratio * setCount) 
        });
    }
    
    VkDescriptorPoolCreateInfo info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    info.flags = 0;
    info.maxSets = setCount;
    info.pPoolSizes = poolSizes.data();
    info.poolSizeCount = poolSizes.size();
    
    VkDescriptorPool pool;
    vkCreateDescriptorPool(device, &info, nullptr, &pool);
    return pool;
}

VkDescriptorSet DescriptorAllocatorGrowable::allocate(VkDevice device, VkDescriptorSetLayout layout, void* pNext)
{
    VkDescriptorPool poolToUse = get_pool(device);
    
    VkDescriptorSetAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = pNext,
        .descriptorPool = poolToUse,
        .descriptorSetCount = 1,
        .pSetLayouts = &layout,
    };
    
    VkDescriptorSet set;
    VkResult result = vkAllocateDescriptorSets(device, &allocInfo, &set);
    if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL)
    {
        fullPools.push_back(poolToUse);
        
        poolToUse = get_pool(device);
        allocInfo.descriptorPool = poolToUse;
        VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &set));
    }
    
    readyPools.push_back(poolToUse);
    return set;
}

// DescriptorWriter
void DescriptorWriter::write_image(int32_t binding, VkImageView image, VkSampler sampler, VkImageLayout layout, VkDescriptorType type)
{
    VkDescriptorImageInfo& imageInfo = imageInfos.emplace_back(VkDescriptorImageInfo
        {
            .sampler = sampler,
            .imageView = image,
            .imageLayout = layout,
        });
    
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = VK_NULL_HANDLE,
        .dstBinding = (uint32_t)binding,
        .descriptorCount = 1,
        .descriptorType = type,
        .pImageInfo = &imageInfo,
    };
    writes.push_back(write);
}

void DescriptorWriter::write_buffer(int32_t binding, VkBuffer buffer, uint32_t size, uint32_t offset, VkDescriptorType type)
{
    VkDescriptorBufferInfo bufferInfo = {
        .buffer = buffer,
        .offset = offset,
        .range = size,
    };
    bufferInfos.push_back(bufferInfo);
    
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = VK_NULL_HANDLE,
        .dstBinding = (uint32_t)binding,
        .descriptorCount = 1,
        .descriptorType = type,
        .pBufferInfo = &bufferInfos.back(),
    };
    writes.push_back(write);
}

void DescriptorWriter::clear()
{
    imageInfos.clear();
    bufferInfos.clear();
    writes.clear();
}

void DescriptorWriter::update_set(VkDevice device, VkDescriptorSet set)
{
    for (auto& w : writes)
    {
        w.dstSet = set;
    }
    
    vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
}
