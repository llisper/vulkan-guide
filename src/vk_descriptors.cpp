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
