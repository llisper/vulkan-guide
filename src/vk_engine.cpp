//> includes
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include "vk_initializers.h"
#include "vk_images.h"

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#define VMA_IMPLEMENTATION
#include <AsyncInfo.h>

#include "vk_mem_alloc.h"

#include <chrono>
#include <thread>
#include <glm/gtx/transform.hpp>

#include "vk_pipelines.h"
#include "vk_loader.h"

VulkanEngine* loadedEngine = nullptr;

constexpr bool bUseValidationLayers = false;

VulkanEngine& VulkanEngine::Get() { return *loadedEngine; }
void VulkanEngine::init()
{
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    _window = SDL_CreateWindow(
        "Vulkan Engine",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        _windowExtent.width,
        _windowExtent.height,
        window_flags);
    
    init_vulkan();
    init_swapchain();
    init_commands();
    init_sync_structures();
    init_descriptors();
    init_pipelines();
    init_imgui();
    init_default_data();

    // everything went fine
    _isInitialized = true;
}

void VulkanEngine::cleanup()
{
    if (_isInitialized) 
    {
        vkDeviceWaitIdle(_device);
        
        for (uint32_t i = 0; i < FRAME_OVERLAP; ++i)
        {
            vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);
            
            vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
            vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
            vkDestroySemaphore(_device, _frames[i]._swapchainSemaphore, nullptr);
            
            _frames[i]._deletionQueue.flush();
        }
        
        _mainDeletionQueue.flush();
        
        destroy_swapchain();
        
        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkDestroyDevice(_device, nullptr);
        
        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        vkDestroyInstance(_instance, nullptr);
        SDL_DestroyWindow(_window);
    }

    // clear engine pointer
    loadedEngine = nullptr;
}

void VulkanEngine::draw()
{
    VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true, 1000000000));
    
    get_current_frame()._deletionQueue.flush();
    get_current_frame()._frameDescriptors.clear_pools(_device);
    
    VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));
    
    uint32_t swapchainImageIndex;
    VkResult result = vkAcquireNextImageKHR(_device, _swapchain, 1000000000, get_current_frame()._swapchainSemaphore, nullptr, &swapchainImageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        resize_requested = true;
        return;
    }
    
    VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;
    VK_CHECK(vkResetCommandBuffer(cmd, 0));
    
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
    
    _drawExtent.width = std::min(_swapchainExtent.width, _drawImage.imageExtent.width) * renderScale;
    _drawExtent.height = std::min(_swapchainExtent.height, _drawImage.imageExtent.height) * renderScale;
    // transition our main draw image into general layout so we can write into it
    // we will overwrite it all so we dont care about what was the older layout
    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    draw_background(cmd);
    
    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    draw_geometry(cmd);
    
    VkImage swapchain_image = _swapchainImages[swapchainImageIndex];
    VkImageView swapchain_image_view = _swapchainImageViews[swapchainImageIndex];
    // transition the draw image and the swapchain image into their correct transfer layouts
    // ready for copy
    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkutil::transition_image(cmd, swapchain_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    // copy image into swapchain
    vkutil::copy_image(cmd, _drawImage.image, swapchain_image, _drawExtent, _swapchainExtent);
    
    // set swapchain image layout to Attachment Optimal so we can draw it
    vkutil::transition_image(cmd, swapchain_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    // draw imgui to swapchain image
    draw_imgui(cmd, swapchain_image_view);
    
    // set swapchain image layout to Present so we can show it on the screen
    vkutil::transition_image(cmd, swapchain_image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    
    VK_CHECK(vkEndCommandBuffer(cmd));
    
    //prepare the submission to the queue. 
    //we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
    //we will signal the _renderSemaphore, to signal that rendering has finished
    VkCommandBufferSubmitInfo cmdInfo = vkinit::command_buffer_submit_info(cmd);
    
    VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, get_current_frame()._swapchainSemaphore);
    VkSemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, get_current_frame()._renderSemaphore);
    
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdInfo, &signalInfo, &waitInfo);
    
    //submit command buffer to the queue and execute it.
    // _renderFence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, get_current_frame()._renderFence));
    
    //prepare present
    // this will put the image we just rendered to into the visible window.
    // we want to wait on the _renderSemaphore for that, 
    // as its necessary that drawing commands have finished before the image is displayed to the user
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &_swapchain;
    
    presentInfo.pWaitSemaphores = &get_current_frame()._renderSemaphore;
    presentInfo.waitSemaphoreCount = 1;
    
    presentInfo.pImageIndices = &swapchainImageIndex;
    
    result = vkQueuePresentKHR(_graphicsQueue, &presentInfo);
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        resize_requested = true;
        return;
    }
    
    ++_frameNumber;
}

void VulkanEngine::draw_background(VkCommandBuffer cmd)
{
    // float flash = std::abs(std::sin(_frameNumber / 120.0f));
    // VkClearColorValue clearValue = {{0.0f, 0.0f, flash, 1.0f}};
    //
    // VkImageSubresourceRange clearRange = vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
    // vkCmdClearColorImage(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &clearRange);
    
    ComputeEffect& effect = backgroundEffects[currentBackgroundEffect];
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);
    
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _gradientPipelineLayout,
        0, 1, &_drawImageDescriptors,
        0, nullptr);
    
    vkCmdPushConstants(cmd, _gradientPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants), &effect.pushConstants);
    
    vkCmdDispatch(cmd, 
        std::ceil(_drawExtent.width / 16.0),
        std::ceil(_drawExtent.height / 16.0),
        1);
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView)
{
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo renderingInfo = vkinit::rendering_info(_drawExtent, &colorAttachment, nullptr);
    
    vkCmdBeginRendering(cmd, &renderingInfo);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);
}

void VulkanEngine::draw_geometry(VkCommandBuffer cmd)
{
    // allocate uniform buffer for scene data
    AllocatedBuffer gpuSceneDataBuffer = create_buffer(sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    get_current_frame()._deletionQueue.push_function([gpuSceneDataBuffer, this]()
    {
        destroy_buffer(gpuSceneDataBuffer);
    });
    
    // write the buffer with scene data
    GPUSceneData* uniformSceneData = (GPUSceneData*)gpuSceneDataBuffer.allocation->GetMappedData();
    *uniformSceneData = sceneData;
    
    // create global descriptor and use DescriptorWrite to update it with uniform buffer that contains scene data
    VkDescriptorSet globalDescriptor = get_current_frame()._frameDescriptors.allocate(_device, _gpuSceneDataDescriptorLayout);
    DescriptorWriter write;
    write.write_buffer(0, gpuSceneDataBuffer.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    write.update_set(_device, globalDescriptor);
    
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(_drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info(_depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    VkRenderingInfo renderingInfo = vkinit::rendering_info(_drawExtent, &colorAttachment, &depthAttachment);
    vkCmdBeginRendering(cmd, &renderingInfo);
    
    // _meshPipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _meshPipeline); 
    
    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = _drawExtent.width;
    viewport.height = _drawExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    
    VkRect2D scissor = {};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = _drawExtent.width;
    scissor.extent.height = _drawExtent.height;
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    
    glm::mat4 view = glm::translate(glm::vec3(0.0f, 0.0f, -5.0f)); 
    // invert near and far plane so that 1.0 at near plane and 0.0 at far plane for better depth test quality
    glm::mat4 projection = glm::perspective(glm::radians(70.0f), (float)_drawExtent.width / (float)_drawExtent.height, 10000.0f, 0.1f);
    // invert the Y direction on projection matrix so that we are more similar
    // to opengl and gltf axis
    projection[1][1] *= -1.0f;
    
    // draw testMeshes[2]
    GPUDrawPushConstants pushConstants;
    pushConstants.worldMatrix = projection * view;
    pushConstants.vertexBuffer = testMeshes[2]->meshBuffers.vertexBufferAddress;
    vkCmdPushConstants(cmd, _meshPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdBindIndexBuffer(cmd, testMeshes[2]->meshBuffers.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    
    vkCmdDrawIndexed(cmd, testMeshes[2]->surfaces[0].count, 1, testMeshes[2]->surfaces[0].startIndex, 0, 0);
    
    vkCmdEndRendering(cmd);
}

AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
    VkBufferCreateInfo bufferInfo = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.pNext = nullptr;
    bufferInfo.size = allocSize;
    bufferInfo.usage = usage;
    
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = memoryUsage;
    allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    
    AllocatedBuffer newBuffer;
    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &allocInfo, &newBuffer.buffer, &newBuffer.allocation, &newBuffer.info));
    
    return newBuffer;
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer& buffer)
{
    vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
}

GPUMeshBuffers VulkanEngine::upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices)
{
    GPUMeshBuffers newSurface;
    
    size_t indexSize = indices.size() * sizeof(uint32_t);
    size_t vertexSize = vertices.size() * sizeof(Vertex);
    
    newSurface.vertexBuffer = create_buffer(
        vertexSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    
    VkBufferDeviceAddressInfo addressInfo = { .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
    addressInfo.pNext = nullptr;
    addressInfo.buffer = newSurface.vertexBuffer.buffer;
    newSurface.vertexBufferAddress = vkGetBufferDeviceAddress(_device, &addressInfo);
    
    newSurface.indexBuffer = create_buffer(
        indexSize,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    
    AllocatedBuffer stagingBuffer = create_buffer(indexSize + vertexSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    void* data = stagingBuffer.allocation->GetMappedData();
    
    memcpy(data, vertices.data(), vertexSize);
    memcpy((char*)data + vertexSize, indices.data(), indexSize);
    
    immediate_submit([&](VkCommandBuffer cmdBuffer)
    {
        VkBufferCopy copy = {};
        copy.srcOffset = 0;
        copy.dstOffset = 0;
        copy.size = vertexSize;
        vkCmdCopyBuffer(cmdBuffer, stagingBuffer.buffer, newSurface.vertexBuffer.buffer, 1, &copy);
        
        copy.srcOffset = vertexSize;
        copy.dstOffset = 0;
        copy.size = indexSize;
        vkCmdCopyBuffer(cmdBuffer, stagingBuffer.buffer, newSurface.indexBuffer.buffer, 1, &copy);
    });
    
    destroy_buffer(stagingBuffer);
    
    return newSurface;
}

AllocatedImage VulkanEngine::create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usageFlags, bool mipmapped)
{
    AllocatedImage newImage;
    newImage.imageFormat = format;
    newImage.imageExtent = size;
    
    VkImageCreateInfo imageCreateInfo = vkinit::image_create_info(format, usageFlags, size);
    if (mipmapped)
    {
        imageCreateInfo.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1;
    }
    
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    VK_CHECK(vmaCreateImage(_allocator, &imageCreateInfo, &allocInfo, &newImage.image, &newImage.allocation, nullptr));
    
    VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
    if (format == VK_FORMAT_D32_SFLOAT)
    {
        aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    VkImageViewCreateInfo imageViewCreateInfo = vkinit::imageview_create_info(format, newImage.image, aspectFlags);
    imageViewCreateInfo.subresourceRange.levelCount = imageCreateInfo.mipLevels;
    VK_CHECK(vkCreateImageView(_device, &imageViewCreateInfo, nullptr, &newImage.imageView));
    
    return newImage;
}

AllocatedImage VulkanEngine::create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usageFlags, bool mipmapped)
{
    size_t dataSize = size.width * size.height * size.depth * 4;
    AllocatedBuffer uploadBuffer = create_buffer(dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    memcpy(uploadBuffer.info.pMappedData, data, dataSize);
    
    AllocatedImage newImage = create_image(size, format, usageFlags | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, mipmapped);
    immediate_submit([&](VkCommandBuffer cmd)
    {
        vkutil::transition_image(cmd, newImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        
        VkBufferImageCopy copyRegion = {};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = size;
        
        vkCmdCopyBufferToImage(cmd, uploadBuffer.buffer, newImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
        
        vkutil::transition_image(cmd, newImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    
    destroy_buffer(uploadBuffer);
    
    return newImage;
}

void VulkanEngine::destroy_image(const AllocatedImage& image)
{
    vkDestroyImageView(_device, image.imageView, nullptr);
    vmaDestroyImage(_allocator, image.image, image.allocation);
}

void VulkanEngine::run()
{
    SDL_Event e;
    bool bQuit = false;

    // main loop
    while (!bQuit) {
        // Handle events on queue
        while (SDL_PollEvent(&e) != 0) {
            // close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT)
                bQuit = true;

            if (e.type == SDL_WINDOWEVENT) {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
                    stop_rendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
                    stop_rendering = false;
                }
            }
            
            ImGui_ImplSDL2_ProcessEvent(&e);
        }

        // do not draw if we are minimized
        if (stop_rendering) {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        if (resize_requested)
        {
            resize_swapchain();
        }
        
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        if (ImGui::Begin("background")) {
            ImGui::SliderFloat("Render Scale",&renderScale, 0.3f, 1.f);
			
            ComputeEffect& selected = backgroundEffects[currentBackgroundEffect];
		
            ImGui::Text("Selected effect: ", selected.name.c_str());
		
            ImGui::SliderInt("Effect Index", &currentBackgroundEffect,0, backgroundEffects.size() - 1);
		          
            ImGui::InputFloat4("data1",(float*)& selected.pushConstants.data1);
            ImGui::InputFloat4("data2",(float*)& selected.pushConstants.data2);
            ImGui::InputFloat4("data3",(float*)& selected.pushConstants.data3);
            ImGui::InputFloat4("data4",(float*)& selected.pushConstants.data4);
        }
        ImGui::End();

        ImGui::Render();

        draw();
    }
}

// void VulkanEngine::init_triangle_pipeline()
// {
//     VkShaderModule triangleFragShader;
//     if (!vkutil::load_shader_module("../../shaders/colored_triangle.frag.spv", _device, &triangleFragShader)) {
//         fmt::print("Error when building the triangle fragment shader module");
//     }
//     else {
//         fmt::print("Triangle fragment shader succesfully loaded");
//     }
//
//     VkShaderModule triangleVertexShader;
//     if (!vkutil::load_shader_module("../../shaders/colored_triangle.vert.spv", _device, &triangleVertexShader)) {
//         fmt::print("Error when building the triangle vertex shader module");
//     }
//     else {
//         fmt::print("Triangle vertex shader succesfully loaded");
//     }
//     
//     VkPipelineLayoutCreateInfo pipelineCreateInfo = vkinit::pipeline_layout_create_info();
//     VK_CHECK(vkCreatePipelineLayout(_device, &pipelineCreateInfo, nullptr, &_trianglePipelineLayout));
//     
//     PipelineBuilder pipelineBuilder;
//     pipelineBuilder._pipelineLayout = _trianglePipelineLayout;
//     pipelineBuilder.set_shaders(triangleVertexShader, triangleFragShader);
//     pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
//     pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
//     pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
//     pipelineBuilder.set_multisampling_none();
//     pipelineBuilder.disable_blending();
//     pipelineBuilder.disable_depthtest();
//     
//     pipelineBuilder.set_color_attachment_format(_drawImage.imageFormat);
//     pipelineBuilder.set_depth_format(VK_FORMAT_UNDEFINED);
//     
//     _trianglePipeline = pipelineBuilder.build_pipeline(_device);
//     
//     vkDestroyShaderModule(_device, triangleVertexShader, nullptr);
//     vkDestroyShaderModule(_device, triangleFragShader, nullptr);
//     
//     _mainDeletionQueue.push_function([&]()
//     {
//         vkDestroyPipelineLayout(_device, _trianglePipelineLayout, nullptr);
//         vkDestroyPipeline(_device, _trianglePipeline, nullptr);
//         
//     });
// }

void VulkanEngine::init_mesh_pipeline()
{
    VkShaderModule triangleFragShader;
    if (!vkutil::load_shader_module("../../shaders/colored_triangle.frag.spv", _device, &triangleFragShader)) {
        fmt::print("Error when building the triangle fragment shader module");
    }
    else {
        fmt::print("Triangle fragment shader succesfully loaded");
    }

    VkShaderModule triangleVertexShader;
    if (!vkutil::load_shader_module("../../shaders/colored_triangle_mesh.vert.spv", _device, &triangleVertexShader)) {
        fmt::print("Error when building the triangle vertex shader module");
    }
    else {
        fmt::print("Triangle vertex shader succesfully loaded");
    }
    
    VkPushConstantRange pushConstantRange;
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(GPUDrawPushConstants);
    
    VkPipelineLayoutCreateInfo pipelineCreateInfo = vkinit::pipeline_layout_create_info();
    pipelineCreateInfo.pushConstantRangeCount = 1;
    pipelineCreateInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK(vkCreatePipelineLayout(_device, &pipelineCreateInfo, nullptr, &_meshPipelineLayout));
    
    PipelineBuilder pipelineBuilder;
    pipelineBuilder._pipelineLayout = _meshPipelineLayout;
    pipelineBuilder.set_shaders(triangleVertexShader, triangleFragShader);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    // pipelineBuilder.disable_blending();
    pipelineBuilder.enable_blending_additive();
    // pipelineBuilder.enable_blending_alphablend();
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
    
    pipelineBuilder.set_color_attachment_format(_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(_depthImage.imageFormat);
    
    _meshPipeline = pipelineBuilder.build_pipeline(_device);
    
    vkDestroyShaderModule(_device, triangleVertexShader, nullptr);
    vkDestroyShaderModule(_device, triangleFragShader, nullptr);
    
    _mainDeletionQueue.push_function([&]()
    {
        vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
        vkDestroyPipeline(_device, _meshPipeline, nullptr);
        
    });
}

void VulkanEngine::init_default_data()
{
    // std::array<Vertex,4> rect_vertices;
    //
    // rect_vertices[0].position = {0.5,-0.5, 0};
    // rect_vertices[1].position = {0.5,0.5, 0};
    // rect_vertices[2].position = {-0.5,-0.5, 0};
    // rect_vertices[3].position = {-0.5,0.5, 0};
    //
    // rect_vertices[0].color = {0,0, 0,1};
    // rect_vertices[1].color = { 0.5,0.5,0.5 ,1};
    // rect_vertices[2].color = { 1,0, 0,1 };
    // rect_vertices[3].color = { 0,1, 0,1 };
    //
    // std::array<uint32_t,6> rect_indices;
    //
    // rect_indices[0] = 0;
    // rect_indices[1] = 1;
    // rect_indices[2] = 2;
    //
    // rect_indices[3] = 2;
    // rect_indices[4] = 1;
    // rect_indices[5] = 3;
    //
    // rectangle = upload_mesh(rect_indices, rect_vertices);
    //
    // _mainDeletionQueue.push_function([&]()
    // {
    //     destroy_buffer(rectangle.vertexBuffer);
    //     destroy_buffer(rectangle.indexBuffer);
    // });
    
    // test
    testMeshes = load_gltf_meshes(this, R"(..\..\assets\basicmesh.glb)").value();
    _mainDeletionQueue.push_function([&]()
    {
        for (auto ptr : testMeshes)
        {
            destroy_buffer(ptr->meshBuffers.vertexBuffer);
            destroy_buffer(ptr->meshBuffers.indexBuffer);
        }
    });
    
    uint32_t white = glm::packUnorm4x8(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
    _whiteImage = create_image(&white, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);
    
    uint32_t black = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1.0f));
    _blackImage = create_image(&black, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);
    
    uint32_t grey = glm::packUnorm4x8(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    _greyImage = create_image(&grey, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);
    
    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 *16 > pixels; //for 16x16 checkerboard texture
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            pixels[y*16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }
    _errorCheckerboardImage = create_image(pixels.data(), VkExtent3D{16, 16, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);
    
    VkSamplerCreateInfo samplerCreateInfo = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
    vkCreateSampler(_device, &samplerCreateInfo, nullptr, &_defaultSamplerLinear);
    
    samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
    vkCreateSampler(_device, &samplerCreateInfo, nullptr, &_defaultSamplerNearest);
    
    _mainDeletionQueue.push_function([&]()
    {
        vkDestroySampler(_device, _defaultSamplerLinear, nullptr);
        vkDestroySampler(_device, _defaultSamplerNearest, nullptr);
        
        destroy_image(_errorCheckerboardImage);
        destroy_image(_greyImage);
        destroy_image(_blackImage);
        destroy_image(_whiteImage);
    });
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer)>&& function)
{
    VK_CHECK(vkResetFences(_device, 1, &_immFence));
    VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));
    
    VkCommandBuffer cmd = _immCommandBuffer;
    
    VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
    
    function(cmd);
    
    VK_CHECK(vkEndCommandBuffer(cmd));
    
    VkCommandBufferSubmitInfo cmdInfo = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdInfo, nullptr, nullptr);
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, _immFence));
    VK_CHECK(vkWaitForFences(_device, 1, &_immFence, true, UINT64_MAX));
}

void VulkanEngine::init_vulkan()
{
    vkb::InstanceBuilder builder;
    auto inst = builder.set_app_name("Learning Vulkan Application") 
        .request_validation_layers(bUseValidationLayers)
        .use_default_debug_messenger()
        .require_api_version(1, 3, 0)
        .build();
    
    vkb::Instance vkb_inst = inst.value();
    _instance = vkb_inst.instance;
    _debug_messenger = vkb_inst.debug_messenger;
    
    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);
    
    // NOTE: remember to use {} to zero initialize the struct, or else it crashes!
    VkPhysicalDeviceVulkan13Features features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features.dynamicRendering = true;
    features.synchronization2 = true;
    
    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.bufferDeviceAddress = true;
    features12.descriptorIndexing = true;
    
    vkb::PhysicalDeviceSelector selector{vkb_inst};
    vkb::PhysicalDevice physical_device = selector
        .set_minimum_version(1, 3)
        .set_required_features_13(features)
        .set_required_features_12(features12)
        .set_surface(_surface)
        .select()
        .value();
    
    vkb::DeviceBuilder device_builder{physical_device};
    vkb::Device vkb_device = device_builder.build().value();
    _device = vkb_device.device;
    _chosenGPU = vkb_device.physical_device;
    
    _graphicsQueue = vkb_device.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkb_device.get_queue_index(vkb::QueueType::graphics).value();
    
    VmaAllocatorCreateInfo allocatorCreateInfo{};
    allocatorCreateInfo.physicalDevice = _chosenGPU;
    allocatorCreateInfo.device = _device;
    allocatorCreateInfo.instance = _instance;
    allocatorCreateInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorCreateInfo, &_allocator);
    _mainDeletionQueue.push_function([this]()
    {
        vmaDestroyAllocator(_allocator);
    });
    
}

void VulkanEngine::init_swapchain()
{
    create_swapchain(_windowExtent.width, _windowExtent.height);
    
    VkExtent3D drawImageExtent = {
        _windowExtent.width,
        _windowExtent.height, 
        1
    };
    
    // create _drawImage
    _drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    _drawImage.imageExtent = drawImageExtent;
    
    VkImageUsageFlags drawImageFlags = 
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | 
        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    
    VkImageCreateInfo rimg_info = vkinit::image_create_info(_drawImage.imageFormat, drawImageFlags, drawImageExtent);
    
    VmaAllocationCreateInfo rimg_allocinfo{};
    rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    rimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    vmaCreateImage(_allocator, &rimg_info, &rimg_allocinfo, &_drawImage.image, &_drawImage.allocation, nullptr);
    
    VkImageViewCreateInfo rview_info = vkinit::imageview_create_info(_drawImage.imageFormat, _drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);
    VK_CHECK(vkCreateImageView(_device, &rview_info, nullptr, &_drawImage.imageView));
    
    // create _depthImage
    _depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
    _depthImage.imageExtent = drawImageExtent;
    
    VkImageUsageFlags depthImageUsage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthImage.imageFormat, depthImageUsage, drawImageExtent);
    
    vmaCreateImage(_allocator, &dimg_info, &rimg_allocinfo, &_depthImage.image, &_depthImage.allocation, nullptr);
    
    VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_depthImage.imageFormat, _depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);
    VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImage.imageView));
    
    _mainDeletionQueue.push_function([this]()
    {
        vkDestroyImageView(_device, _drawImage.imageView, nullptr);
        vmaDestroyImage(_allocator, _drawImage.image, _drawImage.allocation);
        
        vkDestroyImageView(_device, _depthImage.imageView, nullptr);
        vmaDestroyImage(_allocator, _depthImage.image, _depthImage.allocation);
    });
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height)
{
    vkb::SwapchainBuilder swapchainBuilder{_chosenGPU, _device, _surface};  
    vkb::Swapchain swapchain = swapchainBuilder
        .set_desired_format(VkSurfaceFormatKHR{.format = _swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .build()
        .value();
    
    _swapchainExtent = swapchain.extent;
    _swapchain = swapchain.swapchain;
    _swapchainImages = swapchain.get_images().value();
    _swapchainImageViews = swapchain.get_image_views().value();
}

void VulkanEngine::destroy_swapchain()
{
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);
    for (const VkImageView& view : _swapchainImageViews)
    {
        vkDestroyImageView(_device, view, nullptr);
    }
}

void VulkanEngine::resize_swapchain()
{
    vkDeviceWaitIdle(_device);
    destroy_swapchain();
    
    int w, h;
    SDL_GetWindowSize(_window, &w, &h);
    _windowExtent.width = w;
    _windowExtent.height = h;
    
    create_swapchain(w, h);
    
    resize_requested = false;
}

void VulkanEngine::init_commands()
{
    VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(
        _graphicsQueueFamily,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    
    for (auto& _frame : _frames)
    {
        VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frame._commandPool));
        VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_frame._commandPool, 1);
        VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frame._mainCommandBuffer));
    }
    
    {
        VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_immCommandPool));
        VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_immCommandPool, 1);
        VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_immCommandBuffer));
        
        _mainDeletionQueue.push_function([this]()
        {
            vkDestroyCommandPool(_device, _immCommandPool, nullptr);
        });
    }
}

void VulkanEngine::init_sync_structures()
{
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();
    
    for (auto& _frame : _frames)
    {
        VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_frame._renderFence));
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frame._renderSemaphore));
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frame._swapchainSemaphore));
    }
    
    {
        VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_immFence));
        _mainDeletionQueue.push_function([this]()
        {
            vkDestroyFence(_device, _immFence, nullptr);
        });
    }
}

void VulkanEngine::init_descriptors()
{
    std::vector<DescriptorAllocator::PoolSizeRatio> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1.0f },
    };
    globalDescriptorAllocator.init_pool(_device, 10, poolSizes);
    
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        _drawImageDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);
    }
    
    _drawImageDescriptors = globalDescriptorAllocator.allocate(_device, _drawImageDescriptorLayout);
    
    DescriptorWriter writer;
    writer.write_image(0, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    writer.update_set(_device, _drawImageDescriptors);
    
    _mainDeletionQueue.push_function([&]()
    {
        globalDescriptorAllocator.destroy_pool(_device);
        vkDestroyDescriptorSetLayout(_device, _drawImageDescriptorLayout, nullptr);
    });
    
    for (auto& _frame : _frames)
    {
        std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> poolSizes = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 },
        };

        _frame._frameDescriptors = DescriptorAllocatorGrowable{};
        _frame._frameDescriptors.init(_device, 1000, poolSizes);
    }
    
    _mainDeletionQueue.push_function([&]()
    {
        for (auto& _frame : _frames)
        {
            _frame._frameDescriptors.destroy_pools(_device);
        }
    });
    
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        _gpuSceneDataDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT);
    }
}

void VulkanEngine::init_pipelines()
{
    init_background_pipelines();
    // init_triangle_pipeline(); 
    init_mesh_pipeline();
}

void VulkanEngine::init_background_pipelines()
{
    VkPipelineLayoutCreateInfo computeLayoutCreateInfo = {};
    computeLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    computeLayoutCreateInfo.pNext = nullptr;
    computeLayoutCreateInfo.pSetLayouts = &_drawImageDescriptorLayout;
    computeLayoutCreateInfo.setLayoutCount = 1;
    
    VkPushConstantRange pushConstant = {};
    pushConstant.offset = 0;
    pushConstant.size = sizeof(ComputePushConstants);
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    computeLayoutCreateInfo.pPushConstantRanges = &pushConstant;
    computeLayoutCreateInfo.pushConstantRangeCount = 1;
    
    VK_CHECK(vkCreatePipelineLayout(_device, &computeLayoutCreateInfo, nullptr, &_gradientPipelineLayout));
    
    VkShaderModule gradientShader;
    if (!vkutil::load_shader_module("../../shaders/gradient_color.comp.spv", _device, &gradientShader))
    {
        fmt::print("Error when building the compute shader \n");
    }
    
    VkShaderModule skyShader;
    if (!vkutil::load_shader_module("../../shaders/sky.comp.spv", _device, &skyShader))
    {
        fmt::print("Error when building the compute shader \n");
    }
    
    VkPipelineShaderStageCreateInfo stageInfo = {};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.pNext = nullptr;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = gradientShader;
    stageInfo.pName = "main";
    
    VkComputePipelineCreateInfo computePipelineCreateInfo = {};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.pNext = nullptr;
    computePipelineCreateInfo.stage = stageInfo;
    computePipelineCreateInfo.layout = _gradientPipelineLayout;
    
    ComputeEffect gradientEffect = {};
    gradientEffect.name = "gradient";
    gradientEffect.layout = _gradientPipelineLayout;
    gradientEffect.pushConstants.data1 = glm::vec4(1.0, 0.0, 0.0, 1.0);
    gradientEffect.pushConstants.data2 = glm::vec4(0.0, 0.0, 1.0, 1.0);
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &gradientEffect.pipeline));
    backgroundEffects.push_back(gradientEffect);
    
    
    computePipelineCreateInfo.stage.module = skyShader;
    
    ComputeEffect skyEffect = {};
    skyEffect.name = "sky";
    skyEffect.layout = _gradientPipelineLayout;
    skyEffect.pushConstants.data1 = glm::vec4(0.1, 0.2, 0.4, 0.97);
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &skyEffect.pipeline));
    backgroundEffects.push_back(skyEffect);
    
    vkDestroyShaderModule(_device, gradientShader, nullptr);
    vkDestroyShaderModule(_device, skyShader, nullptr);
    _mainDeletionQueue.push_function([this, gradientEffect, skyEffect]()
    {
        vkDestroyPipelineLayout(_device, _gradientPipelineLayout, nullptr);
        vkDestroyPipeline(_device, skyEffect.pipeline, nullptr);
        vkDestroyPipeline(_device, gradientEffect.pipeline, nullptr);
    });
}

void VulkanEngine::init_imgui()
{
    // 1: create descriptor pool for IMGUI
	//  the size of the pool is very oversize, but it's copied from imgui demo
	//  itself.
	VkDescriptorPoolSize pool_sizes[] = { { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 } };

	VkDescriptorPoolCreateInfo pool_info = {};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	pool_info.maxSets = 1000;
	pool_info.poolSizeCount = (uint32_t)std::size(pool_sizes);
	pool_info.pPoolSizes = pool_sizes;

	VkDescriptorPool imguiPool;
	VK_CHECK(vkCreateDescriptorPool(_device, &pool_info, nullptr, &imguiPool));

	// 2: initialize imgui library

	// this initializes the core structures of imgui
	ImGui::CreateContext();

	// this initializes imgui for SDL
	ImGui_ImplSDL2_InitForVulkan(_window);

	// this initializes imgui for Vulkan
	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance = _instance;
	init_info.PhysicalDevice = _chosenGPU;
	init_info.Device = _device;
	init_info.Queue = _graphicsQueue;
	init_info.DescriptorPool = imguiPool;
	init_info.MinImageCount = 3;
	init_info.ImageCount = 3;
	init_info.UseDynamicRendering = true;

	//dynamic rendering parameters for imgui to use
	init_info.PipelineRenderingCreateInfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
	init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
	init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &_swapchainImageFormat;
	

	init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

	ImGui_ImplVulkan_Init(&init_info);

	ImGui_ImplVulkan_CreateFontsTexture();

	// add the destroy the imgui created structures
	_mainDeletionQueue.push_function([=]() {
		ImGui_ImplVulkan_Shutdown();
		vkDestroyDescriptorPool(_device, imguiPool, nullptr);
	});
}
