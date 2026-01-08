// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include "vk_types.h"
#include "vk_descriptors.h"
#include "VkBootstrap.h"

struct MeshAsset;

struct FrameData
{
	VkCommandPool _commandPool{ nullptr };
	VkCommandBuffer _mainCommandBuffer{ nullptr };
	
	VkSemaphore _swapchainSemaphore;
	VkSemaphore _renderSemaphore;
	VkFence _renderFence;
	
	DeletionQueue _deletionQueue;
};

constexpr uint32_t FRAME_OVERLAP = 2;

struct ComputePushConstants
{
	glm::vec4 data1;
	glm::vec4 data2;
	glm::vec4 data3;
	glm::vec4 data4;
};

struct ComputeEffect
{
	std::string name;
	
	VkPipeline pipeline{ nullptr };
	VkPipelineLayout layout{ nullptr };
	
	ComputePushConstants pushConstants;
};

class VulkanEngine 
{
public:
	bool _isInitialized{ false };
	int _frameNumber {0};
	bool stop_rendering{ false };
	bool resize_requested{ false };
	VkExtent2D _windowExtent{ 1700 , 900 };

	struct SDL_Window* _window{ nullptr };

	static VulkanEngine& Get();

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	//run main loop
	void run();
	
	FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; }
	
public:
	DescriptorAllocator	globalDescriptorAllocator;
	VkDescriptorSet _drawImageDescriptors{ nullptr };
	VkDescriptorSetLayout _drawImageDescriptorLayout{ nullptr };
	
public:
	VkPipeline _gradientPipeline{ nullptr };
	VkPipelineLayout _gradientPipelineLayout{ nullptr };
	
	std::vector<ComputeEffect> backgroundEffects;
	int currentBackgroundEffect{0};
	
	// VkPipeline _trianglePipeline{ nullptr };
	// VkPipelineLayout _trianglePipelineLayout{ nullptr };
	
	// void init_triangle_pipeline();
	
	VkPipeline _meshPipeline{ nullptr };
	VkPipelineLayout _meshPipelineLayout{ nullptr };
	
	// GPUMeshBuffers rectangle;
	
	std::vector<std::shared_ptr<MeshAsset>> testMeshes;
	
	void init_mesh_pipeline();
	void init_default_data();
	
	GPUMeshBuffers upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices);
public:
	VkFence _immFence{ nullptr };
	VkCommandBuffer _immCommandBuffer{ nullptr };
	VkCommandPool _immCommandPool{ nullptr };
	
	void immediate_submit(std::function<void (VkCommandBuffer)>&& function);
	
private:
	// vulkan library handle
	VkInstance _instance{nullptr};
	// vulkan debug output handle
	VkDebugUtilsMessengerEXT _debug_messenger{ nullptr };
	// GPU device
	VkPhysicalDevice _chosenGPU{ nullptr };
	// vulkan device for command
	VkDevice _device{ nullptr };
	// vulkan window surface
	VkSurfaceKHR _surface{ nullptr };
	
	VkSwapchainKHR _swapchain{ nullptr };
	VkFormat _swapchainImageFormat{VK_FORMAT_B8G8R8A8_UNORM};
	
	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;
	VkExtent2D _swapchainExtent;
	
	FrameData _frames[FRAME_OVERLAP];
	VkQueue _graphicsQueue{ nullptr };
	uint32_t _graphicsQueueFamily;
	
	DeletionQueue _mainDeletionQueue;
	
	VmaAllocator _allocator;
	VkExtent2D _drawExtent;
	AllocatedImage _drawImage;
	AllocatedImage _depthImage;
	float renderScale{ 1.0f };
	
	void init_vulkan();
	
	void init_swapchain();
	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();
	void resize_swapchain();
	
	void init_commands();
	void init_sync_structures();
	
	void init_descriptors();
	
	void init_pipelines();
	void init_background_pipelines();
	
	void init_imgui();
	
	void draw_background(VkCommandBuffer cmd);
	void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
	void draw_geometry(VkCommandBuffer cmd);	
	
	AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	void destroy_buffer(const AllocatedBuffer& buffer);
};
