
#include "stb_image.h"
#include <iostream>
#include <vk_loader.h>

#include "vk_engine.h"
#include "vk_initializers.h"
#include "vk_types.h"
#include <glm/gtx/quaternion.hpp>

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/parser.hpp>
#include <fastgltf/tools.hpp>

std::optional<std::vector<std::shared_ptr<MeshAsset>>> load_gltf_meshes(
    VulkanEngine* engine,
    std::filesystem::path filePath)
{
    std::cout << "loading GLTF: " << filePath << std::endl;
    
    fastgltf::GltfDataBuffer data;
    data.loadFromFile(filePath);
    
    constexpr auto gltfOptions = fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers;
    
    fastgltf::Asset gltf{};
    fastgltf::Parser parser{};
    
    auto loaded = parser.loadBinaryGLTF(&data, filePath.parent_path(), gltfOptions);
    if (loaded)
    {
        gltf = std::move(loaded.get());
    }
    else
    {
        fmt::print("Failed to load glTF: {} \n", fastgltf::to_underlying(loaded.error()));
        return {};
    }
    
    std::vector<std::shared_ptr<MeshAsset>> meshes;
    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;
    for (fastgltf::Mesh& mesh : gltf.meshes)
    {
        MeshAsset newMesh;
        newMesh.name = mesh.name;
        
        indices.clear();
        vertices.clear();
        
        for (auto& p : mesh.primitives)
        {
            GeoSurface newSurface;
            newSurface.startIndex = indices.size();
            newSurface.count = gltf.accessors[p.indicesAccessor.value()].count;
            
            size_t initial_index = indices.size();
            
            {
                fastgltf::Accessor& indexAccessor = gltf.accessors[p.indicesAccessor.value()];
                indices.reserve(indices.size() + indexAccessor.count);
                fastgltf::iterateAccessor<uint32_t>(
                    gltf, indexAccessor,
                    [&](uint32_t idx) { indices.push_back(idx + initial_index); });
            }
            
            {
                fastgltf::Accessor& posAccessor = gltf.accessors[p.findAttribute("POSITION")->second];
                vertices.resize(vertices.size() + posAccessor.count);
                
                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor,
                    [&](const glm::vec3& position, uint32_t idx)
                    {
                        Vertex vertex = {};
                        vertex.position = position;
                        vertex.normal = {1.0f, 0.0f, 0.0f};
                        vertex.color = glm::vec4(1.0f);
                        vertices[initial_index + idx] = vertex;
                    });
            }
            
            auto normals = p.findAttribute("NORMAL");
            if (normals != p.attributes.end())
            {
                fastgltf::Accessor& normalAccessor = gltf.accessors[normals->second];
                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, normalAccessor,
                    [&](const glm::vec3& normal, uint32_t idx)
                    {
                        vertices[initial_index + idx].normal = normal;
                    });
            }
            
            auto uv = p.findAttribute("TEXCOORD_0");
            if (uv != p.attributes.end())
            {
                fastgltf::Accessor& uvAccessor = gltf.accessors[uv->second];
                fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, uvAccessor,
                    [&](const glm::vec2& v, uint32_t idx)
                    {
                        vertices[initial_index + idx].uv_x = v.x;
                        vertices[initial_index + idx].uv_y = v.y;
                    });
            }
            
            auto colors = p.findAttribute("COLOR_0");
            if (colors != p.attributes.end())
            {
                fastgltf::Accessor& colorAccessor = gltf.accessors[colors->second];
                fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, colorAccessor,
                    [&](const glm::vec4& color, uint32_t idx)
                    {
                        vertices[initial_index + idx].color = color;
                    });
            }
            
            newMesh.surfaces.push_back(newSurface);
        }
        
        constexpr bool bOverrideColors = true;
        if (bOverrideColors)
        {
            for (Vertex& vertex : vertices)
            {
                vertex.color = glm::vec4(vertex.normal, 1.0f);
            }
        }
        
        newMesh.meshBuffers = engine->upload_mesh(indices, vertices);
        meshes.emplace_back(std::make_shared<MeshAsset>(std::move(newMesh)));
    }
    
    return meshes;
}
