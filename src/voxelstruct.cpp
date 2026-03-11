#include "voxelstruct.hpp"
#include <open3d/Open3D.h>
#include <open3d/io/PointCloudIO.h>
#include <open3d/geometry/PointCloud.h>
#include <octomap/Pointcloud.h>
#include <octomap/ColorOcTree.h>
#include <octomap/octomap.h>
#include <vector>
#include <string>
#include <unordered_set>
#include <cstdint>
#include <iostream>

voxelstruct::voxelstruct(double resolution) 
{
    this->resolution = resolution;
    this->tree = new octomap::ColorOcTree(resolution);
}

// for further optimization
void voxelstruct::insertPointCloud(
    open3d::geometry::PointCloud* pcd,
    Eigen::Vector3d camera) 
{
    if(!pcd||pcd->points_.empty()) return;
    
    if(this->pcd) {
        delete this->pcd;
        this->pcd = nullptr;
    }
    this->pcd = pcd;

    for (size_t i = 0; i < pcd->points_.size(); ++i) {
        const auto& p = pcd->points_[i];
        const auto& c = pcd->colors_[i];

        octomap::point3d origin(camera(0), camera(1), camera(2));
        octomap::point3d endpoint(p(0), p(1), p(2));

        // Insert ray (marks free voxels + marks endpoint occupied)
        tree->insertRay(origin, endpoint);

        // Now add color to the occupied endpoint
        unsigned char r = static_cast<unsigned char>(c(0) * 255);
        unsigned char g = static_cast<unsigned char>(c(1) * 255);
        unsigned char b = static_cast<unsigned char>(c(2) * 255);

        tree->integrateNodeColor(endpoint.x(), endpoint.y(), endpoint.z(), r, g, b);
    }

    tree->updateInnerOccupancy();
}

void voxelstruct::classifyVoxels() 
{
    this->surface_frontiers.clear();
    this->occupied_voxels.clear();
    this->roi_surface_frontier.clear();

    if (!this->tree) return;

    const double res = this->resolution;

    const int dirs[6][3] = {
        {1, 0, 0}, {-1, 0, 0},
        {0, 1, 0}, {0, -1, 0},
        {0, 0, 1}, {0, 0, -1}
    };

    // Helper: pack an OcTreeKey into a single integer for hashing
    auto packKey = [](const octomap::OcTreeKey& k) -> uint64_t {
        // Each component is typically <= 16 bits at reasonable depths
        return (uint64_t(k.k[0]) << 42) ^ (uint64_t(k.k[1]) << 21) ^ uint64_t(k.k[2]);
    };

    std::unordered_set<uint64_t> frontier_keyset;
    frontier_keyset.reserve(this->tree->size() * 2);

    for (auto it = this->tree->begin_leafs(), end = this->tree->end_leafs(); it != end; ++it) {

        // Only consider occupied leaf voxels as "surface candidates"
        if (!tree->isNodeOccupied(*it)) continue;

        const octomap::ColorOcTreeNode* occ_node = &(*it);

        // Store occupied voxel center (like you already do)
        Eigen::Vector3d occ_center(it.getX(), it.getY(), it.getZ());

        // Optional: keep your red classification for occupied list if you want
        // (I’m leaving occupied_voxels as "everything occupied" like before.)
        this->occupied_voxels.push_back(occ_center);

        // Check if this occupied voxel is "red"
        octomap::ColorOcTreeNode::Color occ_color = occ_node->getColor();
        bool occ_is_red = (occ_color.r >= 140);

        // For each 6-neighbor of this occupied voxel:
        // If neighbor is UNKNOWN => candidate Vu voxel.
        for (const auto& d : dirs) {

            octomap::point3d cand_pt(
                it.getX() + d[0] * res,
                it.getY() + d[1] * res,
                it.getZ() + d[2] * res
            );

            octomap::OcTreeKey cand_key;
            if (!tree->coordToKeyChecked(cand_pt, cand_key)) continue;

            auto* cand_node = tree->search(cand_key);

            // Candidate must be UNKNOWN (Vu)
            if (cand_node != nullptr) continue;

            // Now test Jia frontier condition for this unknown voxel:
            // "frontier if both empty (free) and occupied exist in neighborhood"
            bool has_free = false;
            bool has_occ  = false;
            bool touches_red_occ = false;

            // Check 6-neighborhood of this UNKNOWN voxel
            for (const auto& dd : dirs) {
                octomap::point3d nb_pt(
                    cand_pt.x() + dd[0] * res,
                    cand_pt.y() + dd[1] * res,
                    cand_pt.z() + dd[2] * res
                );

                octomap::OcTreeKey nb_key;
                if (!tree->coordToKeyChecked(nb_pt, nb_key)) continue;

                auto* nb_node = tree->search(nb_key);

                if (nb_node == nullptr) {
                    // unknown neighbor -> doesn't help for free/occ test
                    continue;
                }

                if (tree->isNodeOccupied(*nb_node)) {
                    has_occ = true;

                    // If any adjacent occupied voxel is red, mark as ROI frontier
                    const auto* cnode = static_cast<octomap::ColorOcTreeNode*>(nb_node);
                    auto c = cnode->getColor();
                    if (c.r >= 140) touches_red_occ = true;

                } else {
                    // Node exists and is not occupied => free (Ve)
                    has_free = true;
                }

                if (has_free && has_occ) break;
            }

            if (!(has_free && has_occ)) continue; // not frontier by Jia definition

            // Deduplicate the frontier unknown voxel
            uint64_t keyhash = packKey(cand_key);
            if (!frontier_keyset.insert(keyhash).second) continue;

            // Save frontier voxel center (unknown voxel center)
            Eigen::Vector3d f_center(cand_pt.x(), cand_pt.y(), cand_pt.z());

            if (touches_red_occ || occ_is_red) {
                this->roi_surface_frontier.push_back(f_center);
            } else {
                this->surface_frontiers.push_back(f_center);
            }
        }
    }
}

std::vector<Eigen::Vector3d> voxelstruct::getSurfaceFrontiers() {
    return this->surface_frontiers;
};  

std::vector<Eigen::Vector3d> voxelstruct::getOccupiedVoxels() {
    return this->occupied_voxels;
};

std::vector<Eigen::Vector3d> voxelstruct::getROISurfaceFrontiers() {
    return this->roi_surface_frontier;
};

int voxelstruct::size() {
    return this->tree->size();
}

void voxelstruct::showVoxelTree()
{
    if (!tree) return;

    auto cloud = std::make_shared<open3d::geometry::PointCloud>();
    cloud->points_.reserve(tree->size());

    for (auto it = tree->begin(), end = tree->end(); it != end; ++it) {
        if (tree->isNodeOccupied(*it)) {
            auto c = it.getCoordinate();
            cloud->points_.push_back(Eigen::Vector3d(c.x(), c.y(), c.z()));
        }
    }

    cloud->PaintUniformColor(Eigen::Vector3d(0, 0, 1)); // blue or whatever
    open3d::visualization::DrawGeometries({cloud}, "Occupied Voxels (points)");
}

double voxelstruct::getResolution() 
{
    return this->resolution;
}

void voxelstruct::killVoxelStruct()
{
    surface_frontiers.clear();
    occupied_voxels.clear();
    roi_surface_frontier.clear();

    if (pcd) {
        delete pcd;
        pcd = nullptr;
    }

    if (tree) {
        delete tree;
        tree = nullptr;
    }
}

bool voxelstruct::saveOctree(const std::string& path) const
{
    if (!tree) {
        std::cout << "Octree not initialized.\n";
        return false;
    }
    // ColorOcTree inherits OcTree; write() exists
    bool ok = tree->write(path);
    if (!ok) std::cout << "Failed to save octree to: " << path << "\n";
    else std::cout << "Saved octree to: " << path << "\n";
    return ok;
}

void voxelstruct::showClassifiedVoxels()
{
    if (!tree) {
        std::cout << "Octree not initialized.\n";
        return;
    }

    // Make sure lists are up-to-date
    this->classifyVoxels();

    const auto sf  = this->surface_frontiers;
    const auto occ = this->occupied_voxels;
    const auto roi = this->roi_surface_frontier;

    auto cloud = std::make_shared<open3d::geometry::PointCloud>();
    cloud->points_.reserve(sf.size() + occ.size() + roi.size());
    cloud->colors_.reserve(sf.size() + occ.size() + roi.size());

    // Occupied (blue)
    for (const auto& p : occ) {
        cloud->points_.push_back(p);
        cloud->colors_.push_back(Eigen::Vector3d(0.0, 0.0, 1.0));
    }

    // Surface frontiers (red)
    for (const auto& p : sf) {
        cloud->points_.push_back(p);
        cloud->colors_.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
    }

    // ROI surface frontiers (green)
    for (const auto& p : roi) {
        cloud->points_.push_back(p);
        cloud->colors_.push_back(Eigen::Vector3d(0.0, 1.0, 0.0));
    }

    open3d::visualization::DrawGeometries({cloud}, "Occupied (Blue), Frontier (Red), ROI (Green)");
}