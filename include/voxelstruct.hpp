#pragma once
#include "structs.hpp"
#include <string>
#include <Eigen/Dense>
#include <open3d/Open3D.h>
#include <octomap/Pointcloud.h>
#include <octomap/ColorOcTree.h>
#include <octomap/octomap.h>

/* just as a sanity check:
Need this to literally just manage the octomap - same as before
Want to insert point clouds, see what it looks like and decide whether or not it wants to keep it - could probably just copy paste the old merging
Don't need a pcd list anymore since the camera module will also handle that
*/

class voxelstruct {
public:
    voxelstruct(double resolution);
    
    void insertPointCloud(open3d::geometry::PointCloud* pcd, Eigen::Vector3d camera);

    void classifyVoxels();
    std::vector<Eigen::Vector3d> getSurfaceFrontiers();
    std::vector<Eigen::Vector3d> getOccupiedVoxels();
    std::vector<Eigen::Vector3d> getROISurfaceFrontiers();

    // helper functions
    int size();
    double getResolution();

    void showVoxelTree();

    void killVoxelStruct();
    bool saveOctree(const std::string& path) const;

private:
    double resolution;
    
    octomap::ColorOcTree* tree;
    open3d::geometry::PointCloud* pcd;

    std::vector<Eigen::Vector3d> surface_frontiers;
    std::vector<Eigen::Vector3d> occupied_voxels;
    std::vector<Eigen::Vector3d> roi_surface_frontier;
};