#pragma once
#include "voxelstruct.hpp"
#include "ellipsoid.hpp"
#include "cuda_projection.hpp"
#include "structs.hpp"
#include "nbvtransform.hpp"
#include "json.hpp"
#include "utils.hpp"

class nbvstrategy
{
public:
    //initialization and destruction
    int initialize(std::string settings_path);
    void generateViewpoints();
    void kill();

    //inserting pointclouds
    void insertTransformedCloud(std::string file_path, 
        Eigen::Vector3d vp);

    //getNBVs
    void getNBV();
    
    //NEXT BEST VIEW
    double best_score;
    cv::Mat best_image;
    Eigen::Matrix<double,6,1> best_viewpoint;
    size_t best_viewpoint_index;
private:
    //SETTINGS
    Camera cam_intrinsics;
    Eigen::Vector3d bbx_min, bbx_max;
    std::vector<PlantBBX> bbx_plants;
    int min_clusters, max_clusters;
    double resolution;
    double dx, dy, dz, dyaw, dpitch;

    voxelstruct* voxel_struct;
    ellipsoid* ellipsoid_fitting;
    nbvtransform transform;

    //VIEWPOINTS
    std::vector<Eigen::Matrix<double,6,1>> viewpoints;
};