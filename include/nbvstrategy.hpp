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
    void generateViewpoints_fixedDistance();
    void kill();

    //inserting pointclouds
    void insertTransformedCloud(std::string file_path, 
        Eigen::Vector3d vp);

    //getNBVs
    void getNBV();
    std::pair<double, cv::Mat> projectEllipsoidstoImage(
        const std::vector<EllipsoidParam> &ellipsoids,
        const Eigen::Matrix4d &T_cam_world);
    Eigen::Matrix3d compute_ellipsoid_projection(
        const Eigen::Matrix<double, 3, 4>& camera_matrix,
        const Eigen::Matrix4d& ellipsoid_matrix_dual);
    Eigen::Matrix4d create_ellipsoid_dual_matrix(const EllipsoidParam &param);
    bool SphereInFrustum(
        const Eigen::Vector3d &center_cam,
        double rx, double ry, double rz,
        const Eigen::Matrix3d &R_e_cam,
        const Camera &cam);

    //more viz stuff (chatgpt)
    bool hasBestNBV() const;
    void showLastNBVInfo() const;
    void showLastNBVInVoxelTree();
    bool saveLastNBV(const std::string& txt_path, const std::string& img_path) const;
    
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
    double r, dc;

    voxelstruct* voxel_struct;
    ellipsoid* ellipsoid_fitting;
    nbvtransform transform;

    //VIEWPOINTS
    std::vector<Eigen::Matrix<double,6,1>> viewpoints;
};