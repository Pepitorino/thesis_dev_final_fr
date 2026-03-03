#pragma once
#include <Eigen/Dense>
#include <open3d/Open3D.h>
#include "structs.hpp"
#include <string>
// start with this -> do transforms then worry about octomap then worry about ellipsoids then worry about the nbvstrategy

class nbvtransform {
public:
    void cropBBX(
        const Eigen::Vector3d& bbx_max, 
        const Eigen::Vector3d& bbx_min,
        open3d::geometry::PointCloud* cloud);

    Eigen::Matrix4d getCameraPose(
        const Eigen::Matrix<double,6,1> &vp
    );

    open3d::geometry::PointCloud translateToWorldFrame(
        const Eigen::Matrix4d &T_cam_world, 
        const open3d::geometry::PointCloud* pcd
    );

    void switchXYZ(
        char x, char y, char z,
        open3d::geometry::PointCloud* pcd
    );

    void loadPCD(std::string path);
    bool savePCD(const std::string& path) const;
    void viewPCD(const std::string& window_title = "Open3D Viewer") const;
    void killNBVTransform();
    void printAllPointsToFile(const std::string& path) const;

    open3d::geometry::PointCloud* pcd;
};