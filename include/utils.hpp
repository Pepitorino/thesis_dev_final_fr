#pragma once
#include "json.hpp"
#include <string>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <open3d/Open3D.h>
#include "structs.hpp"

using json = nlohmann::json;

bool saveVec3AsPLY(const std::string& path, const std::vector<Eigen::Vector3d>& pts);

bool parseVec3(const std::string& s, Eigen::Vector3d& out);

bool writeVec3ListToFile(
    const std::string& path,
    const std::vector<Eigen::Vector3d>& pts,
    const std::string& header
);

std::string prompt(const std::string& msg);

bool parseViewpoint6(const std::string& s, Eigen::Matrix<double,6,1>& out);

std::vector<Eigen::Vector3d> cloudToVec3(const open3d::geometry::PointCloud& cloud);

bool loadCloudAsVec3(const std::string& path, std::vector<Eigen::Vector3d>& out);

std::shared_ptr<open3d::geometry::PointCloud>
buildCloudFromClusters(
    const std::vector<std::vector<Eigen::Vector3d>>& clusters,
    const Eigen::Vector3d& color,
    double voxel_size = 0.0
);

std::vector<std::shared_ptr<const open3d::geometry::Geometry>>
buildEllipsoidGeoms(
    const std::vector<EllipsoidParam>& ellipsoids,
    bool wireframe,
    int sphere_resolution
);

bool saveGeomsScreenshot(
    const std::vector<std::shared_ptr<const open3d::geometry::Geometry>>& geoms,
    const std::string& png_path,
    const std::string& title = "viz",
    int w = 1400,
    int h = 900
);

template<typename T>
T getOrDefault(const json& j, const std::string& key, T fallback) {
    if (j.contains(key) && !j[key].is_null()) {
        return j[key].get<T>();
    }
    return fallback;
}

std::vector<double> getVectorOrEmpty(const json& j, 
    const std::string& key);

Eigen::Vector3d getVec3OrDefault(const json& j, 
    const std::string& key);