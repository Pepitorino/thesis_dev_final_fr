#include "nbvtransform.hpp"
#include <open3d/io/PointCloudIO.h>
#include <fstream>
#include <iostream>
#include <open3d/visualization/utility/DrawGeometry.h>
#include <open3d/geometry/TriangleMesh.h>

void nbvtransform::cropBBX(
    const Eigen::Vector3d& bbx_min,
    const Eigen::Vector3d& bbx_max,
    open3d::geometry::PointCloud* cloud)
{
    if (!cloud || cloud->points_.empty())
        return;

    auto& points = cloud->points_;
    auto& colors = cloud->colors_;

    std::vector<Eigen::Vector3d> new_points;
    std::vector<Eigen::Vector3d> new_colors;

    new_points.reserve(points.size());
    if (!colors.empty())
        new_colors.reserve(colors.size());

    for (size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector3d& p = points[i];

        if ((p.array() >= bbx_min.array()).all() &&
            (p.array() <= bbx_max.array()).all()) {

            new_points.push_back(p);
            if (!colors.empty()) {
                new_colors.push_back(colors[i]);
            }
        }
    }

    points.swap(new_points);
    if (!colors.empty())
        colors.swap(new_colors);
}

Eigen::Matrix4d nbvtransform::getCameraPose(
    const Eigen::Matrix<double,6,1> &vp) 
{
    Eigen::Matrix3d R_yaw = Eigen::AngleAxisd(
                            vp(3),Eigen::Vector3d::UnitY())
                            .toRotationMatrix();
    Eigen::Matrix3d R_pitch = Eigen::AngleAxisd(
                            -vp(4), Eigen::Vector3d::UnitX())
                            .toRotationMatrix();
    Eigen::Matrix3d R_roll = Eigen::AngleAxisd(
                            vp(5), Eigen::Vector3d::UnitZ())
                            .toRotationMatrix();

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = R_yaw * R_pitch * R_roll;   // roll pitch yaw
    T.block<3,1>(0,3) = Eigen::Vector3d(vp(0), vp(1), vp(2));
    return T;
}

open3d::geometry::PointCloud nbvtransform::translateToWorldFrame(
    const Eigen::Matrix4d &T_cam_world, 
    const open3d::geometry::PointCloud* pcd
)
{
    open3d::geometry::PointCloud pcd_world;
    pcd_world.points_.resize(pcd->points_.size());
    pcd_world.colors_.resize(pcd->colors_.size());

    Eigen::Matrix4d T = T_cam_world;

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(pcd->points_.size()); i++) {
        const auto &p = pcd->points_[i];
        Eigen::Vector4d p_homogeneous(p(0),p(1),p(2), 1.0);
        Eigen::Vector4d p_transformed_homogeneous = T * p_homogeneous;
        pcd_world.points_[i] = p_transformed_homogeneous.head<3>();
        pcd_world.colors_[i] = pcd->colors_[i];
    }

    return pcd_world;
}

// x = x, y = y, z = z
// a = -x, b = -y, c = -z
void nbvtransform::switchXYZ(
    char x, char y, char z,
    open3d::geometry::PointCloud* pcd
)
{
    if (!pcd) return;

    auto decode = [](char c, int& axis, double& sign) {
        switch (c) {
            case 'x': axis = 0; sign =  1.0; break;
            case 'y': axis = 1; sign =  1.0; break;
            case 'z': axis = 2; sign =  1.0; break;
            case 'a': axis = 0; sign = -1.0; break; // -x
            case 'b': axis = 1; sign = -1.0; break; // -y
            case 'c': axis = 2; sign = -1.0; break; // -z
            default:  axis = 0; sign =  0.0; break; // fallback -> zero row
        }
    };

    int ax0, ax1, ax2;
    double s0, s1, s2;
    decode(x, ax0, s0);
    decode(y, ax1, s1);
    decode(z, ax2, s2);

    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    R(0, ax0) = s0;
    R(1, ax1) = s1;
    R(2, ax2) = s2;

    for (auto& p : pcd->points_) {
        Eigen::Vector3d old = p;
        p = R * old;
    }
}

void nbvtransform::loadPCD(std::string path) 
{
    if (!pcd)
        pcd = new open3d::geometry::PointCloud();

    pcd->Clear();

    bool ok = open3d::io::ReadPointCloud(path, *pcd);

    if (!ok) {
        std::cout << "Failed to load point cloud: " << path << "\n";
        delete pcd;
        pcd = nullptr;
        return;
    }

    std::cout << "Loaded point cloud: " << path << "\n";
    std::cout << "Points: " << pcd->points_.size() << "\n";
}

bool nbvtransform::savePCD(const std::string& path) const 
{
    if (!pcd || pcd->points_.empty()) {
        std::cout << "No point cloud loaded.\n";
        return false;
    }

    // Automatically detects format from extension
    bool ok = open3d::io::WritePointCloud(path, *pcd);

    if (!ok)
        std::cout << "Failed to save point cloud.\n";
    else
        std::cout << "Saved to: " << path << "\n";

    return ok;
}


void nbvtransform::viewPCD(const std::string& window_title) const 
{
    if (!pcd) {
        std::cout << "No point cloud loaded.\n";
        return;
    }

    if (pcd->points_.empty()) {
        std::cout << "Point cloud is empty.\n";
        return;
    }

    // Create coordinate axes at origin
    auto axes = open3d::geometry::TriangleMesh::CreateCoordinateFrame(
        0.5,                      // axis length
        Eigen::Vector3d(0,0,0)    // origin
    );

    open3d::visualization::DrawGeometries(
        {
            std::make_shared<open3d::geometry::PointCloud>(*pcd),
            axes
        },
        window_title
    );
}

void nbvtransform::printAllPointsToFile(const std::string& path) const 
{
    if (!pcd) {
        std::cout << "No point cloud loaded.\n";
        return;
    }
    
    if (pcd->points_.empty()) {
        std::cout << "Point cloud is empty.\n";
        return;
    }
    
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cout << "Failed to open file: " << path << "\n";
        return;
    }
    
    out << "# index x y z\n";
    
    for (size_t i = 0; i < pcd->points_.size(); ++i) {
        const auto& p = pcd->points_[i];
        out << i << " "
        << p.x() << " "
        << p.y() << " "
        << p.z() << "\n";
    }
    
    out.close();
    std::cout << "Wrote " << pcd->points_.size()
    << " points to " << path << "\n";
}

void nbvtransform::killNBVTransform() 
{
    if (pcd) {
        delete pcd;
        pcd = nullptr;
        std::cout << "Point cloud memory freed.\n";
    }
}