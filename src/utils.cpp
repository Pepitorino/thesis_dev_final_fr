#include "utils.hpp"
#include "json.hpp"
#include <string>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <open3d/Open3D.h>
#include "structs.hpp"

std::vector<double> getVectorOrEmpty(const json& j, const std::string& key) 
{
    if (j.contains(key) && j[key].is_array())
        return j[key].get<std::vector<double>>();
    return {};
}

Eigen::Vector3d getVec3OrDefault(const json& j, const std::string& key) 
{
    if (j.contains(key)) {
        const auto& obj = j[key];
        if (obj.contains("x") && obj.contains("y") && obj.contains("z")) {
            return Eigen::Vector3d(obj["x"], obj["y"], obj["z"]);
        }
    }
    return Eigen::Vector3d::Zero();
}

bool saveVec3AsPLY(const std::string& path, const std::vector<Eigen::Vector3d>& pts) {
    open3d::geometry::PointCloud cloud;
    cloud.points_ = pts;              // copies points
    // no colors needed for ellipsoid fitting; add if you want
    bool ok = open3d::io::WritePointCloud(path, cloud);
    if (!ok) std::cout << "Failed to save: " << path << "\n";
    else     std::cout << "Saved: " << path << " (" << pts.size() << " pts)\n";
    return ok;
}

bool parseVec3(const std::string& s, Eigen::Vector3d& out) {
    std::string t = s;
    for (char& ch : t) if (ch == ',') ch = ' ';
    std::istringstream iss(t);

    double a, b, c;
    if (!(iss >> a >> b >> c)) return false;
    out = Eigen::Vector3d(a, b, c);
    return true;
}

bool writeVec3ListToFile(
    const std::string& path,
    const std::vector<Eigen::Vector3d>& pts,
    const std::string& header)
{
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cout << "Failed to open file: " << path << "\n";
        return false;
    }
    out << header << "\n";
    for (size_t i = 0; i < pts.size(); ++i) {
        out << i << " " << pts[i].x() << " " << pts[i].y() << " " << pts[i].z() << "\n";
    }
    std::cout << "Wrote " << pts.size() << " points to " << path << "\n";
    return true;
}

std::string prompt(const std::string& msg) {
    std::cout << msg;
    std::string s;
    std::getline(std::cin, s);
    return s;
}

 bool parseViewpoint6(const std::string& s, Eigen::Matrix<double,6,1>& out) {
    std::string t = s;
    for (char& ch : t) {
        if (ch == ',') ch = ' ';
    }
    std::istringstream iss(t);
    double v[6];
    for (int i = 0; i < 6; ++i) {
        if (!(iss >> v[i])) return false;
    }
    v[3] = v[3] * std::acos(-1.0) / 180.0;
    v[4] = v[4] * std::acos(-1.0) / 180.0;
    v[5] = v[5] * std::acos(-1.0) / 180.0;
    out << v[0], v[1], v[2], v[3], v[4], v[5];
    return true;
}

// Convert Open3D PointCloud -> vector<Eigen::Vector3d>
std::vector<Eigen::Vector3d> cloudToVec3(const open3d::geometry::PointCloud& cloud) {
    std::vector<Eigen::Vector3d> out;
    out.reserve(cloud.points_.size());
    for (const auto& p : cloud.points_) out.push_back(p);
    return out;
}

 bool loadCloudAsVec3(const std::string& path, std::vector<Eigen::Vector3d>& out) {
    open3d::geometry::PointCloud cloud;
    if (!open3d::io::ReadPointCloud(path, cloud) || cloud.points_.empty()) {
        std::cout << "Failed to load or empty: " << path << "\n";
    }
    out = cloudToVec3(cloud);
    std::cout << "Loaded " << out.size() << " points from " << path << "\n";
    return true;
}

std::shared_ptr<open3d::geometry::PointCloud>
buildCloudFromClusters(const std::vector<std::vector<Eigen::Vector3d>>& clusters,
                       const Eigen::Vector3d& color,
                        double voxel_size)
{
    auto cloud = std::make_shared<open3d::geometry::PointCloud>();

    size_t total = 0;
    for (const auto& c : clusters) total += c.size();
    cloud->points_.reserve(total);

    for (const auto& c : clusters)
        for (const auto& p : c)
            cloud->points_.push_back(p);

    if (voxel_size > 0.0) {
        cloud = cloud->VoxelDownSample(voxel_size);
    }

    cloud->PaintUniformColor(color);
    return cloud;
}

// Create ellipsoid meshes/linesets (same logic as your showEllipsoidsOpen3D)
std::vector<std::shared_ptr<const open3d::geometry::Geometry>>
buildEllipsoidGeoms(const std::vector<EllipsoidParam>& ellipsoids,
                    bool wireframe,
                    int sphere_resolution)
{
    std::vector<std::shared_ptr<const open3d::geometry::Geometry>> geoms;

    auto color_for_type = [](const std::string& type) -> Eigen::Vector3d {
        if (type == "roi_surface_frontier") return {0.0, 1.0, 0.0}; // green (you wanted ROI green earlier)
        if (type == "frontier")            return {1.0, 0.0, 0.0}; // red
        if (type == "occupied")            return {0.0, 0.0, 1.0}; // blue
        return {0.7, 0.7, 0.7};
    };

    for (const auto& e : ellipsoids) {
        auto sphere = open3d::geometry::TriangleMesh::CreateSphere(1.0, sphere_resolution);
        sphere->ComputeVertexNormals();

        // Anisotropic scaling by directly scaling vertices
        for (auto& v : sphere->vertices_) {
            v(0) *= e.radii.x();
            v(1) *= e.radii.y();
            v(2) *= e.radii.z();
        }

        sphere->Transform(e.pose);
        sphere->PaintUniformColor(color_for_type(e.type));

        if (wireframe) {
            auto ls = open3d::geometry::LineSet::CreateFromTriangleMesh(*sphere);
            ls->PaintUniformColor(color_for_type(e.type));
            geoms.push_back(ls);
        } else {
            geoms.push_back(sphere);
        }
    }

    return geoms;
}

// Save a visualization to PNG by rendering the geometries off the normal DrawGeometries path
bool saveGeomsScreenshot(const std::vector<std::shared_ptr<const open3d::geometry::Geometry>>& geoms,
                                const std::string& png_path,
                                const std::string& title,
                                int w,
                                int h)
{
    if (geoms.empty()) {
        std::cout << "Nothing to render.\n";
        return false;
    }

    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow(title, w, h, 50, 50, /*visible=*/true);

    for (const auto& g : geoms) vis.AddGeometry(g);

    // Let Open3D build the view once
    vis.PollEvents();
    vis.UpdateRender();

    vis.CaptureScreenImage(png_path, /*do_render=*/true);
    vis.DestroyVisualizerWindow();

    std::cout << "Saved screenshot (requested): " << png_path << "\n";
    return true;
}