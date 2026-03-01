#pragma once 
#include "structs.hpp"
#include <Eigen/Dense>
#include <CGAL/Cartesian_d.h>
#include <CGAL/MP_Float.h>
#include <CGAL/point_generators_d.h>
#include <CGAL/Approximate_min_ellipsoid_d.h>
#include <CGAL/Approximate_min_ellipsoid_d_traits_d.h>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

class ellipsoid
{
public:
    ellipsoid(int min_clusters, int max_clusters);
    std::vector<std::vector<Eigen::Vector3d>> gmm_clustering(
        const std::vector<Eigen::Vector3d> &voxels
    );
    std::vector<EllipsoidParam> ellipsoidize_clusters_CGAL(
        const std::vector<std::vector<Eigen::Vector3d>> frontier_clusters,
        const std::vector<std::vector<Eigen::Vector3d>> clusters,
        const std::vector<std::vector<Eigen::Vector3d>> roi_surface_frontier
    );
    void showAllClustersColored(
        const std::vector<std::vector<Eigen::Vector3d>>& frontier,
        const std::vector<std::vector<Eigen::Vector3d>>& occupied,
        const std::vector<std::vector<Eigen::Vector3d>>& roi,
        double voxel_size = 0.0
    );
    void showEllipsoidsOpen3D(
        const std::vector<EllipsoidParam>& ellipsoids,
        bool wireframe = true,
        int sphere_resolution = 20
    );


private:
    int min_clusters;
    int max_clusters;
};