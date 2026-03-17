#include "ellipsoid.hpp"
#include <open3d/Open3D.h>

ellipsoid::ellipsoid(int min_clusters, int max_clusters) {
    this->min_clusters = min_clusters;
    this->max_clusters = max_clusters;
}

std::vector<std::vector<Eigen::Vector3d>> ellipsoid::gmm_clustering(
    const std::vector<Eigen::Vector3d> &voxels
) {
    // 创建一个数据集
    cv::Mat samples(voxels.size(), 3, CV_32F);
    for (size_t i = 0; i < voxels.size(); i++)
    {
        samples.at<float>(i, 0) = static_cast<float>(voxels[i][0]);
        samples.at<float>(i, 1) = static_cast<float>(voxels[i][1]);
        samples.at<float>(i, 2) = static_cast<float>(voxels[i][2]);
    }
    if (samples.empty())
    {
        std::vector<std::vector<Eigen::Vector3d>> zero_clouds;
        return zero_clouds;
    }
    
    printf("Sampling done!\n");
    
    #ifdef DEBUG
    {
        std::cout << samples.row(0) << std::endl;
        std::cout << samples.row(10) << std::endl;
        cv::Scalar mean, stddev;
        cv::meanStdDev(samples, mean, stddev);
        std::cout << "samples.rows=" << samples.rows << " cols=" << samples.cols
                  << " mean: " << mean << " stddev: " << stddev << std::endl;
    }
    #endif

    std::vector<std::vector<Eigen::Vector3d>> clustered_clouds;

    if (samples.rows >= 2*this->max_clusters)
    {
        printf("sample rows > 2*this->max_clusters");
        // 创建并训练 GMM
        cv::Mat output_labels;
        int gmm_cnt = this->max_clusters - this->min_clusters + 1;
        // 用于多线程存储
        std::vector<double> cluster_values(gmm_cnt);
        std::vector<std::vector<std::vector<Eigen::Vector3d>>> clustered_clouds_vec(gmm_cnt);
        
        printf("clustering!\n");
        #pragma omp parallel for
        for (size_t i = this->min_clusters; i <= size_t(this->max_clusters); i++)
        {   
            if (size_t(samples.rows) < i)
            {
                cluster_values[i] = DBL_MIN;
                continue;
            }
            
            cv::Ptr<cv::ml::EM> em_model = cv::ml::EM::create();
            em_model->setCovarianceMatrixType(cv::ml::EM::COV_MAT_DIAGONAL);
            // em_model->setCovarianceMatrixType(cv::ml::EM::COV_MAT_SPHERICAL);
            em_model->setClustersNumber(i); // 高斯混合模型的数量
            em_model->trainEM(samples);

            // 使用训练好的 GMM 进行预测
            cv::Mat labels;
            cv::Mat logLikelihoods;
            em_model->predict(samples, labels);
            double total_log_likelihood = 0.0;
            // Calculate likelihood
            std::vector<std::vector<Eigen::Vector3d>> clustered_clouds_tmp(i);
            // 提取每个cluster的点云
            for (size_t j = 0; j < size_t(samples.rows); j++)
            {
                double max = -1;
                int max_index = -1;

                // 找到最大概率及其索引
                for (size_t k = 0; k < i; k++)
                {
                    double current_prob = labels.row(j).at<double>(k);
                    if(current_prob > max){
                        max = current_prob;
                        max_index = k;
                    }
                }

                // 假设max是概率，计算对数似然
                if (max > 0) {
                    double log_likelihood = log(max); // 使用对数函数计算对数似然
                    total_log_likelihood += log_likelihood; // 累加到总对数似然
                }

                clustered_clouds_tmp[max_index].push_back(voxels[j]);
            }

            // 计算评价函数
            // 评价函数为 total_log_likelihood 最大似然函数的的值 - 每个cluster 中的点的数量的倒数 * 2
            double value = total_log_likelihood;
            // bic 准则
            value = 3*log(samples.rows) - 2 * value; 
            // for (size_t j = 0; j < i; j++)
            // {
            //     if (clustered_clouds_tmp[j].size() > 1)
            //     {
            //         value -= 1.0 * 1.0 / clustered_clouds_tmp[j].size();
            //     }
            //     else{
            //         value -= 1.0;
            //     }
            // }
            // mtx.lock();
            // LOG(INFO) << "cluster num: " << i << " value: " << value;
            // mtx.unlock();
            cluster_values[i-this->min_clusters] = value;
            clustered_clouds_vec[i-this->min_clusters] = clustered_clouds_tmp;
        }

        // // 在 cluster_values 中找出最大值的索引
        // int max_idx = std::distance(cluster_values.begin(), std::max_element(cluster_values.begin(), cluster_values.end()));
        // clustered_clouds = clustered_clouds_vec[max_idx];
        // 在 cluster_values 中找出最小值的索引
        int min_idx = std::distance(cluster_values.begin(), std::min_element(cluster_values.begin(), cluster_values.end()));
        clustered_clouds = clustered_clouds_vec[min_idx];
    }
    else if (samples.rows > 3)
    {
        printf("clustering!\n");
        int cluster_num = 2;
        cv::Ptr<cv::ml::EM> em_model = cv::ml::EM::create();
        em_model->setCovarianceMatrixType(cv::ml::EM::COV_MAT_DIAGONAL);
        em_model->setClustersNumber(cluster_num); // 高斯混合模型的数量
        clustered_clouds.resize(cluster_num);
        em_model->trainEM(samples);
        // 使用训练好的 GMM 进行预测
        cv::Mat labels;
        em_model->predict(samples, labels);

        // 提取每个cluster的点云
        for (size_t j = 0; j < size_t(samples.rows); j++)
        {
            double max = -1;
            int max_index = -1;

            // 找到最大概率及其索引
            for (size_t k = 0; k < size_t(cluster_num); k++)
            {
                double current_prob = labels.row(j).at<double>(k);
                if(current_prob > max){
                    max = current_prob;
                    max_index = k;
                }
            }

            clustered_clouds[max_index].push_back(voxels[j]);
        }
    }else{
        clustered_clouds.resize(1);
        for (size_t i = 0; i < size_t (samples.rows); i++)
        {
            clustered_clouds[0].push_back(voxels[i]);
        }
    }
    

    return clustered_clouds;
}

static EllipsoidParam makeFallbackEllipsoid(
    const std::vector<Eigen::Vector3d>& cluster,
    const std::string& type,
    double min_radius = 0.005)
{
    min_radius = 0.05;
    EllipsoidParam e;
    e.pose = Eigen::Matrix4d::Identity();
    e.type = type;

    if (cluster.empty()) {
        e.radii = Eigen::Vector3d(min_radius, min_radius, min_radius);
        return e;
    }

    // centroid
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& p : cluster) mean += p;
    mean /= static_cast<double>(cluster.size());

    // covariance
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& p : cluster) {
        Eigen::Vector3d d = p - mean;
        cov += d * d.transpose();
    }

    if (cluster.size() > 1)
        cov /= static_cast<double>(cluster.size() - 1);

    // PCA
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    if (solver.info() != Eigen::Success) {
        e.pose.block<3,1>(0,3) = mean;
        e.radii = Eigen::Vector3d(min_radius, min_radius, min_radius);
        return e;
    }

    // Eigen gives ascending eigenvalues
    Eigen::Vector3d evals = solver.eigenvalues();
    Eigen::Matrix3d evecs = solver.eigenvectors();

    // Reorder so largest axis comes first
    std::array<int,3> order = {2,1,0};
    Eigen::Matrix3d R;
    Eigen::Vector3d radii;

    for (int j = 0; j < 3; ++j) {
        int idx = order[j];
        R.col(j) = evecs.col(idx);

        // stddev-like radius estimate, clamped
        double r = std::sqrt(std::max(0.0, evals(idx)));
        radii(j) = std::max(min_radius, r);
    }

    // Make rotation right-handed
    if (R.determinant() < 0.0) {
        R.col(2) *= -1.0;
    }

    e.pose.block<3,1>(0,3) = mean;
    e.pose.block<3,3>(0,0) = R;
    e.radii = radii;

    return e;
}

std::vector<EllipsoidParam> ellipsoid::ellipsoidize_clusters_CGAL(
    const std::vector<std::vector<Eigen::Vector3d>> frontier_clusters,
    const std::vector<std::vector<Eigen::Vector3d>> occupied_clusters,
    const std::vector<std::vector<Eigen::Vector3d>> roi_surface_frontier
) {
    typedef CGAL::Cartesian_d<double>                              Kernel;
    typedef CGAL::MP_Float                                         ET;
    typedef CGAL::Approximate_min_ellipsoid_d_traits_d<Kernel, ET> Traits;
    typedef Traits::Point                                          Point;
    typedef std::vector<Point>                                     Point_list;
    typedef CGAL::Approximate_min_ellipsoid_d<Traits>              AME;

    const double eps = 0.01; // approximation ratio (1+eps)
    Traits traits;
    const int d = 3;

    std::vector<EllipsoidParam> ellipsoid_vec;

    std::cout << "Frontier clusters: " << frontier_clusters.size() << std::endl;
    std::cout << "Occupied clusters: " << occupied_clusters.size() << std::endl;
    std::cout << "ROI Surface Frontier clusters: " << roi_surface_frontier.size() << std::endl;

    // --- FRONTIER CLUSTERS ---
    if (!frontier_clusters.empty())
    {
        #pragma omp parallel for
        for (size_t i = 0; i < frontier_clusters.size(); ++i)
        {
            Point_list points;
            for (const auto &v : frontier_clusters[i])
            {
                std::vector<double> vec(v.data(), v.data() + 3);
                points.push_back(Point(3, vec.begin(), vec.end()));
            }

            EllipsoidParam e;

            if (frontier_clusters[i].size() < 4) {
                e = makeFallbackEllipsoid(frontier_clusters[i], "frontier", 0.005);
            } else {
                AME mel(eps, points.begin(), points.end(), traits);

                if (!mel.is_full_dimensional()) {
                    e = makeFallbackEllipsoid(frontier_clusters[i], "frontier", 0.005);
                } else {
                    auto radii = mel.axes_lengths_begin();
                    auto centroid = mel.center_cartesian_begin();
                    auto d0 = mel.axis_direction_cartesian_begin(0);
                    auto d1 = mel.axis_direction_cartesian_begin(1);
                    auto d2 = mel.axis_direction_cartesian_begin(2);

                    e.pose = Eigen::Matrix4d::Identity();
                    e.pose.block<3,1>(0,3) = Eigen::Vector3d(centroid[0], centroid[1], centroid[2]);
                    e.pose.block<3,3>(0,0) = (Eigen::Matrix3d() <<
                        d0[0], d1[0], d2[0],
                        d0[1], d1[1], d2[1],
                        d0[2], d1[2], d2[2]).finished();

                    if (e.pose.block<3,3>(0,0).determinant() < 0.0) {
                        e.pose.block<3,1>(0,2) *= -1.0;
                    }

                    e.radii = Eigen::Vector3d(
                        std::max(0.005, radii[0]),
                        std::max(0.005, radii[1]),
                        std::max(0.005, radii[2])
                    );
                    e.type = "frontier";
                }
            }

            #pragma omp critical
            ellipsoid_vec.push_back(e);
        }

        std::cout << "Frontier ellipsoids computed: "
                << frontier_clusters.size() << std::endl;
    }

    std::cout << "Total ellipsoids after frontier computed: " << ellipsoid_vec.size() << std::endl;

    // --- ROI Surface Frontier ---
    if (!roi_surface_frontier.empty())
    {
        #pragma omp parallel for
        for (size_t i = 0; i < roi_surface_frontier.size(); ++i)
        {
            Point_list points;
            for (const auto &v : roi_surface_frontier[i])
            {
                std::vector<double> vec(v.data(), v.data() + 3);
                points.push_back(Point(3, vec.begin(), vec.end()));
            }

            EllipsoidParam e;

            // tiny clusters can skip straight to fallback if you want
            if (roi_surface_frontier[i].size() < 4) {
                e = makeFallbackEllipsoid(roi_surface_frontier[i], "roi_surface_frontier", 0.005);
            } else {
                AME mel(eps, points.begin(), points.end(), traits);

                if (!mel.is_full_dimensional()) {
                    e = makeFallbackEllipsoid(roi_surface_frontier[i], "roi_surface_frontier", 0.005);
                } else {
                    auto radii = mel.axes_lengths_begin();
                    auto centroid = mel.center_cartesian_begin();
                    auto d0 = mel.axis_direction_cartesian_begin(0);
                    auto d1 = mel.axis_direction_cartesian_begin(1);
                    auto d2 = mel.axis_direction_cartesian_begin(2);

                    e.pose = Eigen::Matrix4d::Identity();
                    e.pose.block<3,1>(0,3) = Eigen::Vector3d(centroid[0], centroid[1], centroid[2]);
                    e.pose.block<3,3>(0,0) = (Eigen::Matrix3d() <<
                        d0[0], d1[0], d2[0],
                        d0[1], d1[1], d2[1],
                        d0[2], d1[2], d2[2]).finished();

                    if (e.pose.block<3,3>(0,0).determinant() < 0.0) {
                        e.pose.block<3,1>(0,2) *= -1.0;
                    }

                    e.radii = Eigen::Vector3d(
                        std::max(0.005, radii[0]),
                        std::max(0.005, radii[1]),
                        std::max(0.005, radii[2])
                    );
                    e.type = "roi_surface_frontier";
                }
            }

            #pragma omp critical
            ellipsoid_vec.push_back(e);
        }

        std::cout << "ROI Surface Frontier ellipsoids computed: "
                << roi_surface_frontier.size() << std::endl;
    }

    std::cout << "Total ellipsoids after roi frontier computed: " << ellipsoid_vec.size() << std::endl;

    // --- OCCUPIED CLUSTERS ---
    //occupied_clusters
    if (!occupied_clusters.empty())
    {
        #pragma omp parallel for
        for (size_t i = 0; i < occupied_clusters.size(); ++i)
        {
            Point_list points;
            for (const auto &v : occupied_clusters[i])
            {
                std::vector<double> vec(v.data(), v.data() + 3);
                points.push_back(Point(3, vec.begin(), vec.end()));
            }

            EllipsoidParam e;

            if (occupied_clusters[i].size() < 4) {
                e = makeFallbackEllipsoid(occupied_clusters[i], "occupied", 0.005);
            } else {
                AME mel(eps, points.begin(), points.end(), traits);

                if (!mel.is_full_dimensional()) {
                    e = makeFallbackEllipsoid(occupied_clusters[i], "occupied", 0.005);
                } else {
                    auto radii = mel.axes_lengths_begin();
                    auto centroid = mel.center_cartesian_begin();
                    auto d0 = mel.axis_direction_cartesian_begin(0);
                    auto d1 = mel.axis_direction_cartesian_begin(1);
                    auto d2 = mel.axis_direction_cartesian_begin(2);

                    e.pose = Eigen::Matrix4d::Identity();
                    e.pose.block<3,1>(0,3) = Eigen::Vector3d(centroid[0], centroid[1], centroid[2]);
                    e.pose.block<3,3>(0,0) = (Eigen::Matrix3d() <<
                        d0[0], d1[0], d2[0],
                        d0[1], d1[1], d2[1],
                        d0[2], d1[2], d2[2]).finished();

                    if (e.pose.block<3,3>(0,0).determinant() < 0.0) {
                        e.pose.block<3,1>(0,2) *= -1.0;
                    }

                    e.radii = Eigen::Vector3d(
                        std::max(0.005, radii[0]),
                        std::max(0.005, radii[1]),
                        std::max(0.005, radii[2])
                    );
                    e.type = "occupied";
                }
            }

            #pragma omp critical
            ellipsoid_vec.push_back(e);
        }

        std::cout << "Occupied ellipsoids computed: "
                << occupied_clusters.size() << std::endl;
    }

    std::cout << "Total ellipsoids: " << ellipsoid_vec.size() << std::endl;

    return ellipsoid_vec;
}

void ellipsoid::showAllClustersColored(
    const std::vector<std::vector<Eigen::Vector3d>>& frontier,
    const std::vector<std::vector<Eigen::Vector3d>>& occupied,
    const std::vector<std::vector<Eigen::Vector3d>>& roi,
    double voxel_size)
{
    std::vector<std::shared_ptr<const open3d::geometry::Geometry>> geoms;

    auto buildCloud = [&](const auto& clusters,
                          const Eigen::Vector3d& color)
    {
        auto cloud = std::make_shared<open3d::geometry::PointCloud>();

        size_t total_points = 0;
        for (const auto& cluster : clusters)
            total_points += cluster.size();

        cloud->points_.reserve(total_points);

        for (const auto& cluster : clusters)
            for (const auto& p : cluster)
                cloud->points_.push_back(p);

        if (voxel_size > 0.0) {
            cloud = cloud->VoxelDownSample(voxel_size);
        }

        cloud->PaintUniformColor(color);
        return cloud;
    };

    Eigen::Vector3d frontier_color(1.0, 1.0, 0.0); // yellow
    Eigen::Vector3d occupied_color(0.0, 0.0, 1.0); // blue
    Eigen::Vector3d roi_color(1.0, 0.0, 0.0);      // red

    auto f = buildCloud(frontier, frontier_color);
    auto o = buildCloud(occupied, occupied_color);
    auto r = buildCloud(roi, roi_color);

    if (!f->points_.empty()) geoms.push_back(f);
    if (!o->points_.empty()) geoms.push_back(o);
    if (!r->points_.empty()) geoms.push_back(r);

    if (geoms.empty()) {
        std::cout << "No points to visualize.\n";
        return;
    }

    open3d::visualization::DrawGeometries(
        geoms,
        "Clustered Voxels (Frontier/ROI/Occupied)",
        1400,
        900
    );
}

void ellipsoid::showEllipsoidsOpen3D(
    const std::vector<EllipsoidParam>& ellipsoids,
    bool wireframe,
    int sphere_resolution)
{
    std::vector<std::shared_ptr<const open3d::geometry::Geometry>> geoms;

    if (ellipsoids.empty()) {
        std::cout << "No ellipsoids to display.\n";
        return;
    }

    auto color_for_type = [](const std::string& type) -> Eigen::Vector3d {
        if (type == "roi_surface_frontier") return {1.0, 0.0, 0.0}; // red
        if (type == "frontier")            return {1.0, 1.0, 0.0}; // yellow
        if (type == "occupied")            return {0.0, 0.0, 1.0}; // blue
        return {0.7, 0.7, 0.7}; // fallback gray
    };

    for (const auto& e : ellipsoids)
    {
        // Base sphere
        auto sphere = open3d::geometry::TriangleMesh::CreateSphere(1.0, sphere_resolution);
        sphere->ComputeVertexNormals();

        // Scale sphere into ellipsoid (radii are semi-axes)
        sphere->Scale(e.radii.x(), Eigen::Vector3d(0,0,0)); // scale uniformly then anisotropic? no
        // Open3D TriangleMesh::Scale is uniform only, so do anisotropic scaling via Transform vertices:
        for (auto &v : sphere->vertices_) {
            v(0) *= e.radii.x();
            v(1) *= e.radii.y();
            v(2) *= e.radii.z();
        }

        // Apply pose (rotation + translation)
        sphere->Transform(e.pose);

        // Color
        sphere->PaintUniformColor(color_for_type(e.type));

        if (wireframe) {
            // Convert to LineSet for wireframe display
            auto ls = open3d::geometry::LineSet::CreateFromTriangleMesh(*sphere);
            ls->PaintUniformColor(color_for_type(e.type));
            geoms.push_back(ls);
        } else {
            geoms.push_back(sphere);
        }
    }

    open3d::visualization::DrawGeometries(
        geoms,
        "Ellipsoids (Frontier/ROI/Occupied)",
        1400,
        900
    );
}
