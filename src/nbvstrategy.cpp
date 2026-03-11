#include "nbvstrategy.hpp"
#include "structs.hpp"
#include "utils.hpp"
#include "cuda_projection.hpp"

int nbvstrategy::initialize(std::string settings_path) 
{
    std::ifstream file(settings_path);
    if(!file.is_open()) {
        std::cerr << "Failed to load settings";
        return -1;
    }

    json cfg;
    file >> cfg;

    //CAMERA INTRINSICS
    auto cam = cfg["camera"];

    int width = getOrDefault<int>(cam, "width", 0);
    int height = getOrDefault<int>(cam, "height", 0);
    double fx = getOrDefault<double>(cam, "fx", 0);
    double fy = getOrDefault<double>(cam, "fy", 0);
    double cx = getOrDefault<double>(cam, "cx", 0);
    double cy = getOrDefault<double>(cam, "cy", 0);
    double min_range = getOrDefault<double>(cam, "min_range", 0);
    double max_range = getOrDefault<double>(cam, "max_range", 0);

    std::cout << "Camera loaded!" << std::endl;
    std::cout << "Parameters: \n" <<
        "Width: " << width << "\n" <<
        "Heigh: " << height << "\n" <<
        "fx: " << fx << "\n" <<
        "fy: " << fy << "\n" <<
        "cx: " << cx << "\n" <<
        "cy: " << cy << "\n" <<
        "Minimum Range: " << min_range << "\n" <<
        "Maximum Range: " << max_range << "\n" << std::endl;

    this->cam_intrinsics.width = width;
    this->cam_intrinsics.height = height;
    this->cam_intrinsics.fx = fx;
    this->cam_intrinsics.fy = fy;
    this->cam_intrinsics.cx = cx;
    this->cam_intrinsics.cy = cy;
    this->cam_intrinsics.max_range = max_range;

    //BBX
    auto bbx = cfg["bounding_box"];
    std::vector<double> bbx_min = getVectorOrEmpty(bbx, "bbx_min");
    std::vector<double> bbx_max = getVectorOrEmpty(bbx, "bbx_max");

    std::cout << "Bounding Box loaded!" << std::endl;
    std::cout << "Parameters: \n" <<
        "Minimum Bounding Box (x,y,z): " << bbx_min[0] << ", " << bbx_min[1] << ", " << bbx_min[2] << "\n" <<
        "Maximum Bounding Box (x,y,z): " << bbx_max[0] << ", " << bbx_max[1] << ", " << bbx_max[2] << std::endl << std::endl;      

    if (bbx_min.size() != 3 || bbx_max.size() != 3) throw std::runtime_error("bbx_min or bbx_max must have 3 elements");
    Eigen::Vector3d bbx_min_eigen(bbx_min[0],bbx_min[1],bbx_min[2]);
    this->bbx_min = bbx_min_eigen;
    Eigen::Vector3d bbx_max_eigen(bbx_max[0],bbx_max[1],bbx_max[2]);
    this->bbx_max = bbx_max_eigen;

    //PLANT BBX
    std::vector<PlantBBX> plant_bbx_list;
    auto plants_region = cfg["plants_region"];
    for (auto& p : plants_region["plants"]) {
        PlantBBX pb;
        pb.min = getVec3OrDefault(p, "min");
        pb.max = getVec3OrDefault(p, "max");
        plant_bbx_list.push_back(pb);
    }

    std::cout << "Plants Bounding Boxes loaded! " << std::endl;
    for (auto& p : plant_bbx_list) {
        std::cout << "Plant Bounding Box (x,y,z): " << p.min[0] << ", " << p.min[1] << ", " << p.min[2] << std::endl;
        std::cout << "Plant Bounding Box (x,y,z): " << p.max[0] << ", " << p.max[1] << ", " << p.max[2] << std::endl << std::endl;
    }

    this->bbx_plants = plant_bbx_list;

    //GMM CLUSTERS
    auto clustering = cfg["clustering"];
    int min_clusters = getOrDefault(clustering, "min_clusters", 2);
    int max_clusters = getOrDefault(clustering, "max_clusters", 10);

    std::cout << "Cluster sized loaded! " << std::endl;
    std::cout << "Minimum Clusters: " << min_clusters << std::endl;
    std::cout << "Maximum Clusters: " << max_clusters << std::endl << std::endl;

    this->min_clusters = min_clusters;
    this->max_clusters = max_clusters;

    //OCTOMAP RESOLUTION
    auto octomap = cfg["octomap"];
    double resolution = getOrDefault(octomap, "resolution",  0.05);

    std::cout << "Octomap resolution loaded! " << std::endl;
    std::cout << "Resolution: " << resolution << std::endl << std::endl;

    this->resolution = resolution;

    //VIEWPOINT GENERATION FREQUENCY
    auto xyzypgenf = cfg["xyzypgenf"];
    double dx = getOrDefault(xyzypgenf, "dx", 0.05);
    double dy = getOrDefault(xyzypgenf, "dy", 0.05);
    double dz = getOrDefault(xyzypgenf, "dz", 0.05);
    double dyaw = getOrDefault(xyzypgenf, "dyaw", 4);
    double dpitch = getOrDefault(xyzypgenf, "dpitch", 4);

    std::cout << "Viewpoint Generation Frequency loaded!" << std::endl;
    std::cout << "dx: " << dx << std::endl;
    std::cout << "dy: " << dy << std::endl;
    std::cout << "dz: " << dz << std::endl;
    std::cout << "dyaw: " << dyaw << std::endl;
    std::cout << "dpitch: " << dpitch << std::endl << std::endl;

    this->dx = dx;
    this->dy = dy;
    this->dz = dz;
    this->dyaw = M_PI/dyaw;
    this->dpitch = M_PI/dpitch;

    this->voxel_struct = new voxelstruct(this->resolution);
    this->ellipsoid_fitting = new ellipsoid(this->min_clusters, this->max_clusters);

    return 1;
}

//merged plant checker and viewpoint creator to one
void nbvstrategy::generateViewpoints() 
{
    auto isInsideAnyPlantBBX = [this](double x, double y, double z) -> bool
    {
        for (const auto& bbx : this->bbx_plants) {
            if (x >= bbx.min(0) && x <= bbx.max(0) &&
                y >= bbx.min(1) && y <= bbx.max(1) &&
                z >= bbx.min(2) && z <= bbx.max(2))
                return true;
        }
        return false;
    };

    size_t nx = (this->bbx_max[0] - this->bbx_min[0])/this->dx + 1;
    // need to hardcode this part since the camera cant just be on the ground, but points can be in the bbx
    size_t ny = (1.005 - 0.49)/this->dy;
    size_t nz = (this->bbx_max[2] - this->bbx_min[2])/this->dz + 1;
    size_t nyaw = (M_PI/2)/this->dyaw+1;
    size_t npitch = (M_PI/2)/this->dpitch+1;
    double roll = 0.0;

    size_t total = nx*ny*nz*nyaw*npitch+5;
    this->viewpoints.clear();
    this->viewpoints.reserve(total);

    #pragma omp parallel
    {
        std::vector<Eigen::Matrix<double, 6, 1>> local_views;

        #pragma omp for nowait
        for (size_t ix = 0; ix < nx; ++ix) {
            double x = this->bbx_min[0] + (double)ix * this->dx;
            for (double y = 0.45; y <= 1.005; y += this->dy) {
                for (double z = this->bbx_min[2]; z <= this->bbx_max[2]; z += this->dz) {
                    if (isInsideAnyPlantBBX(x,y,z)) continue; 
                    for (double pitch = -M_PI/2; pitch < M_PI/2; pitch+= this->dpitch) {
                        for (double yaw = -M_PI/2; yaw < M_PI/2; yaw += this->dyaw) {
                            Eigen::Matrix<double,6,1> vp;
                            vp << x, y, z, pitch, yaw, roll;
                            local_views.push_back(vp);

                        }
                    }
                }
            }
        }

        #pragma omp critical
        this->viewpoints.insert(this->viewpoints.end(), local_views.begin(), local_views.end());
    }

    std::cout << "Viewpoints Generated!" << std::endl;
    std::cout << "Number of viewpoints: " << this->viewpoints.size() << std::endl;
    std::cout << "First viewpoint: " << this->viewpoints.front().transpose() << std::endl;
    std::cout << "Last viewpoint: " << this->viewpoints.back(). transpose() << std::endl;
}

void nbvstrategy::kill() 
{
    if (voxel_struct) {
        this->voxel_struct->killVoxelStruct();
        delete voxel_struct;
        voxel_struct = nullptr;
    }
    if (ellipsoid_fitting) {
        delete ellipsoid_fitting;
        ellipsoid_fitting = nullptr;
    }
}

void nbvstrategy::insertTransformedCloud(std::string file_path, 
    Eigen::Vector3d vp)
{
    open3d::geometry::PointCloud* pcd = nullptr;
    pcd = new open3d::geometry::PointCloud();
    bool ok = open3d::io::ReadPointCloud(file_path,*pcd);
    if (!ok) {
        std::cout << "Failed to load point cloud: " << file_path << "\n";
        delete pcd;
        pcd = nullptr;
        return;
    }

    std::cout << "Loaded point cloud: " << file_path << "\n";
    std::cout << "Points: " << pcd->points_.size() << "\n";

    this->voxel_struct->insertPointCloud(pcd,vp);

    return;
}

bool nbvstrategy::SphereInFrustum(
        const Eigen::Vector3d &center_cam,
        double rx, double ry, double rz,
        const Eigen::Matrix3d &R_e_cam,
        const Camera &cam)
{
    // Compute bounding sphere in camera coordinates
    Eigen::Vector3d axis_x = R_e_cam.col(0) * rx;
    Eigen::Vector3d axis_y = R_e_cam.col(1) * ry;
    Eigen::Vector3d axis_z = R_e_cam.col(2) * rz;
    double R = std::max({axis_x.norm(), axis_y.norm(), axis_z.norm()});

    double x = center_cam.x();
    double y = center_cam.y();
    double z = center_cam.z();

    // Depth rejection
    if (z + R <= 0) return false;
    if (z - R > this->cam_intrinsics.max_range) return false;
    if (z <= 1e-6) return false;

    // Project center to pixels
    double u_center = cam.fx * (x / z) + cam.cx;
    double v_center = cam.fy * (y / z) + cam.cy;

    // Project sphere radius
    double u_rad = cam.fx * (R / z);
    double v_rad = cam.fy * (R / z);

    double u_min = u_center - u_rad;
    double u_max = u_center + u_rad;
    double v_min = v_center - v_rad;
    double v_max = v_center + v_rad;

    // Cull if completely outside image
    if (u_max < 0) return false;
    if (u_min >= cam.width) return false;
    if (v_max < 0) return false;
    if (v_min >= cam.height) return false;

    return true;
}

Eigen::Matrix4d nbvstrategy::
create_ellipsoid_dual_matrix(const EllipsoidParam &param)
{
    
    Eigen::Matrix4d matrix = Eigen::Matrix4d::Zero(); 
    Eigen::Vector3d radii_pow = param.radii.array().square(); 
    Eigen::Vector3d radii_inv = radii_pow.array().inverse();
    Eigen::Matrix4d transformation = param.pose; 

    // 将radii_inv设置为matrix的前3x3的对角矩阵
    matrix.block<3,3>(0,0) = radii_inv.asDiagonal();
    matrix(3, 3) = -1;

    double det = matrix.determinant(); 
    if (det == 0) {
        std::cout << "The determinant of the matrix is 0, the matrix is not invertible." << std::endl;
        return Eigen::Matrix4d::Zero(); 
    } else {
        Eigen::Matrix4d matrix_dual_origin = matrix.inverse(); 
        // std::cout << "Matrix Dual Origin:\n" << matrix_dual_origin << std::endl;
        Eigen::Matrix4d matrix_dual = transformation * matrix_dual_origin * transformation.transpose(); // 计算最终的矩阵
        // std::cout << "Matrix Dual:\n" << matrix_dual << std::endl;
        return matrix_dual;
    }
}

Eigen::Matrix3d nbvstrategy::
compute_ellipsoid_projection(
    const Eigen::Matrix<double, 3, 4>& camera_matrix,
    const Eigen::Matrix4d& ellipsoid_matrix_dual)
{
    

    Eigen::Matrix3d ellipse_dual = camera_matrix * 
                                    ellipsoid_matrix_dual * 
                                    camera_matrix.transpose();

    double det = ellipse_dual.determinant(); 
    if (std::abs(det) < 1e-12) {
        std::cout << "The determinant of the matrix is 0, the matrix is not invertible." << std::endl;
        return Eigen::Matrix3d::Zero(); 
    } else {
        Eigen::Matrix3d ellipse = ellipse_dual.inverse();
        return ellipse;
    }

}

std::pair<double, cv::Mat> nbvstrategy::projectEllipsoidstoImage(
        const std::vector<EllipsoidParam> &ellipsoids,
        const Eigen::Matrix4d &T_cam_world) 
{
    // T_cam_world is expected to be the pose of the viewpoint
    Eigen::Matrix4d T_world_cam = T_cam_world.inverse();
    Eigen::Matrix3d R_wc = T_world_cam.block<3,3>(0,0);
    Eigen::Vector3d t_wc = T_world_cam.block<3,1>(0,3);
    
    // sanity reminder, don't need the camera matrix here yet since
    // transformations from 3d to 3d don't require intrinsic parameters
    // however projections for example 3d to 2d do required it 
    // so that will be done later
    std::vector<Eigen::Vector3d> centers(ellipsoids.size());
    #pragma omp parallel for
    for (size_t i = 0; i < ellipsoids.size(); i++) {
        Eigen::Vector3d center_world = ellipsoids[i].pose.block<3,1>(0,3);
        centers[i] = R_wc * center_world + t_wc; 
        // this will put it into the camera's coordinates (e.g. if the camera was 0,0)
    }

    // weighting the centers, reminder that 
    // still don't need intrinsic parameters for this
    // since its already in the camera's coordinates, we can use the z axis to weigh them
    std::vector<size_t> idx(centers.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
        [&centers](size_t i1, size_t i2) { return centers[i1].z() < centers[i2].z(); });

    // assigning the weights
    std::vector<double> weights(centers.size());
    for (size_t i = 0; i < idx.size(); i++) {
        weights[idx[i]] = pow(0.5, static_cast<double>(i));
    }
    
    // building camera matrix
    // CHECK ON THIS LATER
    Eigen::Matrix3d K;
    K << this->cam_intrinsics.fx, 0, this->cam_intrinsics.cx,
         0,      this->cam_intrinsics.fy, this->cam_intrinsics.cy,
         0,      0,      1;
    Eigen::Matrix<double,3,4> Rt_inv;
    Rt_inv.block<3,3>(0,0) = R_wc;
    Rt_inv.block<3,1>(0,3) = t_wc;
    Eigen::Matrix<double,3,4> P = K * Rt_inv;  

    // setting up image and ellipse vectors
    std::vector<double> a_vec(ellipsoids.size());
    std::vector<double> b_vec(ellipsoids.size());
    std::vector<double> c_vec(ellipsoids.size());
    std::vector<double> d_vec(ellipsoids.size());
    std::vector<double> e_vec(ellipsoids.size());
    std::vector<double> f_vec(ellipsoids.size());
    
    // projecting the ellipsoids
    // now this part will use the camera matrix
    // sanity reminder that the center transformations already put it in the camera world
    for (size_t k = 0; k < ellipsoids.size(); k++) {
        size_t i = idx[k];

        Eigen::Matrix3d R_e_world = ellipsoids[i].pose.block<3,3>(0,0);
        Eigen::Matrix3d R_e_cam = R_wc * R_e_world;
        
        double rx = ellipsoids[i].radii.x();
        double ry = ellipsoids[i].radii.y();
        double rz = ellipsoids[i].radii.z();
        
        //cull ellipsoids first too
        if (!SphereInFrustum(centers[i], rx, ry, rz, R_e_cam, this->cam_intrinsics)) {
            a_vec[i] = 0;
            b_vec[i] = 0;
            c_vec[i] = 0;
            d_vec[i] = 0;
            e_vec[i] = 0;
            f_vec[i] = 0;
            continue; // skip this ellipsoid
        }
        
        Eigen::Matrix4d dual = this->create_ellipsoid_dual_matrix(ellipsoids[i]);
        if (dual.isZero(1e-12))
        {
            a_vec[i] = 0;
            b_vec[i] = 0;
            c_vec[i] = 0;
            d_vec[i] = 0;
            e_vec[i] = 0;
            f_vec[i] = 0;
            continue;
        }
        
        Eigen::Matrix3d ellipse_matrix = this->compute_ellipsoid_projection(P, dual);
        if (ellipse_matrix.isZero(1e-12))
        {
            a_vec[i] = 0;
            b_vec[i] = 0;
            c_vec[i] = 0;
            d_vec[i] = 0;
            e_vec[i] = 0;
            f_vec[i] = 0;
            continue;
        }
        
        a_vec[i] = ellipse_matrix(0, 0);
        b_vec[i] = ellipse_matrix(1, 1);
        c_vec[i] = ellipse_matrix(0, 1) + ellipse_matrix(1, 0);
        d_vec[i] = ellipse_matrix(0, 2) + ellipse_matrix(2, 0);
        e_vec[i] = ellipse_matrix(1, 2) + ellipse_matrix(2, 1);
        f_vec[i] = ellipse_matrix(2, 2); 
    }
    
    // okay so we now have the centers, and the ellipse matrix coefficients
    // (a,b,c,d,e,f) for the ellipse equation
    // time to "project it"
    cv::Mat img;
    double frontier_res = 0.0;
    double roi_surface_frontier_res = 0.0;
    double occupied_res = 0.0;

    // Convert ellipsoid types once
    std::vector<int> type_ids(ellipsoids.size());
    for (size_t k = 0; k < ellipsoids.size(); k++) {
        size_t i = idx[k];
        if (ellipsoids[i].type == "frontier")
            type_ids[i] = 0;
        else if (ellipsoids[i].type == "roi_surface_frontier")
            type_ids[i] = 1;
        else
            type_ids[i] = 2;
    }

    project_pixels_cuda(
        this->cam_intrinsics.width,
        this->cam_intrinsics.height,
        a_vec, b_vec, c_vec, d_vec, e_vec, f_vec,
        type_ids,
        weights,
        img,
        frontier_res,
        roi_surface_frontier_res,
        occupied_res
    );

    double score = 2 * roi_surface_frontier_res + frontier_res - occupied_res;
    return std::make_pair(score, img);
}   

void nbvstrategy::getNBV()
{
    this->best_score = 0;
    this->best_image = cv::Mat(this->cam_intrinsics.height, this->cam_intrinsics.width, CV_8UC3, cv::Scalar(0,0,0));
    this->best_viewpoint = Eigen::Matrix<double,6,1>(0.0,0.0,0.0,0.0,0.0,0.0);
    this->best_viewpoint_index = 0;

    //voxel classification
    this->voxel_struct->showVoxelTree();
    this->voxel_struct->classifyVoxels();
    std::vector<Eigen::Vector3d> surface_frontiers = this->voxel_struct->getSurfaceFrontiers();
    std::vector<Eigen::Vector3d> occupied_voxels = this->voxel_struct->getOccupiedVoxels();
    std::vector<Eigen::Vector3d> roi_surface_frontier = this->voxel_struct->getROISurfaceFrontiers();

    std::cout << "\nVoxels Classified!" << std::endl;
    std::cout << "Number of frontier voxels: " << surface_frontiers.size() << std::endl;
    std::cout << "Number of occupied voxels: " << occupied_voxels.size() << std::endl;
    std::cout << "Number of roi frontier voxels: " << roi_surface_frontier.size() << std::endl;

    this->voxel_struct->showClassifiedVoxels();
    
    //clustering
    std::vector<std::vector<Eigen::Vector3d>> frontier_clusters = this->ellipsoid_fitting->gmm_clustering(surface_frontiers);
    std::vector<std::vector<Eigen::Vector3d>> occupied_clusters = this->ellipsoid_fitting->gmm_clustering(occupied_voxels);
    std::vector<std::vector<Eigen::Vector3d>> roi_surface_frontier_clusters = this->ellipsoid_fitting->gmm_clustering(roi_surface_frontier);
    
    std::cout << "\nVoxels Clustered!" << std::endl;
    std::cout << "Number of frontier clusters: " << frontier_clusters.size() << std::endl;
    std::cout << "Number of occupied clusters: " << occupied_clusters.size() << std::endl;
    std::cout << "Number of roi frontier clusters: " << roi_surface_frontier_clusters.size() << std::endl;    

    //ellipsoid fitting
    this->ellipsoid_fitting->showAllClustersColored(
        frontier_clusters,
        occupied_clusters,
        roi_surface_frontier_clusters,
        this->resolution  // voxel size
    );

    std::vector<EllipsoidParam> ellipsoids = this->ellipsoid_fitting->ellipsoidize_clusters_CGAL(
        frontier_clusters,
        occupied_clusters,
        roi_surface_frontier_clusters
    ); 

    std::cout << "\nEllipsoids fitted!" << std::endl;
    std::cout << "Number of Ellipsoids: " << ellipsoids.size() << std::endl;
    
    this->ellipsoid_fitting->showEllipsoidsOpen3D(ellipsoids, false); // solid meshes

    std::string input;
    std::cout << "Continue? (Y/n): ";
    std::cin >> input;
    input.erase(0, input.find_first_not_of(" \t\n\r"));
    input.erase(input.find_last_not_of(" \t\n\r") + 1);
    std::transform(input.begin(), input.end(), input.begin(), ::tolower);
    if (input == "y" || input == "yes" || input == "Y");
    if (input == "n" || input == "no" || input == "N") return;

    //projection
    #pragma omp parallel for
    for (size_t i = 0; i < this->viewpoints.size(); i++) {
        auto T_cam_world = this->transform.getCameraPose(this->viewpoints[i]);
        std::pair<double, cv::Mat> score_and_img = this->projectEllipsoidstoImage(ellipsoids,T_cam_world);

        double score = score_and_img.first;
        cv::Mat img = score_and_img.second;
        
        std::cout << "Viewpoint [" << i << "] : " << score << std::endl;

        #pragma omp critical
        {
            if (score > best_score) {
                this->best_score = score;
                this->best_image = img.clone();
                this->best_viewpoint = this->viewpoints[i]; 
                this->best_viewpoint_index = i;
            }
        }
    }

    if (best_viewpoint_index < this->viewpoints.size()) {
        this->viewpoints.erase(this->viewpoints.begin() + best_viewpoint_index);
    }

    std::cout << "\nBest viewpoint: " << this->best_viewpoint.transpose() << std::endl;
    std::cout << "Best viewpoint score: " << this->best_score << std::endl;
    //Display the best viewpoint image
    // cv::imshow("Best Viewpoint", best_image);
    cv::imwrite("best_viewpoint.png", best_image);
}

bool nbvstrategy::hasBestNBV() const
{
    return (this->best_score > 0.0 || !this->best_image.empty());
}

void nbvstrategy::showLastNBVInfo() const
{
    if (!hasBestNBV()) {
        std::cout << "No NBV has been computed yet.\n";
        return;
    }

    std::cout << "\n--- LAST NBV ---\n";
    std::cout << "Score: " << this->best_score << "\n";
    std::cout << "Viewpoint (x,y,z,pitch,yaw,roll): "
              << this->best_viewpoint.transpose() << "\n";
    std::cout << "Index: " << this->best_viewpoint_index << "\n";
}

bool nbvstrategy::saveLastNBV(const std::string& txt_path, const std::string& img_path) const
{
    if (!hasBestNBV()) {
        std::cout << "No NBV has been computed yet.\n";
        return false;
    }

    std::ofstream out(txt_path);
    if (!out.is_open()) {
        std::cout << "Failed to open text output file: " << txt_path << "\n";
        return false;
    }

    out << "best_score: " << this->best_score << "\n";
    out << "best_viewpoint_index: " << this->best_viewpoint_index << "\n";
    out << "best_viewpoint_x_y_z_pitch_yaw_roll: "
        << this->best_viewpoint.transpose() << "\n";
    out.close();

    bool ok_img = false;
    if (!this->best_image.empty()) {
        ok_img = cv::imwrite(img_path, this->best_image);
    }

    if (!ok_img) {
        std::cout << "Failed to save NBV image: " << img_path << "\n";
        return false;
    }

    std::cout << "Saved NBV text to: " << txt_path << "\n";
    std::cout << "Saved NBV image to: " << img_path << "\n";
    return true;
}

void nbvstrategy::showLastNBVInVoxelTree()
{
    if (!hasBestNBV()) {
        std::cout << "No NBV has been computed yet.\n";
        return;
    }
    if (!this->voxel_struct) {
        std::cout << "Voxel structure not initialized.\n";
        return;
    }

    // Make sure occupied voxels exist
    this->voxel_struct->classifyVoxels();
    std::vector<Eigen::Vector3d> occ = this->voxel_struct->getOccupiedVoxels();

    voxelstruct temp_vs(this->resolution);

    // 1) Insert occupied voxels as blue
    auto occ_cloud = new open3d::geometry::PointCloud();
    occ_cloud->points_ = occ;
    occ_cloud->colors_.resize(occ.size(), Eigen::Vector3d(0.0, 0.0, 1.0));

    // camera origin for ray insertion; any fixed point is fine for this temp viz
    Eigen::Vector3d fake_cam(0.0, 0.0, 0.0);
    temp_vs.insertPointCloud(occ_cloud, fake_cam);

    // 2) Insert best viewpoint position as red
    auto nbv_cloud = new open3d::geometry::PointCloud();
    nbv_cloud->points_.push_back(this->best_viewpoint.block<3,1>(0,0));
    nbv_cloud->colors_.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));

    // offset the fake camera slightly so the ray endpoint is valid
    Eigen::Vector3d nbv_cam = this->best_viewpoint.block<3,1>(0,0) + Eigen::Vector3d(this->resolution, 0, 0);
    temp_vs.insertPointCloud(nbv_cloud, nbv_cam);

    temp_vs.showVoxelTree();
    temp_vs.killVoxelStruct();
}