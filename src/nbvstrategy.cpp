#include "nbvstrategy.hpp"
#include "structs.hpp"
#include "utils.hpp"

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
        this->viewpoints.clear();
        this->viewpoints.reserve(total);
        this->viewpoints.insert(this->viewpoints.end(), local_views.begin(), local_views.end());
    }
}

void nbvstrategy::kill() 
{
    this->voxel_struct->killVoxelStruct();
    if (voxel_struct) {
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
    open3d::geometry::PointCloud* pcd;
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

    delete pcd;
    return;
}

void nbvstrategy::getNBV()
{
    this->best_score = 0;
    this->best_image = cv::Mat(this->cam_intrinsics.height, this->cam_intrinsics.width, CV_8UC3, cv::Scalar(0,0,0));
    this->best_viewpoint = Eigen::Matrix<double,6,1>(0.0,0.0,0.0,0.0,0.0,0.0);
    this->best_viewpoint_index = 0;
}