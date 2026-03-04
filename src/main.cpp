#include <iostream>
#include <string>
#include "nbvtransform.hpp"
#include <cmath>
#include "voxelstruct.hpp"

//random helper functions
static bool saveVec3AsPLY(const std::string& path, const std::vector<Eigen::Vector3d>& pts) {
    open3d::geometry::PointCloud cloud;
    cloud.points_ = pts;              // copies points
    // no colors needed for ellipsoid fitting; add if you want
    bool ok = open3d::io::WritePointCloud(path, cloud);
    if (!ok) std::cout << "Failed to save: " << path << "\n";
    else     std::cout << "Saved: " << path << " (" << pts.size() << " pts)\n";
    return ok;
}

static bool parseVec3(const std::string& s, Eigen::Vector3d& out) {
    std::string t = s;
    for (char& ch : t) if (ch == ',') ch = ' ';
    std::istringstream iss(t);

    double a, b, c;
    if (!(iss >> a >> b >> c)) return false;
    out = Eigen::Vector3d(a, b, c);
    return true;
}

static bool writeVec3ListToFile(
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

static std::string prompt(const std::string& msg) {
    std::cout << msg;
    std::string s;
    std::getline(std::cin, s);
    return s;
}

static bool parseViewpoint6(const std::string& s, Eigen::Matrix<double,6,1>& out) {
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

// ---------- Menus ----------
static void printMainMenu() {
    std::cout << "\n=== MAIN MENU ===\n"
              << "1) Next Best View\n"
              << "2) Transforms\n"
              << "3) Voxelstruct\n"
              << "4) Ellipsoid Fitting\n"
              << "0) Exit\n";
}

static void printNBVMenu() {
    std::cout << "\n--- NEXT BEST VIEW ---\n"
              << "1) Option 1\n"
              << "2) Option 2\n"
              << "0) Back\n";
}

static void printTransformMenu() 
{
    std::cout << "\n--- TRANSFORMS ---\n"
              << "1) Load PCD\n"
              << "2) Translate to world frame\n"
              << "3) Switch XYZ\n" 
              << "4) Crop BBX\n"
              << "5) View PCD\n"
              << "6) Save PCD\n"
              << "7) Print all points to File\n"
              << "0) Back\n";
}

static void printVoxelMenu() 
{
    std::cout << "\n--- VOXELSTRUCT ---\n"
              << "1) Initialize voxelstruct\n"
              << "2) Insert pointcloud\n"
              << "3) Show voxel tree\n"
              << "4) Print voxels to file\n"
              << "5) Print Surface Frontiers, Occupied, and ROI Surface Frontier voxels to file\n"
              << "6) Save octree\n"
              << "7) Show classified voxels (occ/frontier/roi)\n"
              << "8) Save point cloud\n"
              << "0) Back\n";
}

static void printEllipsoidMenu() {
    std::cout << "\n--- ELLIPSOID FITTING ---\n"
              << "1) Option 1\n"
              << "2) Option 2\n"
              << "0) Back\n";
}

// ---------- Submenu loops ----------
static void nextBestViewMenu() {
    bool inMenu = true;
    while (inMenu) {
        printNBVMenu();
        std::string c = prompt("Choice: ");

        if (c == "0") inMenu = false;
        else if (c == "1") { /* TODO */ }
        else if (c == "2") { /* TODO */ }
        else std::cout << "Invalid choice.\n";
    }
}

static void transformsMenu() 
{
    bool inMenu = true;

    // Avoid heap alloc / leaks
    nbvtransform transform;
    transform.pcd = nullptr;

    bool pcd_loaded = false;

    while (inMenu) {
        printTransformMenu();
        std::string c = prompt("Choice: ");

        if (c == "0") {
            transform.killNBVTransform();
            inMenu = false;
        }
        else if (c == "1") {
            std::string path = prompt("Enter point cloud path: ");
            transform.loadPCD(path);
            pcd_loaded = (transform.pcd != nullptr);
            if (!pcd_loaded) std::cout << "Failed to load PCD.\n";
            else std::cout << "PCD loaded.\n";
        }
        else if (c == "2") {
            // Translate to world frame
            if (transform.pcd == nullptr || pcd_loaded == false) {
                std::cout << "Load point cloud first.\n";
                continue;
            }

            std::string vp_str = prompt("Enter x,y,z,yaw,pitch,roll: ");
            Eigen::Matrix<double,6,1> vp;
            if (!parseViewpoint6(vp_str, vp)) {
                std::cout << "Invalid input. Expected 6 numbers.\n";
                continue;
            }

            Eigen::Matrix4d T_cam_world = transform.getCameraPose(vp);

            // Your function returns a *new* point cloud
            open3d::geometry::PointCloud pcd_world =
                transform.translateToWorldFrame(T_cam_world, transform.pcd);

            // Replace the currently loaded pcd contents with the transformed one
            *(transform.pcd) = std::move(pcd_world);

            std::cout << "Translated to world frame.\n";
        }
        else if (c == "3") {
            // Switch XYZ
            if (transform.pcd == nullptr || pcd_loaded == false) {
                std::cout << "Load point cloud first.\n";
                continue;
            }

            std::string axes = prompt("Enter 3 chars (x/y/z or a/b/c), e.g. a b z: ");

            // Allow "abz" or "a b z" or "a,b,z"
            for (char& ch : axes) if (ch == ',') ch = ' ';
            std::istringstream iss(axes);

            char cx = 0, cy = 0, cz = 0;
            if (!(iss >> cx >> cy >> cz)) {
                // try compact form like "abz"
                if (axes.size() >= 3) {
                    cx = axes[0];
                    cy = axes[1];
                    cz = axes[2];
                } else {
                    std::cout << "Invalid input. Example: a b z (or abz)\n";
                    continue;
                }
            }

            transform.switchXYZ(cx, cy, cz, transform.pcd);
            std::cout << "Switched axes.\n";
        }
        else if (c == "4") {
            if (!pcd_loaded || transform.pcd == nullptr) {
                std::cout << "Load point cloud first.\n";
                continue;
            }

            std::string minStr = prompt("Enter bbx_min (x,y,z): ");
            std::string maxStr = prompt("Enter bbx_max (x,y,z): ");

            Eigen::Vector3d bbx_min, bbx_max;
            if (!parseVec3(minStr, bbx_min) || !parseVec3(maxStr, bbx_max)) {
                std::cout << "Invalid input. Expected 3 numbers for each.\n";
                continue;
            }

            // Optional safety: ensure min <= max per-axis
            Eigen::Vector3d mn = bbx_min.cwiseMin(bbx_max);
            Eigen::Vector3d mx = bbx_min.cwiseMax(bbx_max);

            const size_t before = transform.pcd->points_.size();
            transform.cropBBX(mn, mx, transform.pcd);
            const size_t after = transform.pcd->points_.size();

            std::cout << "Cropped BBX. Points: " << before << " -> " << after << "\n";
        }
        else if (c == "5") {
            if (!pcd_loaded || transform.pcd == nullptr) {
                std::cout << "Load point cloud first.\n";
                continue;
            }
            transform.viewPCD("Transforms: PointCloud");
        }
        else if (c == "6") {
            if (!pcd_loaded || transform.pcd == nullptr) {
                std::cout << "Load point cloud first.\n";
                continue;
            }

            std::string path = prompt("Enter output filename (without extension ok): ");
            transform.savePCD(path);
        }
        else if (c == "7") {
            if (!pcd_loaded || transform.pcd == nullptr) {
                std::cout << "Load point cloud first.\n";
                continue;
            }
            std::string path = prompt("Enter output filename: ");
            transform.printAllPointsToFile(path);
        }
        else {
            std::cout << "Invalid choice.\n";
        }
    }
}

static void voxelstructMenu() 
{
    bool inMenu = true;

    voxelstruct* voxel_struct = nullptr;

    // Keep last loaded point cloud alive as long as the menu is open
    std::shared_ptr<open3d::geometry::PointCloud> last_pcd;

    while (inMenu) {
        printVoxelMenu();
        std::string c = prompt("Choice: ");

        if (c == "0") {
            if (voxel_struct) {
                voxel_struct->killVoxelStruct();
                delete voxel_struct;
                voxel_struct = nullptr;
            }
            last_pcd.reset();
            std::cout << "voxelstruct memory freed\n";
            inMenu = false;
        }
        else if (c == "1") {
            // Initialize voxelstruct
            std::string r = prompt("Enter resolution (e.g. 0.05): ");
            double res = 0.0;
            try { res = std::stod(r); }
            catch (...) { std::cout << "Invalid resolution.\n"; continue; }

            // re-init if already exists
            if (voxel_struct) {
                voxel_struct->killVoxelStruct();
                delete voxel_struct;
                voxel_struct = nullptr;
            }

            voxel_struct = new voxelstruct(res);
            std::cout << "Initialized voxelstruct with resolution " << res << "\n";
        }
        else if (c == "2") {
            // Insert pointcloud
            if (!voxel_struct) {
                std::cout << "Initialize voxelstruct first.\n";
                continue;
            }

            std::string path = prompt("Enter point cloud path: ");
            last_pcd = std::make_shared<open3d::geometry::PointCloud>();

            if (!open3d::io::ReadPointCloud(path, *last_pcd) || last_pcd->points_.empty()) {
                std::cout << "Failed to load point cloud or empty: " << path << "\n";
                last_pcd.reset();
                continue;
            }

            std::string camStr = prompt("Enter camera position x,y,z: ");
            Eigen::Vector3d cam;
            if (!parseVec3(camStr, cam)) {
                std::cout << "Invalid camera input.\n";
                last_pcd.reset();
                continue;
            }

            voxel_struct->insertPointCloud(last_pcd.get(), cam);
            std::cout << "Inserted point cloud into octree.\n";
        }
        else if (c == "3") {
            // Show voxel tree
            if (!voxel_struct) {
                std::cout << "Initialize voxelstruct first.\n";
                continue;
            }
            voxel_struct->showVoxelTree();
        }
        else if (c == "4") {
            // Print occupied voxels to file (ensures classify is run)
            if (!voxel_struct) {
                std::cout << "Initialize voxelstruct first.\n";
                continue;
            }

            voxel_struct->classifyVoxels();
            auto occ = voxel_struct->getOccupiedVoxels();

            std::string outPath = prompt("Enter output filename: ");
            writeVec3ListToFile(outPath, occ, "# index x y z");
        }
        else if (c == "5") {
            // Print surface frontiers / occupied / ROI frontiers to files
            if (!voxel_struct) {
                std::cout << "Initialize voxelstruct first.\n";
                continue;
            }

            voxel_struct->classifyVoxels();

            auto sf  = voxel_struct->getSurfaceFrontiers();
            auto occ = voxel_struct->getOccupiedVoxels();
            auto roi = voxel_struct->getROISurfaceFrontiers();

            std::string base = prompt("Enter base filename (no extension): ");
            if (base.empty()) base = "voxels";

            writeVec3ListToFile(base + "_surface_frontiers.txt", sf,  "# surface_frontiers: index x y z");
            writeVec3ListToFile(base + "_occupied.txt",          occ, "# occupied_voxels: index x y z");
            writeVec3ListToFile(base + "_roi_frontiers.txt",     roi, "# roi_surface_frontier: index x y z");
        }
        else if (c == "6") {
            // Save octree
            if (!voxel_struct) {
                std::cout << "Initialize voxelstruct first.\n";
                continue;
            }

            std::string outPath = prompt("Enter octree output filename (e.g. map.bt): ");
            voxel_struct->saveOctree(outPath);
        }
        else if (c == "7") {
            if (!voxel_struct) {
                std::cout << "Initialize voxelstruct first.\n";
                continue;
            }
            voxel_struct->showClassifiedVoxels();
        }
        else if (c == "8") {
            if (!voxel_struct) {
                std::cout << "Initialize voxelstruct first.\n";
                continue;
            }

            voxel_struct->classifyVoxels();

            auto sf  = voxel_struct->getSurfaceFrontiers();
            auto occ = voxel_struct->getOccupiedVoxels();
            auto roi = voxel_struct->getROISurfaceFrontiers();

            std::string base = prompt("Enter base filename (no extension): ");
            if (base.empty()) base = "voxels";

            // Save as 3 separate PLYs (easiest for ellipsoid module)
            saveVec3AsPLY(base + "_occupied.ply", occ);
            saveVec3AsPLY(base + "_frontier.ply", sf);
            saveVec3AsPLY(base + "_roi_frontier.ply", roi);
        }
        else {
            std::cout << "Invalid choice.\n";
        }
    }
}

static void ellipsoidFittingMenu() {
    bool inMenu = true;
    while (inMenu) {
        printEllipsoidMenu();
        std::string c = prompt("Choice: ");

        if (c == "0") inMenu = false;
        else if (c == "1") { /* TODO */ }
        else if (c == "2") { /* TODO */ }
        else std::cout << "Invalid choice.\n";
    }
}

// ---------- Main ----------
int main() {
    bool running = true;

    while (running) {
        printMainMenu();
        std::string c = prompt("Choice: ");

        if (c == "0") running = false;
        else if (c == "1") nextBestViewMenu();
        else if (c == "2") transformsMenu();
        else if (c == "3") voxelstructMenu();
        else if (c == "4") ellipsoidFittingMenu();
        else std::cout << "Invalid choice.\n";
    }

    return 0;
}