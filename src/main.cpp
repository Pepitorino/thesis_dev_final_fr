#include <iostream>
#include <string>
#include "nbvtransform.hpp"
#include <cmath>
#include "voxelstruct.hpp"
#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <algorithm>
#include "ellipsoid.hpp"
#include "utils.hpp"
#include "nbvstrategy.hpp"

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
              << "1) Initialize\n"
              << "2) Generate Viewpoints\n"
              << "3) Insert Transformed Pointcloud to Voxelstruct\n" //need to initialize and generate viewpoints first
              << "4) Get NBV\n" //need to initialize and generate viewpoints first
              << "5) Show last NBV score and position\n" //show position, score, and the visualizer to show where it is
              << "6) Show last NBV image\n" //show opencv "image"
              << "7) Save NBV\n" //Saves the position, score, visualizer, and image
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
              << "9) Count unknown voxels in bbx\n"
              << "0) Back\n";
}

static void printEllipsoidMenu() {
    std::cout << "\n--- ELLIPSOID FITTING ---\n"
              << "1) Load PCDs (Occupied, Frontier, ROI Frontier)\n"
              << "2) GMM Clustering\n"
              << "3) Fit Ellipsoids\n"
              << "4) Clusters Visualization\n"
              << "5) Ellipsoid Visualization\n"
              << "6) Save Cluster Viz\n"
              << "7) Save Ellipsoid Viz\n"
              << "0) Back\n";
}

// ---------- Submenu loops ----------
static void nextBestViewMenu() {
    bool inMenu = true;
    nbvstrategy nbv;

    bool initialized = false;
    bool viewpoints_generated = false;
    bool cloud_inserted = false;

    while (inMenu) {
        printNBVMenu();
        std::string c = prompt("Choice: ");

        if (c == "0") {
            nbv.kill();
            inMenu = false;
        }
        else if (c == "1") {
            std::string settings_path = prompt("Enter settings JSON path: ");
            int ok = nbv.initialize(settings_path);

            if (ok < 0) {
                std::cout << "Failed to initialize NBV.\n";
                initialized = false;
            } else {
                initialized = true;
                std::cout << "NBV initialized.\n";
            }
        }
        else if (c == "2") {
            if (!initialized) {
                std::cout << "Initialize first.\n";
                continue;
            }

            std::string gen_method = prompt("Box (0) or Cylindrical (1): ");
            if (gen_method == "0") {
                nbv.generateViewpoints();
            } else if (gen_method == "1") {
                nbv.generateCylindricalViewpoints();
            }

            viewpoints_generated = true;
            std::cout << "Viewpoints generated.\n";
        }
        else if (c == "3") {
            if (!initialized) {
                std::cout << "Initialize first.\n";
                continue;
            }

            std::string ply_path = prompt("Enter transformed point cloud path: ");
            std::string vp_str = prompt("Enter camera position x,y,z: ");

            Eigen::Vector3d vp;
            if (!parseVec3(vp_str, vp)) {
                std::cout << "Invalid camera position.\n";
                continue;
            }

            nbv.insertTransformedCloud(ply_path, vp);
            cloud_inserted = true;
            std::cout << "Point cloud inserted.\n";
        }
        else if (c == "4") {
            if (!initialized) {
                std::cout << "Initialize first.\n";
                continue;
            }
            if (!viewpoints_generated) {
                std::cout << "Generate viewpoints first.\n";
                continue;
            }
            if (!cloud_inserted) {
                std::cout << "Insert a transformed point cloud first.\n";
                continue;
            }

            nbv.getNBV();
            std::cout << "NBV computation complete.\n";
        }
        else if (c == "5") {
            nbv.showLastNBVInfo();
            nbv.showLastNBVInVoxelTree();
        }
        else if (c == "6") {
            if (!nbv.hasBestNBV() || nbv.best_image.empty()) {
                std::cout << "No NBV image available yet.\n";
                continue;
            }

            const std::string win = "Last NBV Image";
            cv::imshow(win, nbv.best_image);
            while (true) {
                int key = cv::waitKey(30);
                if (key >= 0) break; // any key pressed

                // window closed by user
                if (cv::getWindowProperty(win, cv::WND_PROP_VISIBLE) < 1) {
                    break;
                }
            }
        }
        else if (c == "7") {
            if (!nbv.hasBestNBV()) {
                std::cout << "No NBV available yet.\n";
                continue;
            }

            std::string txt_path = prompt("Enter output text path (e.g. nbv.txt): ");
            std::string img_path = prompt("Enter output image path (e.g. nbv.jpg): ");

            if (txt_path.empty()) txt_path = "nbv.txt";
            if (img_path.empty()) img_path = "nbv.jpg";

            nbv.saveLastNBV(txt_path, img_path);
        }
        else {
            std::cout << "Invalid choice.\n";
        }
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
        else if (c == "9") {
            if (!voxel_struct) {
                std::cout << "Initialize voxelstruct first.\n";
                continue;
            }

            std::string minStr = prompt("Enter bbx_min (x,y,z): ");
            std::string maxStr = prompt("Enter bbx_max (x,y,z): ");

            Eigen::Vector3d bbx_min, bbx_max;
            if (!parseVec3(minStr, bbx_min) || !parseVec3(maxStr, bbx_max)) {
                std::cout << "Invalid input. Expected 3 numbers for each.\n";
                continue;
            }

            auto unknown = voxel_struct->countUnknownInBBX(bbx_min, bbx_max); 

            std::cout << "Number of unknown voxels: " << unknown << std::endl;
        }
        else {
            std::cout << "Invalid choice.\n";
        }
    }
}

static void ellipsoidFittingMenu() {
    bool inMenu = true;

    // --- State that persists while you're in this menu ---
    bool loaded = false;
    bool clustered = false;
    bool fitted = false;

    std::vector<Eigen::Vector3d> occ_pts, frontier_pts, roi_pts;

    std::vector<std::vector<Eigen::Vector3d>> occ_clusters, frontier_clusters, roi_clusters;
    std::vector<EllipsoidParam> ellipsoids;

    // default clustering params (you can prompt these)
    int min_k = 2;
    int max_k = 6;

    // your ellipsoid object
    ellipsoid ell(min_k, max_k);

    while (inMenu) {
        printEllipsoidMenu();
        std::string c = prompt("Choice: ");

        if (c == "0") {
            inMenu = false;
        }
        else if (c == "1") {
            // Load PCDs (Occupied, Frontier, ROI Frontier)
            std::string occ_path = prompt("Enter Occupied PLY/PCD path: ");
            std::string fr_path  = prompt("Enter Frontier PLY/PCD path: ");
            std::string roi_path = prompt("Enter ROI Frontier PLY/PCD path: ");

            bool ok1 = loadCloudAsVec3(occ_path, occ_pts);
            bool ok2 = loadCloudAsVec3(fr_path, frontier_pts);
            bool ok3 = loadCloudAsVec3(roi_path, roi_pts);

            loaded = ok1 && ok2 && ok3;
            clustered = false;
            fitted = false;

            if (!loaded) std::cout << "Load failed (one or more files).\n";
            else std::cout << "All 3 point sets loaded.\n";
        }
        else if (c == "2") {
            // GMM Clustering
            if (!loaded) { std::cout << "Load PCDs first.\n"; continue; }

            std::string smin = prompt("min_clusters (default 2): ");
            std::string smax = prompt("max_clusters (default 6): ");

            if (!smin.empty()) min_k = std::max(1, std::stoi(smin));
            if (!smax.empty()) max_k = std::max(min_k, std::stoi(smax));

            ell = ellipsoid(min_k, max_k);

            std::cout << "Clustering frontier...\n";
            frontier_clusters = ell.gmm_clustering(frontier_pts);

            std::cout << "Clustering occupied...\n";
            occ_clusters = ell.gmm_clustering(occ_pts);

            std::cout << "Clustering ROI frontier...\n";
            roi_clusters = ell.gmm_clustering(roi_pts);

            clustered = true;
            fitted = false;

            std::cout << "Clustering done.\n";
            std::cout << "frontier_clusters=" << frontier_clusters.size()
                      << " occ_clusters=" << occ_clusters.size()
                      << " roi_clusters=" << roi_clusters.size() << "\n";
        }
        else if (c == "3") {
            // Fit Ellipsoids
            if (!clustered) { std::cout << "Run clustering first.\n"; continue; }

            ellipsoids = ell.ellipsoidize_clusters_CGAL(frontier_clusters, occ_clusters, roi_clusters);
            fitted = !ellipsoids.empty();

            std::cout << "Ellipsoid fitting done. Count=" << ellipsoids.size() << "\n";
        }
        else if (c == "4") {
            // Clusters Visualization
            if (!clustered) { std::cout << "Run clustering first.\n"; continue; }

            std::string vs = prompt("VoxelDownSample size (0 to disable): ");
            double voxel_size = 0.0;
            if (!vs.empty()) voxel_size = std::stod(vs);

            // Use your existing function
            ell.showAllClustersColored(frontier_clusters, occ_clusters, roi_clusters, voxel_size);
        }
        else if (c == "5") {
            // Ellipsoid Visualization
            if (!fitted) { std::cout << "Fit ellipsoids first.\n"; continue; }

            std::string wf = prompt("Wireframe? (1=yes, 0=no) [default 1]: ");
            std::string rs = prompt("Sphere resolution [default 20]: ");

            bool wireframe = true;
            int sphere_res = 20;
            if (!wf.empty()) wireframe = (wf != "0");
            if (!rs.empty()) sphere_res = std::max(4, std::stoi(rs));

            ell.showEllipsoidsOpen3D(ellipsoids, wireframe, sphere_res);
        }
        else if (c == "6") {
            // Save Cluster Viz (PNG)
            if (!clustered) { std::cout << "Run clustering first.\n"; continue; }

            std::string png = prompt("Output PNG filename (e.g. clusters.png): ");
            if (png.empty()) { std::cout << "No filename given.\n"; continue; }

            std::string vs = prompt("VoxelDownSample size (0 to disable): ");
            double voxel_size = 0.0;
            if (!vs.empty()) voxel_size = std::stod(vs);

            // frontier red, roi green, occupied blue
            auto f = buildCloudFromClusters(frontier_clusters, Eigen::Vector3d(1,0,0), voxel_size);
            auto r = buildCloudFromClusters(roi_clusters,      Eigen::Vector3d(0,1,0), voxel_size);
            auto o = buildCloudFromClusters(occ_clusters,      Eigen::Vector3d(0,0,1), voxel_size);

            std::vector<std::shared_ptr<const open3d::geometry::Geometry>> geoms;
            if (!o->points_.empty()) geoms.push_back(o);
            if (!f->points_.empty()) geoms.push_back(f);
            if (!r->points_.empty()) geoms.push_back(r);

            saveGeomsScreenshot(geoms, png, "Clusters Viz");
        }
        else if (c == "7") {
            // Save Ellipsoid Viz (PNG)
            if (!fitted) { std::cout << "Fit ellipsoids first.\n"; continue; }

            std::string png = prompt("Output PNG filename (e.g. ellipsoids.png): ");
            if (png.empty()) { std::cout << "No filename given.\n"; continue; }

            std::string wf = prompt("Wireframe? (1=yes, 0=no) [default 1]: ");
            std::string rs = prompt("Sphere resolution [default 20]: ");

            bool wireframe = true;
            int sphere_res = 20;
            if (!wf.empty()) wireframe = (wf != "0");
            if (!rs.empty()) sphere_res = std::max(4, std::stoi(rs));

            auto geoms = buildEllipsoidGeoms(ellipsoids, wireframe, sphere_res);
            saveGeomsScreenshot(geoms, png, "Ellipsoids Viz");
        }
        else {
            std::cout << "Invalid choice.\n";
        }
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