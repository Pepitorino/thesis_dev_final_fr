#include <iostream>
#include <string>
#include "nbvtransform.hpp"
#include <cmath>

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

static void printTransformMenu() {
    std::cout << "\n--- TRANSFORMS ---\n"
              << "1) Load PCD\n"
              << "2) Translate to world frame\n"
              << "3) Switch XYZ\n" 
              << "4) View PCD\n"
              << "5) Save PCD\n"
              << "0) Back\n";
}

static void printVoxelMenu() {
    std::cout << "\n--- VOXELSTRUCT ---\n"
              << "1) Option 1\n"
              << "2) Option 2\n"
              << "0) Back\n";
}

static void printEllipsoidMenu() {
    std::cout << "\n--- ELLIPSOID FITTING ---\n"
              << "1) Option 1\n"
              << "2) Option 2\n"
              << "0) Back\n";
}

// ---------- Submenu loops (empty actions for now) ----------
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
            transform.viewPCD("Transforms: PointCloud");
        }
        else if (c == "5") {
            if (!pcd_loaded || transform.pcd == nullptr) {
                std::cout << "Load point cloud first.\n";
                continue;
            }

            std::string path = prompt("Enter output filename (without extension ok): ");
            transform.savePCD(path);
        }
        else {
            std::cout << "Invalid choice.\n";
        }
    }
}

static void voxelstructMenu() {
    bool inMenu = true;
    while (inMenu) {
        printVoxelMenu();
        std::string c = prompt("Choice: ");

        if (c == "0") inMenu = false;
        else if (c == "1") { /* TODO */ }
        else if (c == "2") { /* TODO */ }
        else std::cout << "Invalid choice.\n";
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