// project_pixels.cu

#include <cuda_runtime.h>
#include <vector>
#include <opencv2/core.hpp>
#include <cstdio>

// ------------------------------------------------------------------
// Type encoding (no std::string on GPU)
// ------------------------------------------------------------------
enum EllipsoidType {
    FRONTIER = 0,
    ROI_SURFACE_FRONTIER = 1,
    OCCUPIED = 2
};

// ------------------------------------------------------------------
// CUDA kernel: one thread per pixel
// ------------------------------------------------------------------
__global__ void project_pixels_kernel(
    int width,
    int height,
    const double* a,
    const double* b,
    const double* c,
    const double* d,
    const double* e,
    const double* f,
    const int* types,
    const double* weights,
    int num_ellipsoids,
    unsigned char* image,   // RGB image, size = width*height*3
    double* scores          // [frontier, roi, occupied]
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    int y = idx / width;
    int x = idx % width;

    double xx = (double)x * x;
    double yy = (double)y * y;
    double xy = (double)x * y;

    unsigned char r = 0, g = 0, b_col = 0;

    for (int i = 0; i < num_ellipsoids; i++) {
        double value =
            a[i] * xx +
            b[i] * yy +
            c[i] * xy +
            d[i] * x +
            e[i] * y +
            f[i];

        if (value < 0.0) {
            int t = types[i];

            if (t == FRONTIER) {
                r = 0; g = 0; b_col = 255;
                atomicAdd(&scores[0], weights[i]);
            }
            else if (t == ROI_SURFACE_FRONTIER) {
                r = 0; g = 255; b_col = 0;
                atomicAdd(&scores[1], weights[i]);
            }
            else if (t == OCCUPIED) {
                r = 255; g = 0; b_col = 0;
                atomicAdd(&scores[2], weights[i]);
            }
        }
    }

    int base = 3 * idx;
    image[base + 0] = b_col;
    image[base + 1] = g;
    image[base + 2] = r;
}

// ------------------------------------------------------------------
// C++ callable wrapper
// ------------------------------------------------------------------
extern "C"
void project_pixels_cuda(
    int width,
    int height,
    const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c,
    const std::vector<double>& d,
    const std::vector<double>& e,
    const std::vector<double>& f,
    const std::vector<int>& types,
    const std::vector<double>& weights,
    cv::Mat& output_image,
    double& frontier_score,
    double& roi_score,
    double& occupied_score
)
{
    int num_pixels = width * height;
    int num_ellipsoids = (int)a.size();

    // --- Device memory ---
    double *d_a, *d_b, *d_c, *d_d, *d_e, *d_f;
    int* d_types;
    double* d_weights;
    unsigned char* d_image;
    double* d_scores;

    cudaMalloc(&d_a, sizeof(double) * num_ellipsoids);
    cudaMalloc(&d_b, sizeof(double) * num_ellipsoids);
    cudaMalloc(&d_c, sizeof(double) * num_ellipsoids);
    cudaMalloc(&d_d, sizeof(double) * num_ellipsoids);
    cudaMalloc(&d_e, sizeof(double) * num_ellipsoids);
    cudaMalloc(&d_f, sizeof(double) * num_ellipsoids);
    cudaMalloc(&d_types, sizeof(int) * num_ellipsoids);
    cudaMalloc(&d_weights, sizeof(double) * num_ellipsoids);

    cudaMalloc(&d_image, sizeof(unsigned char) * num_pixels * 3);
    cudaMalloc(&d_scores, sizeof(double) * 3);

    // --- Copy to device ---
    cudaMemcpy(d_a, a.data(), sizeof(double) * num_ellipsoids, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * num_ellipsoids, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c.data(), sizeof(double) * num_ellipsoids, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d.data(), sizeof(double) * num_ellipsoids, cudaMemcpyHostToDevice);
    cudaMemcpy(d_e, e.data(), sizeof(double) * num_ellipsoids, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, f.data(), sizeof(double) * num_ellipsoids, cudaMemcpyHostToDevice);
    cudaMemcpy(d_types, types.data(), sizeof(int) * num_ellipsoids, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), sizeof(double) * num_ellipsoids, cudaMemcpyHostToDevice);

    double zero_scores[3] = {0.0, 0.0, 0.0};
    cudaMemcpy(d_scores, zero_scores, sizeof(double) * 3, cudaMemcpyHostToDevice);

    // --- Launch kernel ---
    int threads = 256;
    int blocks = (num_pixels + threads - 1) / threads;

    project_pixels_kernel<<<blocks, threads>>>(
        width, height,
        d_a, d_b, d_c, d_d, d_e, d_f,
        d_types, d_weights,
        num_ellipsoids,
        d_image,
        d_scores
    );

    cudaDeviceSynchronize();

    // --- Copy results back ---
    output_image.create(height, width, CV_8UC3);
    cudaMemcpy(output_image.data, d_image,
               sizeof(unsigned char) * num_pixels * 3,
               cudaMemcpyDeviceToHost);

    double h_scores[3];
    cudaMemcpy(h_scores, d_scores, sizeof(double) * 3, cudaMemcpyDeviceToHost);

    frontier_score = h_scores[0];
    roi_score = h_scores[1];
    occupied_score = h_scores[2];

    // --- Cleanup ---
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_d); cudaFree(d_e); cudaFree(d_f);
    cudaFree(d_types); cudaFree(d_weights);
    cudaFree(d_image); cudaFree(d_scores);
}
