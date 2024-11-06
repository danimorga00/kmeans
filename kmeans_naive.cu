#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <limits>

#define BLOCK_SIZE 256

__device__ float calculate_distance(float *a, float *b, int dims) {
    float dist = 0.0;
    for (int i = 0; i < dims; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return sqrt(dist);
}

__global__ void assign_clusters(float *points, float *centroids, int *labels, int n, int k, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float min_dist = FLT_MAX;
    int best_cluster = 0;
    for (int j = 0; j < k; j++) {
        float dist = calculate_distance(&points[idx * dims], &centroids[j * dims], dims);
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = j;
        }
    }
    labels[idx] = best_cluster;
}

__global__ void update_centroids(float *points, float *centroids, int *labels, int n, int k, int dims) {
    int cluster_id = threadIdx.x;
    if (cluster_id >= k) return;

    float *new_centroid = &centroids[cluster_id * dims];
    int count = 0;

    // Inizializza il centroide a 0 per sommare le coordinate
    for (int d = 0; d < dims; d++) {
        new_centroid[d] = 0.0f;
    }

    // Somma le coordinate dei punti nel cluster
    for (int i = 0; i < n; i++) {
        if (labels[i] == cluster_id) {
            for (int d = 0; d < dims; d++) {
                new_centroid[d] += points[i * dims + d];
            }
            count++;
        }
    }

    // Media delle coordinate del centroide
    if (count > 0) {
        for (int d = 0; d < dims; d++) {
            new_centroid[d] /= count;
        }
    }
}

void kmeans(float *points, float *centroids, int *labels, int n, int k, int dims, int max_iters) {
    int *d_labels;
    float *d_points, *d_centroids;

    cudaMalloc(&d_points, n * dims * sizeof(float));
    cudaMalloc(&d_centroids, k * dims * sizeof(float));
    cudaMalloc(&d_labels, n * sizeof(int));

    cudaMemcpy(d_points, points, n * dims * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, k * dims * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int iter = 0; iter < max_iters; iter++) {
        // Assegna ogni punto al cluster più vicino
        assign_clusters<<<grid, block>>>(d_points, d_centroids, d_labels, n, k, dims);
        cudaDeviceSynchronize();

        // Aggiorna i centroidi: lancia K thread, uno per ogni centroide
        update_centroids<<<1, k>>>(d_points, d_centroids, d_labels, n, k, dims);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(labels, d_labels, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(centroids, d_centroids, k * dims * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_labels);
}

int main() {
    int n = 1000000; // numero di punti
    int k = 10;     // numero di cluster
    int dims = 2;  // dimensione dello spazio
    int max_iters = 100;

    std::vector<float> points(n * dims);
    std::vector<float> centroids(k * dims);
    std::vector<int> labels(n);

    // Inizializzazione casuale dei punti e dei centri
    for (int i = 0; i < n * dims; i++) points[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < k * dims; i++) centroids[i] = static_cast<float>(rand()) / RAND_MAX;

    auto start = std::chrono::high_resolution_clock::now();
    kmeans(points.data(), centroids.data(), labels.data(), n, k, dims, max_iters);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Tempo di esecuzione: " << ((float)duration.count())/1000 << " millisecondi" << std::endl;

    std::cout << "K-means terminato.\n";
    return 0;
}
