#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <chrono>
#include <limits>

#define TPB 512

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

    float min_dist = INFINITY;
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

    dim3 block(TPB);
    dim3 grid((n + TPB - 1) / TPB);

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

float experiment(int n, int k) {

    int dims = 2;  // dimensione
    int max_iters = 100;

    std::vector<float> points(n * dims);
    std::vector<float> centroids(k * dims);
    std::vector<int> labels(n);

    for (int i = 0; i < n * dims; i++) points[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < k * dims; i++) centroids[i] = static_cast<float>(rand()) / RAND_MAX;

    auto start = std::chrono::high_resolution_clock::now();
    kmeans(points.data(), centroids.data(), labels.data(), n, k, dims, max_iters);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Tempo di esecuzione: " << ((float)duration.count())/1000 << " millisecondi" << std::endl;

    std::cout << "K-means terminato.\n";
    return ((float)duration.count())/1000;
}

typedef struct {
	float time;
	int numPoints;
	int numClusters;
	int tpb;
}ExperimentResult;

// Funzione per scrivere il vettore di struct in un file CSV
void writeToCSV(const std::vector<ExperimentResult>& results, const std::string& filename) {
	// Apri il file in modalità scrittura
	std::ofstream file;
	file.open (filename);
	// Controlla se il file è aperto correttamente
	if (!file.is_open()) {
		std::cerr << "Errore nell'aprire il file!" << std::endl;
		return;
	}

	// Scrivi l'intestazione del CSV (opzionale)
	file << "numPoints,numClusters,tpb, time\n";

	// Itera attraverso la lista di risultati e scrivi ogni struct nel CSV
	for (const auto& result : results) {
		file << result.numPoints << "," << result.numClusters << "," << result.tpb << "," << result.time << "\n";
	}

	// Chiudi il file
	file.close();

	std::cout << "File CSV scritto correttamente." << std::endl;
}

int main(){

	int it = 10;
	int j=0;

	std::vector<ExperimentResult> results;
	ExperimentResult result = {0};

	for(int i=0;i<it; i++){
		j=pow(2,i);
  		result.time = experiment(50000*j, 10);
  		result.numPoints = 50000*j;
  		result.numClusters = 10;
  		result.tpb = TPB;
        results.push_back(result);
  	}
    writeToCSV(results, "exp1_par_naive.csv");     //FIXME non crea il file
}
