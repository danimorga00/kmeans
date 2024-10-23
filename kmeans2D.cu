#include <time.h>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <cuda_runtime.h>


#define TPB 256
#define MAX_ITER 100

// Struttura per un punto in 2D
typedef struct {
    int x, y;
}Point;

// Funzione per calcolare la distanza euclidea tra due punti
__device__ float distance(Point a, Point b) {
    return sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

__global__ void kMeansClusterAssignment(Point *d_datapoints, int *d_clust_assn, Point *d_centroids, int N, int K)
{
	//get idx for this datapoint
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	//bounds check
	if (idx >= N) return;

	//find the closest centroid to this datapoint
	float min_dist = INFINITY;
	int closest_centroid = 0;

	for(int c = 0; c<K;c++)
	{
		float dist = distance(d_datapoints[idx],d_centroids[c]);

		if(dist < min_dist)
		{
			min_dist = dist;
			closest_centroid=c;
		}
	}

	//assign closest cluster id for this datapoint/thread
	d_clust_assn[idx]=closest_centroid;
	__syncthreads();
}
__global__ void accumulateCentroid(Point *d_datapoints, int *d_clust_assn,
                                     Point *d_centroids, int *d_clust_sizes, int N, int K) {
    // Indice globale del thread
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Controlliamo che il thread stia lavorando su un dato valido
    if (tid < N) {
        // Ottieni il punto corrente
        Point point = d_datapoints[tid];

        // Ottieni l'ID del centroide a cui è stato assegnato questo punto
        int clusterId = d_clust_assn[tid];

        // Aggiorna il centroide corrispondente e il numero di punti associati (in modo atomico)
        atomicAdd(&(d_centroids[clusterId].x), point.x);
        atomicAdd(&(d_centroids[clusterId].y), point.y);
        atomicAdd(&(d_clust_sizes[clusterId]), 1);
    }
}

__global__ void resetCentroids(Point *d_centroids, int *d_clust_sizes, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K) {
        d_centroids[idx].x = 0;
        d_centroids[idx].y = 0;
        d_clust_sizes[idx] = 0;
    }
}
__global__ void finalizeCentroids(Point *d_centroids, int *d_clust_sizes, int K) {
    // Indice del thread
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < K) {
        // Evita la divisione per zero
        if (d_clust_sizes[tid] > 0) {
            d_centroids[tid].x /= d_clust_sizes[tid];
            d_centroids[tid].y /= d_clust_sizes[tid];
        } else {
            // In caso di cluster vuoto, si può lasciare il centroide invariato
            // oppure gestire il caso in altro modo (es: randomizzare il centroide)
        }
    }
}
float kmeans(int N, int K, int tpb)
{

	//allocate memory on the device for the data points
	Point *d_datapoints;
	//allocate memory on the device for the cluster assignments
	int *d_clust_assn = 0;
	//allocate memory on the device for the cluster centroids
	Point *d_centroids;
	//allocate memory on the device for the cluster sizes
	int *d_clust_sizes=0;

	cudaMalloc(&d_datapoints, N*sizeof(Point));
	cudaMalloc(&d_clust_assn,N*sizeof(int));
	cudaMalloc(&d_centroids,K*sizeof(Point));
	cudaMalloc(&d_clust_sizes,K*sizeof(int));

	Point *h_centroids = (Point*)malloc(K*sizeof(Point));
	Point *h_datapoints = (Point*)malloc(N*sizeof(Point));
	int *h_clust_assn = (int*)malloc(N*sizeof(int));
	int *h_clust_sizes = (int*)malloc(K*sizeof(int));

	srand(time(0));

	//initialize centroids
	for(int c=0;c<K;++c)
	{
		h_centroids[c].x = rand() % 1000;
		h_centroids[c].y = rand() % 1000;
    	//std::cout << "{" << h_centroids[c]  << "}" << std::endl;
		h_clust_sizes[c]=0;
	}

	//initalize datapoints
	for(int d = 0; d < N; ++d)
	{
		h_datapoints[d].x = rand() % 1000;
		h_datapoints[d].y = rand() % 1000;
    //std::cout << "{" << h_datapoints[d]  << "}" << std::endl;
	}

	cudaMemcpy(d_centroids,h_centroids,K*sizeof(Point),cudaMemcpyHostToDevice);
	cudaMemcpy(d_datapoints,h_datapoints,N*sizeof(Point),cudaMemcpyHostToDevice);
	cudaMemcpy(d_clust_sizes,h_clust_sizes,K*sizeof(int),cudaMemcpyHostToDevice);

	int cur_iter = 0;

	auto start = std::chrono::high_resolution_clock::now();

	while(cur_iter < MAX_ITER)
	{
		kMeansClusterAssignment<<<(N+tpb-1)/tpb,tpb>>>(d_datapoints,d_clust_assn,d_centroids, N, K);

		resetCentroids<<<(N+tpb-1)/tpb,tpb>>>(d_centroids, d_clust_sizes,  K);
		accumulateCentroid<<<(N+tpb-1)/tpb,tpb>>>(d_datapoints,d_clust_assn,d_centroids,d_clust_sizes, N, K);
		cudaDeviceSynchronize();
		finalizeCentroids<<<(K + tpb - 1) / tpb, tpb>>>(d_centroids, d_clust_sizes, K);

		cur_iter+=1;
	}

	for(int c=0;c<N;++c)
	{
    //std::cout << "{" << h_centroids[c]  << "}" << std::endl;
	}
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  	std::cout << "Tempo di esecuzione: " << ((float)duration.count())/1000 << " millisecondi" << std::endl;

	cudaFree(d_datapoints);
	cudaFree(d_clust_assn);
	cudaFree(d_centroids);
	cudaFree(d_clust_sizes);

	free(h_centroids);
	free(h_datapoints);
	free(h_clust_sizes);

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

std::vector<ExperimentResult> firstExperiment(){
	int it = 10;
	int j=0;
	std::vector<ExperimentResult> results;
	ExperimentResult result = {0};

	for(int i=0;i<it; i++){
		j=pow(2,i);
  		result.time = kmeans(50000, 10*j, TPB);
  		result.numPoints = 50000;
  		result.numClusters = 10*j;
  		result.tpb = TPB;
        results.push_back(result);
  	}
	return results;
}

std::vector<ExperimentResult> secondExperiment(){
	int it = 10;
	int j=0;
	std::vector<ExperimentResult> results;
	ExperimentResult result = {0};

	for(int i=0;i<it; i++){
		j=pow(2,i);
  		result.time = kmeans(50000*j, 10, TPB);
  		result.numPoints = 50000*j;
  		result.numClusters = 10;
  		result.tpb = TPB;
        results.push_back(result);
  	}
	return results;
}

std::vector<ExperimentResult> thirdExperiment(){
	int it = 6;
	int j=0;
	std::vector<ExperimentResult> results;
	ExperimentResult result = {0};

	for(int i=0;i<it; i++){
		j=pow(2,i);
  		result.time = kmeans(5000000, 100, 32*j);
  		result.numPoints = 5000000;
  		result.numClusters = 100;
  		result.tpb = 32*j;
        results.push_back(result);
  	}
	return results;
}

int main(){

/*
	std::vector<ExperimentResult> results = firstExperiment();
    writeToCSV(results, "exp1_2D_par.csv");

	std::vector<ExperimentResult> results = secondExperiment();
    writeToCSV(results, "exp2_2D_par.csv");
*/
	std::vector<ExperimentResult> results = thirdExperiment();
    writeToCSV(results, "exp3.csv");
}