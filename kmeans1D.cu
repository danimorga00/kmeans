#include <time.h>
#include <cstdio>
#include <chrono>
#include <iostream>

#define TPB 512
#define MAX_ITER 5

__device__ float distance(float x1, float x2)
{
	return sqrt((x2-x1)*(x2-x1));
}

__global__ void kMeansClusterAssignment(int *d_datapoints, int *d_clust_assn, int *d_centroids, int N, int K)
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
__global__ void accumulateCentroid(int *d_datapoints, int *d_clust_assn,
                                     int *d_centroids, int *d_clust_sizes, int N, int K) {
    // Indice globale del thread
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Controlliamo che il thread stia lavorando su un dato valido
    if (tid < N) {
        // Ottieni il punto corrente
        int point = d_datapoints[tid];

        // Ottieni l'ID del centroide a cui è stato assegnato questo punto
        int clusterId = d_clust_assn[tid];

        // Aggiorna il centroide corrispondente e il numero di punti associati (in modo atomico)
        atomicAdd(&(d_centroids[clusterId]), point);
        atomicAdd(&(d_clust_sizes[clusterId]), 1);
    }
}

__global__ void resetCentroids(int *d_centroids, int *d_clust_sizes, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K) {
        d_centroids[idx] = 0.0f;
        d_clust_sizes[idx] = 0;
    }
}
__global__ void finalizeCentroids(int *d_centroids, int *d_clust_sizes, int K) {
    // Indice del thread
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < K) {
        // Evita la divisione per zero
        if (d_clust_sizes[tid] > 0) {
            d_centroids[tid] /= d_clust_sizes[tid];
        } else {
            // In caso di cluster vuoto, si può lasciare il centroide invariato
            // oppure gestire il caso in altro modo (es: randomizzare il centroide)
        }
    }
}
int firstExperiment(int N, int K)
{

	//allocate memory on the device for the data points
	int *d_datapoints=0;
	//allocate memory on the device for the cluster assignments
	int *d_clust_assn = 0;
	//allocate memory on the device for the cluster centroids
	int *d_centroids = 0;
	//allocate memory on the device for the cluster sizes
	int *d_clust_sizes=0;

	cudaMalloc(&d_datapoints, N*sizeof(int));
	cudaMalloc(&d_clust_assn,N*sizeof(int));
	cudaMalloc(&d_centroids,K*sizeof(int));
	cudaMalloc(&d_clust_sizes,K*sizeof(int));

	int *h_centroids = (int*)malloc(K*sizeof(int));
	int *h_datapoints = (int*)malloc(N*sizeof(int));
	int *h_clust_assn = (int*)malloc(N*sizeof(int));
	int *h_clust_sizes = (int*)malloc(K*sizeof(int));

	srand(time(0));

	//initialize centroids
	for(int c=0;c<K;++c)
	{
		h_centroids[c]= rand() % 1000;
    std::cout << "{" << h_centroids[c]  << "}" << std::endl;
		h_clust_sizes[c]=0;
	}

	//initalize datapoints
	for(int d = 0; d < N; ++d)
	{
		h_datapoints[d] = rand() % 1000;
    //std::cout << "{" << h_datapoints[d]  << "}" << std::endl;
	}

	cudaMemcpy(d_centroids,h_centroids,K*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_datapoints,h_datapoints,N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_clust_sizes,h_clust_sizes,K*sizeof(int),cudaMemcpyHostToDevice);

	int cur_iter = 0;

	auto start = std::chrono::high_resolution_clock::now();

	while(cur_iter < MAX_ITER)
	{
		//call cluster assignment kernel
		kMeansClusterAssignment<<<(N+TPB-1)/TPB,TPB>>>(d_datapoints,d_clust_assn,d_centroids, N, K);

		//copy new centroids back to host
		cudaMemcpy(h_centroids,d_centroids,K*sizeof(int),cudaMemcpyDeviceToHost);
		cudaMemcpy(h_clust_sizes,d_clust_sizes,K*sizeof(int),cudaMemcpyDeviceToHost);
		cudaMemcpy(h_clust_assn,d_clust_assn,N*sizeof(int),cudaMemcpyDeviceToHost);

		//printf("Iteration %d: point 1000: %f --> %d\n",cur_iter,h_datapoints[0],h_clust_assn[0]);
		//reset centroids and cluster sizes (will be updated in the next kernel)
		resetCentroids<<<(N+TPB-1)/TPB,TPB>>>(d_centroids, d_clust_sizes,  K);

		//call centroid update kernel
		accumulateCentroid<<<(N+TPB-1)/TPB,TPB>>>(d_datapoints,d_clust_assn,d_centroids,d_clust_sizes, N, K);

		cudaDeviceSynchronize();

		// Kernel per dividere i centroidi
		int blocksPerGrid = (K + TPB - 1) / TPB;
		finalizeCentroids<<<blocksPerGrid, TPB>>>(d_centroids, d_clust_sizes, K);
		for(int i =0; i < K; ++i){
			printf("Iteration %d: centroid %d: %d, cluster size: %d\n",cur_iter,i,h_centroids[i], h_clust_sizes[i]);
		}

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

	return 0;
}

int main(){
  firstExperiment(10000000, 3);
}