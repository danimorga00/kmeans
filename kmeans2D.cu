#include <time.h>
#include <cstdio>
#include <chrono>
#include <iostream>

#define TPB 256
#define MAX_ITER 100

// Struttura per un punto in 2D
struct Point {
    float x, y;
};

// Funzione per calcolare la distanza euclidea tra due punti
__device__ float distance(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
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

	for(int c = 0; c<K;++c)
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
}


__global__ void kMeansCentroidUpdate(Point *d_datapoints, int *d_clust_assn, Point *d_centroids, int *d_clust_sizes, int N, int K)
{

	//get idx of thread at grid level
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	//bounds check
	if (idx >= N) return;

	//get idx of thread at the block level
	const int s_idx = threadIdx.x;

	//put the datapoints and corresponding cluster assignments in shared memory so that they can be summed by thread 0 later
	__shared__ Point s_datapoints[TPB];
	s_datapoints[s_idx]= d_datapoints[idx];

	__shared__ int s_clust_assn[TPB];
	s_clust_assn[s_idx] = d_clust_assn[idx];

	__syncthreads();

	//it is the thread with idx 0 (in each block) that sums up all the values within the shared array for the block it is in
	if(s_idx==0)
	{
		Point* b_clust_datapoint_sums=(Point*)malloc(K*sizeof(Point));
		int* b_clust_sizes=(int*)malloc(K*sizeof(float));

		for(int j=0; j< blockDim.x; ++j)
		{
			int clust_id = s_clust_assn[j];
			b_clust_datapoint_sums[clust_id].x+=s_datapoints[j].x;
			b_clust_datapoint_sums[clust_id].y+=s_datapoints[j].y;
			b_clust_sizes[clust_id]+=1;
		}

		//Now we add the sums to the global centroids and add the counts to the global counts.
		for(int z=0; z < K; ++z)
		{
			atomicAdd(&d_centroids[z].x,b_clust_datapoint_sums[z].x);
			atomicAdd(&d_centroids[z].y,b_clust_datapoint_sums[z].y);
			atomicAdd(&d_clust_sizes[z],b_clust_sizes[z]);
		}
	}

	__syncthreads();

	//currently centroids are just sums, so divide by size to get actual centroids
	if(idx < K){
		d_centroids[idx].x = d_centroids[idx].x/d_clust_sizes[idx];
		d_centroids[idx].y = d_centroids[idx].y/d_clust_sizes[idx];
	}

}


int firstExperiment(int N, int K)
{

	//allocate memory on the device for the data points
	Point *d_datapoints=0;
	//allocate memory on the device for the cluster assignments
	int *d_clust_assn = 0;
	//allocate memory on the device for the cluster centroids
	Point *d_centroids = 0;
	//allocate memory on the device for the cluster sizes
	int *d_clust_sizes=0;

	cudaMalloc(&d_datapoints, N*sizeof(Point));
	cudaMalloc(&d_clust_assn,N*sizeof(int));
	cudaMalloc(&d_centroids,K*sizeof(Point));
	cudaMalloc(&d_clust_sizes,K*sizeof(float));

	Point *h_centroids = (Point*)malloc(K*sizeof(Point));
	Point *h_datapoints = (Point*)malloc(N*sizeof(Point));
	int *h_clust_sizes = (int*)malloc(K*sizeof(int));

	srand(time(0));

	//initialize centroids
	for(int c=0;c<K;++c)
	{
		h_centroids[c].x=(float) rand() / (double)RAND_MAX;
		h_centroids[c].y=(float) rand() / (double)RAND_MAX;
		h_clust_sizes[c]=0;
	}

	//initalize datapoints
	for(int d = 0; d < N; ++d)
	{
		h_datapoints[d].x = (float) rand() / (double)RAND_MAX;
		h_datapoints[d].y = (float) rand() / (double)RAND_MAX;
	}

	cudaMemcpy(d_centroids,h_centroids,K*sizeof(Point),cudaMemcpyHostToDevice);
	cudaMemcpy(d_datapoints,h_datapoints,N*sizeof(Point),cudaMemcpyHostToDevice);
	cudaMemcpy(d_clust_sizes,h_clust_sizes,K*sizeof(int),cudaMemcpyHostToDevice);

	int cur_iter = 1;

	auto start = std::chrono::high_resolution_clock::now();

	while(cur_iter < MAX_ITER)
	{
		//call cluster assignment kernel
		kMeansClusterAssignment<<<(N+TPB-1)/TPB,TPB>>>(d_datapoints,d_clust_assn,d_centroids, N, K);

		//copy new centroids back to host
		cudaMemcpy(h_centroids,d_centroids,K*sizeof(float),cudaMemcpyDeviceToHost);

		for(int i =0; i < K; ++i){
			//Uprintf("Iteration %d: centroid %d: %f\n",cur_iter,i,h_centroids[i]);
		}

		//reset centroids and cluster sizes (will be updated in the next kernel)
		cudaMemset(d_centroids,0.0,K*sizeof(Point));
		cudaMemset(d_clust_sizes,0,K*sizeof(int));

		//call centroid update kernel
		kMeansCentroidUpdate<<<(N+TPB-1)/TPB,TPB>>>(d_datapoints,d_clust_assn,d_centroids,d_clust_sizes, N, K);

		cur_iter+=1;
	}

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
  firstExperiment(10000000, 100);
}