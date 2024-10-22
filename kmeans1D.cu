#include <time.h>
#include <cstdio>
#include <chrono>
#include <iostream>

#define TPB 64
#define MAX_ITER 5

__device__ float distance(float x1, float x2)
{
	return sqrt((x2-x1)*(x2-x1));
}

__global__ void kMeansClusterAssignment(float *d_datapoints, int *d_clust_assn, float *d_centroids, int N, int K)
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


__global__ void kMeansCentroidUpdate(float *d_datapoints, int *d_clust_assn, float *d_centroids, int *d_clust_sizes, int N, int K)
{

	//get idx of thread at grid level
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	//bounds check
	if (idx >= N) return;

	//get idx of thread at the block level
	const int s_idx = threadIdx.x;

	//put the datapoints and corresponding cluster assignments in shared memory so that they can be summed by thread 0 later
	__shared__ float s_datapoints[TPB];
	s_datapoints[s_idx]= d_datapoints[idx];

	__shared__ int s_clust_assn[TPB];
	s_clust_assn[s_idx] = d_clust_assn[idx];

	__syncthreads();

	//it is the thread with idx 0 (in each block) that sums up all the values within the shared array for the block it is in
	if(s_idx==0)
	{
		float* b_clust_datapoint_sums=(float*)malloc(K*sizeof(float));
		int* b_clust_sizes=(int*)malloc(K*sizeof(float));

		for(int j=0; j< blockDim.x; j++)
		{
			int clust_id = s_clust_assn[j];
			b_clust_datapoint_sums[clust_id]+=s_datapoints[j];
			b_clust_sizes[clust_id]+=1;
		}

		//Now we add the sums to the global centroids and add the counts to the global counts.
		for(int z=0; z < K; z++)
		{
			atomicAdd(&d_centroids[z],b_clust_datapoint_sums[z]);
			atomicAdd(&d_clust_sizes[z],b_clust_sizes[z]);
		}
	}

	__syncthreads();

	//currently centroids are just sums, so divide by size to get actual centroids
	if(idx < K){
		d_centroids[idx] = d_centroids[idx]/d_clust_sizes[idx];
	}
	__syncthreads();
}


int firstExperiment(int N, int K)
{

	//allocate memory on the device for the data points
	float *d_datapoints=0;
	//allocate memory on the device for the cluster assignments
	int *d_clust_assn = 0;
	//allocate memory on the device for the cluster centroids
	float *d_centroids = 0;
	//allocate memory on the device for the cluster sizes
	int *d_clust_sizes=0;

	cudaMalloc(&d_datapoints, N*sizeof(float));
	cudaMalloc(&d_clust_assn,N*sizeof(int));
	cudaMalloc(&d_centroids,K*sizeof(float));
	cudaMalloc(&d_clust_sizes,K*sizeof(float));

	float *h_centroids = (float*)malloc(K*sizeof(float));
	float *h_datapoints = (float*)malloc(N*sizeof(float));
	int *h_clust_assn = (int*)malloc(N*sizeof(int));
	int *h_clust_sizes = (int*)malloc(K*sizeof(int));

	srand(time(0));

	//initialize centroids
	for(int c=0;c<K;++c)
	{
		h_centroids[c]=(float) rand() / (double)RAND_MAX;
    std::cout << "{" << h_centroids[c]  << "}" << std::endl;
		h_clust_sizes[c]=0;
	}

	//initalize datapoints
	for(int d = 0; d < N; ++d)
	{
		h_datapoints[d] = (float) rand() / (double)RAND_MAX;
    //std::cout << "{" << h_datapoints[d]  << "}" << std::endl;
	}

	cudaMemcpy(d_centroids,h_centroids,K*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_datapoints,h_datapoints,N*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_clust_sizes,h_clust_sizes,K*sizeof(int),cudaMemcpyHostToDevice);

	int cur_iter = 1;

	auto start = std::chrono::high_resolution_clock::now();

	while(cur_iter < MAX_ITER)
	{
		//call cluster assignment kernel
		kMeansClusterAssignment<<<(N+TPB-1)/TPB,TPB>>>(d_datapoints,d_clust_assn,d_centroids, N, K);

		//copy new centroids back to host
		cudaMemcpy(h_centroids,d_centroids,K*sizeof(float),cudaMemcpyDeviceToHost);
		cudaMemcpy(h_clust_sizes,d_clust_sizes,K*sizeof(int),cudaMemcpyDeviceToHost);

		cudaMemcpy(h_clust_assn,d_clust_assn,N*sizeof(int),cudaMemcpyDeviceToHost);
		printf("Iteration %d: point 0: %f --> %d\n",cur_iter,h_datapoints[0],h_clust_assn[0]);

		//reset centroids and cluster sizes (will be updated in the next kernel)
		cudaMemset(d_centroids,0.0,K*sizeof(float));
		cudaMemset(d_clust_sizes,0,K*sizeof(int));

		//call centroid update kernel
		kMeansCentroidUpdate<<<(N+TPB-1)/TPB,TPB>>>(d_datapoints,d_clust_assn,d_centroids,d_clust_sizes, N, K);

		for(int i =0; i < K; ++i){
			printf("Iteration %d: centroid %d: %f, cluster size: %d\n",cur_iter,i,h_centroids[i], h_clust_sizes[i]);
		}

		cur_iter+=1;
	}

	for(int c=0;c<K;++c)
	{
    //std::cout << "{" << h_centroids[c]  << "}" << std::endl;
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
  firstExperiment(100000, 3);
}