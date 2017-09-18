#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "Header/common.h"

int check_neighbour_open(int *maze, int length, int row_length, int pos, int neighbour){
	if(neighbour >= 0 && neighbour < length){
		if(pos % row_length == row_length -1 && neighbour == pos + 1)
			return 0;
		if(pos % row_length == 0 && neighbour == pos - 1)
			return 0;
		if(maze[neighbour] == OPEN || maze[neighbour] == OBJECTIVE){
			return 1;
		}else{
			return 0;
		}
	}
	return 0;
}

void CPU_cellular_automata_solver(int *maze, int width, int height){
	int row_length= width;
	int length = width*height;
	bool again = true;
	int i;
	while(again){
		//setup the boolen value
		again = false;
		//per ogni cella
		for(i=0; i < length; i++){
			if(maze[i] != WALL){
				//check neighbour cells
				int count = 0;
				//up
				count += check_neighbour_open(maze, length, row_length, i, i - row_length);
				//down
				count += check_neighbour_open(maze, length, row_length, i, i + row_length);
				//right
				count += check_neighbour_open(maze, length, row_length, i, i + 1);
				//left
				count += check_neighbour_open(maze, length, row_length, i, i - 1);
				//only if there is one 1 near me
				if(count == 1 && maze[i] == OPEN){
					maze[i] = WALL;
					again = true;
				}
			}
		}
	}
}

__device__ int DEVICE_check_neighbour_open(int *maze, int length, int row_length, int pos, int neighbour){
	if(neighbour >= 0 && neighbour < length){
			if(pos % row_length == row_length -1 && neighbour == pos + 1)
				return 0;
			if(pos % row_length == 0 && neighbour == pos - 1)
				return 0;
			if(maze[neighbour] == OPEN || maze[neighbour] == OBJECTIVE){
				return 1;
			}else{
				return 0;
			}
		}else{
			return 0;
		}
}

__global__ void GPU_check_neighbour_open(int *maze, int length, int row_length, bool *again, int offset){
	int i = blockIdx.x * row_length + offset + threadIdx.x;
	int count = 0;
	//up
	count += DEVICE_check_neighbour_open(maze, length, row_length, i,  i - row_length);
	//down
	count += DEVICE_check_neighbour_open(maze, length, row_length, i, i + row_length);
	//right
	count += DEVICE_check_neighbour_open(maze, length, row_length, i, i + 1);
	//left
	count += DEVICE_check_neighbour_open(maze, length, row_length, i, i - 1);
	__syncthreads();
	if(count == 1 && maze[i] == OPEN){
		maze[i] = WALL;
		*again = true;
	}

}

void GPU_cellular_automata_solver(int *maze, int width, int height){
	int row_length = width;
	int length = width * height;
	bool again = true;
	int *dev_maze;
	bool *dev_again;

	cudaMalloc(&dev_maze, sizeof(int) * length);
	cudaMalloc(&dev_again, sizeof(bool));

	cudaMemcpy(dev_maze,maze,sizeof(int)*length,cudaMemcpyHostToDevice);
	int max_rec = width / MAX_THREAD;
	int offset;
	while(again){
		//setup bool value
		again = false;
		cudaMemcpy(dev_again, &again, sizeof(bool), cudaMemcpyHostToDevice);
		offset = 0;
		for(int i = 0; i < max_rec; i++){
			GPU_check_neighbour_open<<<height, MAX_THREAD>>>(dev_maze, length, row_length, dev_again, offset);
			offset = (i + 1) * MAX_THREAD;
		}
		GPU_check_neighbour_open<<<height, width % MAX_THREAD>>>(dev_maze, length, row_length, dev_again, offset);
		cudaDeviceSynchronize();
		//copy data on host
		cudaMemcpy(&again, dev_again, sizeof(bool), cudaMemcpyDeviceToHost);
	}
	//finish
	cudaMemcpy(maze, dev_maze, sizeof(int) * length, cudaMemcpyDeviceToHost);
}
