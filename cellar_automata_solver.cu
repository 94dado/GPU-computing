#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

int check_neighbour_open(int *maze, int length, int row_length, int pos, int neighbour){
	if(neighbour >= 0 && neighbour < length){
		if(pos % row_length == row_length -1 && neighbour == pos + 1)
			return 0;
		if(pos % row_length == 0 && neighbour == pos - 1)
			return 0;

		return maze[pos] == OPEN? 1 : 0;
	}
	return 0;
}

__global__ void GPU_check_neighbour_open(int *maze, int length, int row_length, bool *again){
	int i = blockIdx.x * row_length + threadIdx.x;
	//controllo le celle vicine
	int count = 0;
	//su
	count += check_neighbour_open(maze, length, i - row_length);
	//giu
	count += check_neighbour_open(maze, length, i + row_length);
	//dx
	count += check_neighbour_open(maze, length, i + 1);
	//sx
	count += check_neighbour_open(maze, length, i - 1);
	//attendo che tutti abbiano fatto
	__synchthreads();
	if(count == 1){
		maze[i] = WALL;
		*again = true;
	}

}

void CPU_cellular_automata_solver(int *maze, int length, int row_length){
	bool again = true;
	int i;
	while(again){
		//risetto il booleano
		again = false;
		//per ogni cella
		for(i=0; i < length; i++){
			//controllo le celle vicine
			int count = 0;
			//su
			count += check_neighbour_open(maze, length, i - row_length);
			//giu
			count += check_neighbour_open(maze, length, i + row_length);
			//dx
			count += check_neighbour_open(maze, length, i + 1);
			//sx
			count += check_neighbour_open(maze, length, i - 1);
			//se ho solo 1 vicino open
			if(count == 1){
				maze[i] = WALL;
				again = true;
			}
		}
	}
}

void GPU_cellular_automata_solver(int *maze, int length, int row_length){
	bool again = true;
	int *dev_maze;
	bool *dev_again;
	//copio su device
	cudaMemcpy(dev_maze,maze,sizeof(int)*length,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_again, &again, sizeof(bool), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	while(again){
		//itero esecuzione
		GPU_check_neighbour_open<<<length/row_length, row_length>>>(dev_maze, length, row_length, dev_again);
		//attendo
		cudaDeviceSynchronize();
		//copio su host
		cudaMemcpy(&again, dev_again, sizeof(bool), cudaMemcpyDeviceToHost);
	}
	//terminata esecuzione su gpu. copio risultato
	cudaMemcpy(maze, dev_maze, sizeof(int) * length, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}