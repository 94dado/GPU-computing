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

void CPU_perfect_maze_solver(int *maze, int length, int row_length){
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
	cudaDeviceSynchronize();
	if(count == 1){
		maze[i] = WALL;
		*again = true;
	}

}

void GPU_perfect_maze_solver(){
	bool again = true;
	//da finire
}
