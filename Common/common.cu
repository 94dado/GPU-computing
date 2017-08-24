#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "../Header/common.h"

using namespace std;

// fill all with wall
void FillWall(int *array, int dimension){
	int i;
	for(i=0; i < dimension; i++){
		array[i] = WALL;
	}
}

//fill all opened
void FillOpen(int *array, int dimension){
	int i;
		for(i=0; i < dimension; i++){
			array[i] = OPEN;
		}
}

void PrintMaze(int *array, int width, int height){
	int i,j;
	for(i = 0; i < height; i++){
		for(j = 0; j < width; j++){
			cout << array[i*width + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

__global__ void GPU_FillWall(int *array, int width, int dimension){
	int index = width * blockIdx.x + threadIdx.x;
	if(index < dimension){
		array[index] = WALL;
	}
}

void FindStartEnd(int *maze, int length, int *start, int *end){
	int count = 0;
	*start = *end = -1;
	for(int i = 0; i < length; i++){
		if(maze[i] == OBJECTIVE){
			if(count == 0){
				*start = i;
				count++;
			}else if (count == 1){
				*end = i;
				break;
			}
		}
	}
}
