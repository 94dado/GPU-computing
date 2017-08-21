#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define WALL 0
#define OPEN 1
#define OBJECTIVE 2

#define bool int
#define true 1
#define false 0


void fill_of_wall(int* array, int dimension){
	int i;
	for(i=0; i < dimension; i++){
		array[i] = WALL;
	}
}

void print_maze(int *array, int width, int height){
	int i,j;
	for(i = 0; i < height; i++){
		for(j = 0; j < width; j++){
			printf("%d",array[i*width + j]);
		}
		printf("\n");
	}
	printf("\n");
}
