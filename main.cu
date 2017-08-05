#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define HEIGHT 5
#define WIDTH 5

#define WALL 0
#define OPEN 1
#define OBJECTIVE 2

int maze[WIDTH*HEIGHT];

void fullWall(int* array, int dimension){
	int i;
	for(i=0; i < dimension; i++){
		array[i] = WALL;
	}
}

void printArray(int *array, int width, int height){
	int i,j;
	for(i = 0; i < height; i++){
		for(j = 0; j < width; j++){
			printf("%d",array[i*width + j]);
		}
		printf("\n");
	}
	printf("\n");
}

int main(int argc, char* argv[]){
	fullWall(maze, WIDTH * HEIGHT);
	printArray(maze,WIDTH,HEIGHT);
}

