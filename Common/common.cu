#include <stdlib.h>
#include <stdio.h>
#include "../Header/node.h"
#include "../Header/common.h"

// fill all with wall
void FillWall(int *array, int dimension){
	int i;
	for(i=0; i < dimension; i++){
		array[i] = WALL;
	}
}

void PrintMaze(int *array, int width, int height){
	int i,j;
	for(i = 0; i < height; i++){
		for(j = 0; j < width; j++){
			cout << array[i*width + j];
		}
		cout << endl;
	}
	cout << endl;
}

// generate  matrix of ints from a matrix of nodes
int *FromNodeToGrid(struct Node *nodes, int *grid, int width, int height) {
    int i, j;
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			grid[width * i + j] = nodes[width * i + j].isSpace;
		}
	}
	return grid;
}
