#include <stdlib.h>
#include <stdio.h>
#include "../Header/node.h"
#include "../Header/vector.h"
#include "../Header/union.h"
#include "../Header/common.h"

// fill all with wall
void fill_of_wall(int *array, int dimension){
	int i;
	for(i=0; i < dimension; i++){
		array[i] = WALL;
	}
}

void print_maze(int *array, int width, int height){
	int i,j;
	for(i = 0; i < height; i++){
		for(j = 0; j < width; j++){
			printf("%d ",array[i*width + j]);
		}
		printf("\n");
	}
	printf("\n");
}

// generate  matrix of ints from a matrix of nodes
void from_node_to_grid(Node *nodes, int *grid, int width, int height) {
    int i, j;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			grid[width * i + j] = nodes[width * i + j].isSpace;
		}
    }
}

// SORT
void bubble_sort(Vector *vector) {
	int x, y;
	int tot = vector_count(vector);

	for (x = 0 ; x < (tot - 1); x++) {
		for (y = 0 ; y < tot - x - 1; y++) {
			if (vector->data[y] > vector->data[y+1]) {
				IntPair *swap = vector->data[y];
				vector->data[y] = vector->data[y + 1];
				vector->data[y + 1] = swap;
			}
		}
	}
}
