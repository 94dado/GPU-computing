#ifndef _common_
#define _common_ 1

#include "node.h"
#include "vector.h"

#define WALL 0
#define OPEN 1
#define OBJECTIVE 2

#define bool int
#define true 1
#define false 0

void fill_of_wall(int *array, int dimension);
void print_maze(int *array, int width, int height);
void from_node_to_grid(Node *nodes, int *grid, int width, int height);
void bubble_sort(Vector *vector);

#endif
