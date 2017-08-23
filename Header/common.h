#ifndef _common_
#define _common_ 1

#include <iostream>
#include "node.h"

using namespace std;

#define WALL 0
#define OPEN 1
#define OBJECTIVE 2

// fill all with wall
void FillWall(int *array, int dimension);

void PrintMaze(int *array, int width, int height);

// generate  matrix of ints from a matrix of nodes
int *FromNodeToGrid(struct Node *nodes, int *grid, int width, int height);

#endif
