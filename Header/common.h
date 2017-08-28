#ifndef _common_
#define _common_ 1

#include <iostream>
#include "union.h"
#include "node.h"

#define WALL 0
#define OPEN 1
#define OBJECTIVE 2

// fill all with wall
void FillWall(int *array, int dimension);

//fill all opened
void FillOpen(int *array, int dimension);

void PrintMaze(int *array, int width, int height);

__global__ void GPU_FillWall(int *array, int width, int dimension);

__global__ void GPU_FillOpen(int *array, int width, int dimension);

void FindStartEnd(int *maze, int length, int *start, int *end);

#endif
