#ifndef _common_
#define _common_ 1

#include <iostream>
#include "union.h"
#include "node.h"

#define WALL 0
#define OPEN 1
#define OBJECTIVE 2

#define MAX_THREAD 1000

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
 }                                                                 \
}

// fill all with wall
void FillWall(int *array, int dimension);

//fill all opened
void FillOpen(int *array, int dimension);

void PrintMaze(int *array, int width, int height);

__global__ void GPU_FillWall(int *array, int width, int dimension, int offset);

__global__ void GPU_FillOpen(int *array, int width, int dimension, int offset);

void FindStartEnd(int *maze, int length, int *start, int *end);

#endif
