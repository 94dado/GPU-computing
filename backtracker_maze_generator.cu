#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "Header/common.h"

/*  Carve the maze starting at x, y. */
void CarveMaze(int *maze, int width, int height, int x, int y) {

   int x1, y1;
   int x2, y2;
   int dx, dy;
   int dir, count;

   dir = rand() % 4;
   count = 0;
   while(count < 4) {
      dx = 0; dy = 0;
      switch(dir) {
      case 0:  dx = 1;  break;
      case 1:  dy = 1;  break;
      case 2:  dx = -1; break;
      default: dy = -1; break;
      }
      x1 = x + dx;
      y1 = y + dy;
      x2 = x1 + dx;
      y2 = y1 + dy;
      if(   x2 > 0 && x2 < width && y2 > 0 && y2 < height
         && maze[y1 * width + x1] == WALL && maze[y2 * width + x2] == WALL) {
         maze[y1 * width + x1] = OPEN;
         maze[y2 * width + x2] = OPEN;
         x = x2; y = y2;
         dir = rand() % 4;
         count = 0;
      } else {
         dir = (dir + 1) % 4;
         count += 1;
      }
   }

}

/* Generate maze in matrix maze with size width, height. */
void CPU_backtracker_maze_generator(int *maze, int width, int height) {
   int x, y;

   srand(time(NULL));
   /* Initialize the maze. */
   for(x = 0; x < width * height; x++) {
      maze[x] = WALL;
   }
   maze[1 * width + 1] = OPEN;

   /* Carve the maze. */
   for(y = 1; y < height; y += 2) {
      for(x = 1; x < width; x += 2) {
         CarveMaze(maze, width, height, x, y);
      }
   }

   /* Set up the entry and exit. */
   maze[0 * width + 1] = OBJECTIVE;
   maze[(height - 1) * width + (width - 2)] = OBJECTIVE;
}

__global__ void GPU_init_maze(int *maze, int length, int row_length, int offset){
	int idx = blockIdx.x * row_length + offset + threadIdx.x;
	maze[idx] = WALL;
}

__global__ void GPU_carve_maze(int *maze,int width, int height, int rand1, int rand2, int i){
//	int y = blockIdx.x;
	int y = i;
	int x = threadIdx.y;
	//only odd numbers
	if(x%2 == 0 && y%2 == 0) return;

	int x1, y1;
	int x2, y2;
	int dx, dy;
	int dir, count;

	dir = rand1;
	count = 0;
	while(count < 4) {
	 dx = 0; dy = 0;
	 switch(dir) {
		 case 0:  dx = 1;  break;
		 case 1:  dy = 1;  break;
		 case 2:  dx = -1; break;
		 default: dy = -1; break;
	 }
	 x1 = x + dx;
	 y1 = y + dy;
	 x2 = x1 + dx;
	 y2 = y1 + dy;
	 if(x2 > 0 && x2 < width && y2 > 0 && y2 < height
		&& maze[y1 * width + x1] == WALL && maze[y2 * width + x2] == WALL) {
		 maze[y1 * width + x1] = OPEN;
		 maze[y2 * width + x2] = OPEN;
		 x = x2; y = y2;
		 dir = rand2;
		 count = 0;
	 } else {
		 dir = (dir + 1) % 4;
		 count += 1;
	 }
	}
}
void GPU_backtracker_maze_generator(int *maze, int width, int height){
	int *dev_maze;
	int length = width * height;
	srand(time(NULL));
	//initialize the maze
	cudaMalloc(&dev_maze, sizeof(int) * length);
	cudaMemcpy(dev_maze, maze, sizeof(int) * length, cudaMemcpyHostToDevice);
	int max_rec = width / MAX_THREAD;
	int offset = 0;
	for(int i = 0; i < max_rec; i++){
		GPU_init_maze<<<height, MAX_THREAD>>>(dev_maze, width* height, width, offset);
		offset = (i+1) * MAX_THREAD;
	}
	GPU_init_maze<<<height, width % MAX_THREAD>>>(dev_maze, width* height, width, offset);
	cudaDeviceSynchronize();
	cudaMemcpy(maze,dev_maze, sizeof(int) * length, cudaMemcpyDeviceToHost);

	maze[1 * width + 1] = OPEN;
	//carve the maze
	for(int y = 1; y < height; y += 2) {
	  for(int x = 1; x < width; x += 2) {
		 CarveMaze(maze, width, height, x, y);
	  }
	}

//	cudaMemcpy(dev_maze, maze, sizeof(int) * length, cudaMemcpyHostToDevice);
//	for(int i = 0; i < width; i++){	//per sicurezza lo lancio 2 volte
//		int rand1 = rand() % 4;
//		int rand2 = rand() % 4;
//		GPU_carve_maze<<<1, width>>>(dev_maze, width, height, rand1, rand2, i);
//		cudaDeviceSynchronize();
//	}
//	cudaMemcpy(maze,dev_maze, sizeof(int) * length, cudaMemcpyDeviceToHost);

	/* Set up the entry and exit. */
	maze[0 * width + 1] = OBJECTIVE;
	maze[(height - 1) * width + (width - 2)] = OBJECTIVE;
}
