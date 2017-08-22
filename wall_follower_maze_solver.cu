#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

void CPU_wall_follower_maze_solver(int *maze, int length, int row_length){
	//directions
		int north = -row_length, south = row_length, west = -1, east = 1;
		int directions[] = {east,south,west,north};
}
