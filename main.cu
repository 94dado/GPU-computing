#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include "Header/CPU_time.h"

#include "Header/common.h"
#include "Header/backtracker_maze_generator.h"
#include "Header/cellular_automata_solver.h"
#include "Header/dfs_maze_generator.h"
#include "Header/wall_follower_maze_solver.h"
#include "Header/division_maze_generator.h"

#define DIM 64
#define SIDE 8

int main(){
	//generate
	int maze[DIM];
	CPU_division_maze_generator(maze, SIDE, SIDE);
	PrintMaze(maze,SIDE,SIDE);
	return 0;
}
