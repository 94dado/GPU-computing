#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "Header/common.h"
#include "Header/backtracker_maze_generator.h"
#include "Header/cellular_automata_solver.h"
#include "Header/dfs_maze_generator.h"

#define DIM 100
#define SIDE 10

int main(){
	//generate
	int maze[DIM];
    CPU_dfs_maze_generator(maze, SIDE, SIDE);
    PrintMaze(maze,SIDE,SIDE);
	return 0;
}
