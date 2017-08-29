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
#include "Header/kruskal_maze_generator.h"
#include "Header/bfs_maze_solver.h"

#define DIM 81
#define SIDE 9

int main(){
	//generate
	int maze[DIM];
	//CPU_kruskal_maze_generator(maze, SIDE, SIDE);
	CPU_backtracker_maze_generator(maze, SIDE, SIDE);
	maze[1] = OBJECTIVE;
	maze[79] = OBJECTIVE;
	PrintMaze(maze,SIDE,SIDE);
	CPU_bfs_maze_solver(maze,SIDE,SIDE);
	return 0;
}
