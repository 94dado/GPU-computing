#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "Header/common.h"
#include "Header/backtracker_maze_generator.h"
#include "Header/cellular_automata_solver.h"
#include "Header/dfs_maze_generator.h"
#include "Header/wall_follower_maze_solver.h"

#define DIM 25
#define SIDE 5

int main(){
	//generate
	int maze[DIM];
    CPU_backtracker_maze_generator(maze, SIDE, SIDE);

    //solve
    maze[0] = OBJECTIVE;
    maze[DIM-1] = OBJECTIVE;
    PrintMaze(maze,SIDE,SIDE);
    GPU_wall_follower_maze_solver(maze,0, DIM-1,SIDE,SIDE);
    cout << "soluzione" << endl;
    PrintMaze(maze,SIDE,SIDE);
	return 0;
}
