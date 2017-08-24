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
    CPU_dfs_maze_generator(maze, SIDE + 1, SIDE + 1);

    //solve
    PrintMaze(maze,SIDE,SIDE);
    int start, end;
    FindStartEnd(maze, DIM, &start, &end);
//    cout << "start: " << start << ", end: " << end << endl;
    GPU_wall_follower_maze_solver(maze, start, end, SIDE, SIDE);

    cout << "soluzione" << endl;
    PrintMaze(maze,SIDE,SIDE);
	return 0;
}
