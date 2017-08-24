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

#define DIM 846400
#define SIDE 920

int main(){
	//generate
	int maze[DIM];
	int mazeGPU[DIM];
    CPU_dfs_maze_generator(maze, SIDE + 1, SIDE + 1);
    memcpy(mazeGPU,maze,DIM * sizeof(int));


//    PrintMaze(maze,SIDE,SIDE);
    int start, end;
    FindStartEnd(maze, DIM, &start, &end);
//    cout << "start: " << start << ", end: " << end << endl;
    double startCPU, finishCPU, startGPU, finishGPU;
    startCPU = seconds();
    CPU_wall_follower_maze_solver(maze, start, end, SIDE, SIDE);
    finishCPU = seconds();

    startGPU = seconds();
    GPU_wall_follower_maze_solver(mazeGPU, start, end, SIDE, SIDE);
    finishGPU = seconds();

//    cout << "soluzione CPU" << endl;
//    PrintMaze(maze,SIDE,SIDE);
//    cout << "soluzione GPU" << endl;
//    PrintMaze(mazeGPU,SIDE,SIDE);

    cout << "tempo CPU: " << finishCPU - startCPU << endl;
    cout << "tempo GPU: " << finishGPU - startGPU << endl;
	return 0;
}
