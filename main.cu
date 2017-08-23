#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "Header/common.h"
#include "Header/backtracker_maze_generator.h"
#include "Header/cellular_automata_solver.h"

#define DIM 25
#define SIDE 5
int main(){
	//generate
	int maze[DIM];
	printf("maze cpu\n");
	CPU_backtracker_maze_generator(maze,SIDE,SIDE);
	maze[0] = OBJECTIVE;
	maze[DIM-1] = OBJECTIVE;
	print_maze(maze,SIDE,SIDE);
	printf("solve cpu\n\n");
	CPU_cellular_automata_solver(maze, DIM, SIDE);
	print_maze(maze,SIDE,SIDE);
	printf("maze gpu\n\n");
	GPU_backtracker_maze_generator(maze,SIDE,SIDE);
	print_maze(maze,SIDE,SIDE);
	printf("solve gpu\n\n");
	GPU_cellular_automata_solver(maze, DIM,SIDE);


	return 0;
}
