#include <cuda_runtime.h>
#include <time.h>
#include <iostream>
#include "Header/CPU_time.h"

#include "Header/common.h"
#include "Header/backtracker_maze_generator.h"
#include "Header/cellular_automata_solver.h"
#include "Header/dfs_maze_generator.h"
#include "Header/wall_follower_maze_solver.h"
#include "Header/division_maze_generator.h"
#include "Header/kruskal_maze_generator.h"
#include "Header/bfs_maze_solver.h"
#include "Header/recursive_maze_solver.h"

#define NUMBER_OF_TEST 3

//DFS: print with side-1

void dfs(int side, int *maze1, int *maze2) {
	cout << endl << "DFS MAZE GENERATOR WITH CPU" << endl;
	time_t start = time(0);
	CPU_dfs_maze_generator(maze1, side, side);
	cout << endl << "The time of computation is " << difftime(time(0), start) << " seconds" << endl;
	PrintMaze(maze1,side-1,side-1);
	cout << endl << "DFS MAZE GENERATOR WITH GPU" << endl;
	start = time(0);
	GPU_dfs_maze_generator(maze2,side,side);
	cout << endl << "The time of computation is " << difftime(time(0), start) << " seconds" << endl;
	PrintMaze(maze2,side-1,side-1);
	cudaDeviceReset();
}

void cellular(int side, int *maze1, int *maze2) {
	cout << endl << "CELLULAR AUTOMATA MAZE SOLVER WITH CPU" << endl;
	time_t start = time(0);
	CPU_cellular_automata_solver(maze1, side, side);
	cout << endl << "The time of computation is " << difftime(time(0), start) << " seconds" << endl;
	PrintMaze(maze1,side,side);
	cout << endl << "CELLULAR AUTOMATA MAZE SOLVER WITH GPU" << endl;
	start = time(0);
	GPU_cellular_automata_solver(maze2,side,side);
	cout << endl << "The time of computation is " << difftime(time(0), start) << " seconds" << endl;
	PrintMaze(maze2,side,side);
	cudaDeviceReset();
}

void backtracker(int side, int *maze1, int *maze2) {
	cout << endl << "BACKTRACKER MAZE GENERATOR WITH CPU" << endl;
	time_t start = time(0);
	CPU_backtracker_maze_generator(maze1, side, side);
	cout << endl << "The time of computation is " << difftime(time(0), start) << " seconds" << endl;
	PrintMaze(maze1,side,side);
	cout << endl << "BACKTRACKER MAZE GENERATOR WITH GPU" << endl;
	start = time(0);
	GPU_backtracker_maze_generator(maze2,side,side);
	cout << endl << "The time of computation is " << difftime(time(0), start) << " seconds" << endl;
	PrintMaze(maze2,side,side);
	cudaDeviceReset();
}

void bfs(int side, int *maze1, int *maze2) {
	cout << endl << "BFS MAZE SOLVER WITH CPU" << endl;
	time_t start = time(0);
	CPU_bfs_maze_solver(maze1, side, side);
	cout << endl << "The time of computation is " << difftime(time(0), start) << " seconds" << endl;
	PrintMaze(maze1,side,side);
	cout << endl << "BFS MAZE SOLVER WITH GPU" << endl;
	start = time(0);
	GPU_bfs_maze_solver(maze2,side,side);
	cout << endl << "The time of computation is " << difftime(time(0), start) << " seconds" << endl;
	PrintMaze(maze2,side,side);
	cudaDeviceReset();
}

void division(int side, int *maze1, int *maze2) {
	cout << endl << "DIVISION MAZE GENERATOR WITH CPU" << endl;
	time_t start = time(0);
	CPU_division_maze_generator(maze1, side, side);
	cout << endl << "The time of computation is " << difftime(time(0), start) << " seconds" << endl;
	PrintMaze(maze1,side,side);
	cout << endl << "DIVISION MAZE GENERATOR WITH GPU" << endl;
	start = time(0);
	GPU_division_maze_generator(maze2,side,side);
	cout << endl << "The time of computation is " << difftime(time(0), start) << " seconds" << endl;
	PrintMaze(maze2,side,side);
	cudaDeviceReset();
}

void recursive(int side, int *maze1, int *maze2) {
	cout << endl << "RECURSIVE MAZE SOLVER WITH CPU" << endl;
	time_t start = time(0);
	CPU_recursive_maze_solver(maze1, side, side);
	cout << endl << "The time of computation is " << difftime(time(0), start) << " seconds" << endl;
	PrintMaze(maze1,side,side);
	cout << endl << "RECURSIVE MAZE SOLVER WITH GPU" << endl;
	start = time(0);
	GPU_recursive_maze_solver(maze2,side,side);
	cout << endl << "The time of computation is " << difftime(time(0), start) << " seconds" << endl;
	PrintMaze(maze2,side,side);
	cudaDeviceReset();
}

void kruskal(int side, int *maze1, int *maze2) {
	cout << endl << "KRUSKAL MAZE GENERATOR WITH CPU" << endl;
	time_t start = time(0);
	CPU_kruskal_maze_generator(maze1, side, side);
	cout << endl << "The time of computation is " << difftime(time(0), start) << " seconds" << endl;
	PrintMaze(maze1,side,side);
	cout << endl << "KRUSKAL MAZE GENERATOR WITH GPU" << endl;
	start = time(0);
	GPU_kruskal_maze_generator(maze2,side,side);
	cout << endl << "The time of computation is " << difftime(time(0), start) << " seconds" << endl;
	PrintMaze(maze2,side,side);
	cudaDeviceReset();
}

void wall_follower(int side, int startP, int endP, int *maze1, int *maze2) {
	cout << endl << "WALL FOLLOWER MAZE SOLVER WITH CPU" << endl;
	time_t start = time(0);
	CPU_wall_follower_maze_solver(maze1, startP, endP, side, side);
	cout << endl << "The time of computation is " << difftime(time(0), start) << " seconds" << endl;
	PrintMaze(maze1,side,side);
	cout << endl << "WALL FOLLOWER MAZE SOLVER WITH GPU" << endl;
	start = time(0);
	GPU_wall_follower_maze_solver(maze2,startP, endP, side,side);
	cout << endl << "The time of computation is " << difftime(time(0), start) << " seconds" << endl;
	PrintMaze(maze2,side,side);
	cudaDeviceReset();
}

//int main(){
//	int side = 1;
//	for (int i = 0; i == NUMBER_OF_TEST; i++) {
//		// setted size of the matrix
//		side *= 10;
//		int dim = side * side;
//		int maze1[dim];
//		int maze2[dim];
//
//		//DFS
//		dfs(side, maze1, maze2);
//		// Cellular automata
//		cellular(side, maze1,maze2);
//
//		// reset maze
//		memset(maze1, 0, sizeof(maze1));
//		memset(maze2, 0, sizeof(maze2));
//		maze1[1] = OBJECTIVE;
//		maze1[dim-2] = OBJECTIVE;
//		maze2[1] = OBJECTIVE;
//		maze2[dim-2] = OBJECTIVE;
//
//		// Backtracker
//		backtracker(side, maze1, maze2);
//		// Wall follower
//		wall_follower(side, 1, dim-2, maze1,maze2);
//
//		// reset maze
//		memset(maze1, 0, sizeof(maze1));
//		memset(maze2, 0, sizeof(maze2));
//
//		// Division
//		division(side, maze1, maze2);
//		// BFS
//		bfs(side, maze1,maze2);
//
//		// reset maze
//		memset(maze1, 0, sizeof(maze1));
//		memset(maze2, 0, sizeof(maze2));
//
//		// Kruskal
//		kruskal(side, maze1, maze2);
//		// Recursive
//		recursive(side, maze1,maze2);
//
//	}
//	return 0;
//}

int main() {
	int side = 8;
	int dim = side * side;
	int maze[dim];
	GPU_dfs_maze_generator(maze,side,side);
	PrintMaze(maze,side-1,side-1);
	return 0;
}
