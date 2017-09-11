#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "./Header/common.h"

#include <unistd.h>
using namespace std;

#define RIGHT 0
#define DOWN 1
#define LEFT 2
#define UP 3

#define directions_length 4
int directions[directions_length];

//function to initialize directions
void setupDirections(int *directions, int width) {
	directions[RIGHT] = 1;
	directions[LEFT] = -1;
	directions[UP] = -width;
	directions[DOWN] = width;
}

//function that return the new position after a movement
int turn(int pos, int movement) {
	return pos + directions[movement];
}

//function that check if the movement is possible
bool is_valid_turn(int *maze, int pos, int newPos, int width, int length) {
	//check if new position is inside the maze
	if (newPos < 0 || newPos >= length)
		return false;
	//check if the old position was at rightmost with a right move
	if (pos % width == width - 1 && newPos == pos + 1)
		return false;
	//check if the old position was at leftmost with a left move
	if (pos % width == 0 && newPos == pos - 1)
		return false;
	//check if the newPosition is not a wall
	if (maze[newPos] == WALL)
		return false;
	//everything is ok!
	return true;
}

//function that checks if an array contains a value
bool array_contains(int *array, int value, int size) {
	for (int i = 0; i < size; i++) {
		if (array[i] == value) {
			return true;
		}
	}
	return false;
}

//function that save the solution in the maze
void update_maze_with_solution(int *maze, int width, int height, int start,
		int end, int *solution, int solution_size) {
	FillWall(maze, width * height);
	for (int i = 0; i < solution_size; i++) {
		maze[solution[i]] = OPEN;
	}
	maze[start] = OBJECTIVE;
	maze[end] = OBJECTIVE;
}

//function of the algorithms
void CPU_wall_follower_maze_solver(int *maze, int start, int end, int width,
		int height) {
	setupDirections(directions, width);
	//counter of done moves
	int count_moves = 0;
	//counter of all cells visited
	int count_already = 0;
	//max possible moves
	int maxMoves = (width * height) + 2;
	//path that i'm following
	int moves[maxMoves];
	//all cell visited
	int already_seen[width * height];
	//I have found the exit?
	bool done = false;
	moves[count_moves++] = start;
	already_seen[count_already++] = start;
	while (!done && count_moves < maxMoves) {
//		cout << "---------------" << endl;
//		cout << "moves: ";
//		PrintMaze(moves, count_moves, 1);
		int move;
		int dests[directions_length];
		for (int i = 0; i < directions_length; i++) {
			move = turn(moves[count_moves - 1], i);
			//check if the current move is valid
			if (is_valid_turn(maze, moves[count_moves - 1], move, width,
					width * height)) {
//				cout << "found possible movement!!" << endl;
//				cout << "move: " << move << endl;
				dests[i] = move;
			} else {
				dests[i] = -1;
			}
		}
		//all possible dests calculated. now choose where to go
		bool backtrackAvailable = false;
		bool chosen = false;
		for (int i = 0; i < directions_length; i++) {
			if (dests[i] != -1) {
				//if is a valid destination
				if (!array_contains(already_seen, dests[i], count_already)) {
					//new cell never visited before. It's ok for me
					moves[count_moves++] = dests[i];
					already_seen[count_already++] = dests[i];
					chosen = true;
					break;
				} else {
					//cell already visited. Is backtracking?
					if (array_contains(moves, dests[i], count_moves)) {
						//i'm trying to backtrack. That's my last choice
						backtrackAvailable = true;
					}
				}
			}
		}
		if (!chosen && backtrackAvailable) {
			//i can only go back
			count_moves--;
//			cout << "backtrack: " << moves[count_moves-1] << endl;
		} else if (chosen) {
//			cout << "move made: " << moves[count_moves-1] << endl;
			if (moves[count_moves - 1] == end) {
//				cout << "reached the end" << endl;
				//finish!!!!
				done = true;
				break;
			}
		} else {
			//no moves available. I'm simply fucked
//			cout << "no moves available. FUCK!" << endl;
			return;
		}
	}
	//check if I have found a solution or not
	if (done) {
		//save the solution in the maze
		update_maze_with_solution(maze, width, height, start, end, moves,
				count_moves);
	} else {
		//no solution
		cout << "solution not found" << endl;
	}
}

__global__ void GPU_array_contains(int *maze, int move, int size,
		bool *result) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
//	printf("-----------\nindex: %d\nmaze[index]: %d\nmove: %d\n",index, maze[index], move);
	if (index < size) {
		if (maze[index] == move) {
//			printf("set true alreadyInside\n");
			*result = true;
		}
	}
}

__global__ void GPU_fill_solution(int *maze, int *solution, int solution_size) {
	int index = threadIdx.x;
	if (index < solution_size) {
		maze[solution[index]] = OPEN;
	}
}

void GPU_update_maze_with_solution(int *maze, int *dev_maze, int width,
		int height, int start, int end, int *solution, int solution_size) {
	//set all to 0
	int *dev_solution;
	GPU_FillWall<<<height, width>>>(dev_maze, width, width*height);
	cudaDeviceSynchronize();
	cudaMemcpy(maze, dev_maze, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
//	PrintMaze(maze, width, height);
	//set to OPEN only the correct path
	cudaMalloc(&dev_solution, sizeof(int) * solution_size);
	cudaMemcpy(dev_solution, solution, sizeof(int) * solution_size, cudaMemcpyHostToDevice);
	GPU_fill_solution<<<1, solution_size>>>(dev_maze, dev_solution, solution_size);
	cudaDeviceSynchronize();
	cudaMemcpy(maze, dev_maze, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
	//set to OBJECTIVE the start and end
	maze[start] = OBJECTIVE;
	maze[end] = OBJECTIVE;
}

__device__ int GPU_turn(int cell, int i, int* directions) {
	return cell + directions[i];
}

__device__ bool GPU_is_valid_turn(int *maze, int pos, int newPos, int width,
		int length) {
	//check if new position is inside the maze
	if (newPos < 0 || newPos >= length)
		return false;
	//check if the old position was at rightmost with a right move
	if (pos % width == width - 1 && newPos == pos + 1)
		return false;
	//check if the old position was at leftmost with a left move
	if (pos % width == 0 && newPos == pos - 1)
		return false;
	//check if the newPosition is not a wall
	if (maze[newPos] == WALL)
		return false;
	//everything is ok!
	return true;
}

__global__ void GPU_calculate_moves(int cell, int *maze, int width, int length, int *directions, int *dests) {
	int i = threadIdx.x;
	int move = GPU_turn(cell, i, directions);
//	printf("move: %d\n",move);
	if (GPU_is_valid_turn(maze, cell, move, width, length)) {
		dests[i] = move;
	} else {
		dests[i] = -1;
	}
//	printf("dests[%d]: %d\n",i,dests[i]);
	__syncthreads();
}

void GPU_wall_follower_maze_solver(int *maze, int start, int end, int width,int height) {
//	cout << "start gpu_wall_follower" << endl;
	int *dev_maze;
	cudaMalloc(&dev_maze, sizeof(int) * width * height);
	cudaMemcpy(dev_maze, maze, sizeof(int) * width * height, cudaMemcpyHostToDevice);

	setupDirections(directions, width);

	int *dev_directions;
	cudaMalloc(&dev_directions, sizeof(int) * directions_length);
	cudaMemcpy(dev_directions, directions, sizeof(int) * directions_length,	cudaMemcpyHostToDevice);

	//counter of done moves
	int count_moves = 0;
	//counter of all cells visited
	int count_already = 0;
	//max possible moves
	int maxMoves = (width * height) + 2;
	//path that i'm following
	int *moves = new int[maxMoves];
	//all cell visited
	int *already_seen = new int[width * height];

	int *dev_already_seen;
	cudaMalloc(&dev_already_seen, sizeof(int) * width * height);

	//I have found the exit?
	bool done = false;
	moves[count_moves++] = start;
	already_seen[count_already++] = start;
	while (!done && count_moves < maxMoves) {
//		cout << "---------------" << endl;
//		cout << "moves: ";
//		PrintMaze(moves, count_moves, 1);
		int dests[directions_length];

		int *dev_dests;
		cudaMalloc(&dev_dests, sizeof(int) * directions_length);
		GPU_calculate_moves<<<1,directions_length>>>(moves[count_moves - 1], dev_maze, width, width* height, dev_directions, dev_dests);
		cudaDeviceSynchronize();
		cudaMemcpy(dests, dev_dests, sizeof(int) * directions_length, cudaMemcpyDeviceToHost);
//		cout << "obtained dests from GPU ";
//		PrintMaze(dests, 1, directions_length);

		//all possible dests calculated. now choose where to go
		bool backtrackAvailable = false;
		bool chosen = false;
		for (int i = 0; i < directions_length; i++) {
			if (dests[i] != -1) {
				//if is a valid destination
				if (!array_contains(already_seen, dests[i], count_already)) {
					//new cell never visited before. It's ok for me
					moves[count_moves++] = dests[i];
					already_seen[count_already++] = dests[i];
					chosen = true;
					break;
				} else {
					//cell already visited. Is backtracking?
					if (array_contains(moves, dests[i], count_moves)) {
						//i'm trying to backtrack. That's my last choice
						backtrackAvailable = true;
					}
				}
			}
		}
		if (!chosen && backtrackAvailable) {
			//i can only go back
			count_moves--;
//			cout << "backtrack: " << moves[count_moves-1] << endl;
		} else if (chosen) {
//			cout << "move made: " << moves[count_moves-1] << endl;
			if (moves[count_moves - 1] == end) {
//				cout << "reached the end" << endl;
				//finish!!!!
				done = true;
				break;
			}
		} else {
			//no moves available. I'm simply fucked
			cout << "no moves available" << endl;
			return;
		}
	}
	//check if I have found a solution or not
	if (done) {
		//save the solution in the maze
//		cout << "solution found" << endl;
		GPU_update_maze_with_solution(maze, dev_maze, width, height, start, end, moves, count_moves);
	} else {
		//no solution
		cout << "solution not found" << endl;
	}
	free(moves);
	free(already_seen);
	cudaDeviceReset();
}
