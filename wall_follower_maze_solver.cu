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
void setupDirections(int *directions, int width){
	directions[RIGHT] = 1;
	directions[LEFT] = -1;
	directions[UP] = -width;
	directions[DOWN] = width;
}

//function that return the new position after a movement
int turn(int pos, int movement){
	return pos + directions[movement];
}

//function that check if the movement is possible
bool is_valid_turn(int *maze, int pos, int newPos, int width, int length){
	//check if new position is inside the maze
	if(newPos < 0 || newPos >= length) return false;
	//check if the old position was at rightmost with a right move
	if(pos % width == width - 1 && newPos == pos+1) return false;
	//check if the old position was at leftmost with a left move
	if(pos % width == 0 && newPos == pos - 1) return false;
	//check if the newPosition is not a wall
	if(maze[newPos] == WALL) return false;
	//everything is ok!
	return true;
}

//function that checks if an array contains a value
bool array_contains(int *array, int value, int size){
	for(int i = 0; i < size; i++){
		if(array[i] == value){
			return true;
		}
	}
	return false;
}

//function that save the solution in the maze
void update_maze_with_solution(int *maze, int width, int height, int start, int end, int *solution, int solution_size){
	FillWall(maze, width * height);
	for(int i = 0; i < solution_size; i++){
		maze[solution[i]] = OPEN;
	}
	maze[start] = OBJECTIVE;
	maze[end] = OBJECTIVE;
}

//function of the algorithms
void CPU_wall_follower_maze_solver(int *maze, int start, int end, int width, int height){
	setupDirections(directions, width);
	//counter of done moves
	int count_moves = 0;
	//max possible moves
	int maxMoves = (width*height) + 2;
	//list of cells visited
	int moves[maxMoves];
	//I have found the exit?
	bool done = false;
	moves[count_moves++] = start;
	while (!done && count_moves < maxMoves){
//		cout << "---------------" << endl;
//		cout << "moves: ";
//		PrintMaze(moves, count_moves, 1);
		int move;
		for(int i = 0; i < directions_length; i++){
			move = turn(moves[count_moves-1], i);
			//check if the current move is valid
			if(is_valid_turn(maze, moves[count_moves -1], move, width, width* height)){
//				cout << "found movement!!" << endl;
//				cout << "move: " << move << endl;
				//save the movement. If i'm going back, I don't have to save this position, but to delete the last
				if(array_contains(maze, move, count_moves)){
//					cout << "i have to get back" << endl;
					count_moves--;
				}else{
//					cout << "go ahead!" << endl;
					moves[count_moves++] = move;
				}
				if(moves[count_moves-1] == end){
//					cout << "reached the end" << endl;
					//finish!!!!
					done = true;
				}
				break;
			}
		}
	}
	//check if I have found a solution or not
	if(done){
		//save the solution in the maze
		update_maze_with_solution(maze, width, height, start, end, moves, count_moves);
	}
	else printf("solution not found \n");
}

__global__ void GPU_array_contains(int *maze, int move, int size, bool *result){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < size){
		if(maze[index] == move){
			*result = true;
		}
	}
}

__global__ void GPU_fill_solution(int *maze, int *solution, int solution_size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < solution_size){
		maze[index] = OPEN;
	}
}

void GPU_update_maze_with_solution(int *maze, int *dev_maze, int width, int height, int start, int end, int *solution, int solution_size){
	//set all to 0
	GPU_FillWall<<<height, width>>>(dev_maze, width, width*height);
	cudaDeviceSynchronize();
	cudaMemcpy(maze, dev_maze, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
	cout << "maze dopo GPUFillWall" << endl;
	PrintMaze(maze, width, height);
	//set to OPEN only the correct path
	GPU_fill_solution<<<1, solution_size>>>(dev_maze, solution, solution_size);
	cudaDeviceSynchronize();
	cudaDeviceReset();
	cudaMemcpy(maze, dev_maze, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
	cout << "maze dopo GPU_fill_solution" <<  endl;
	PrintMaze(maze, width, height);
	//set to OBJECTIVE the start and end
	maze[start] = OBJECTIVE;
	maze[end] = OBJECTIVE;

}

void GPU_wall_follower_maze_solver(int *maze, int start, int end, int width, int height){
	int *dev_maze;
	cudaMalloc(&dev_maze, sizeof(int) * width * height);
	cudaMemcpy(dev_maze, maze,sizeof(int) * width * height, cudaMemcpyHostToDevice);
	setupDirections(directions, width);
		//counter of done moves
		int count_moves = 0;
		//max possible moves
		int maxMoves = (width*height) + 2;
		//list of cells visited
		int moves[maxMoves];
		//I have found the exit?
		bool done = false;
		moves[count_moves++] = start;
		while (!done && count_moves < maxMoves){
	//		cout << "---------------" << endl;
	//		cout << "moves: ";
	//		PrintMaze(moves, count_moves, 1);
			int move;
			for(int i = 0; i < directions_length; i++){
				move = turn(moves[count_moves-1], i);
				//check if the current move is valid
				if(is_valid_turn(maze, moves[count_moves -1], move, width, width* height)){
	//				cout << "found movement!!" << endl;
	//				cout << "move: " << move << endl;
					//save the movement. If i'm going back, I don't have to save this position, but to delete the last
					bool alreadyInside = false;
					bool *dev_alreadyInside;
					cudaMalloc(&dev_alreadyInside, sizeof(bool));
					cudaMemcpy(dev_alreadyInside, &alreadyInside, sizeof(bool), cudaMemcpyHostToDevice);
					GPU_array_contains<<<height, width>>>(dev_maze, move, width*height, dev_alreadyInside);
					cudaDeviceSynchronize();
					cudaMemcpy(&alreadyInside, dev_alreadyInside, sizeof(bool), cudaMemcpyDeviceToHost);
					if(alreadyInside){
	//					cout << "i have to get back" << endl;
						count_moves--;
					}else{
	//					cout << "go ahead!" << endl;
						moves[count_moves++] = move;
					}
					if(moves[count_moves-1] == end){
	//					cout << "reached the end" << endl;
						//finish!!!!
						done = true;
					}
					break;
				}
			}
		}
		//check if I have found a solution or not
			if(done){
				//save the solution in the maze
				GPU_update_maze_with_solution(maze,dev_maze, width, height, start, end, moves, count_moves);
			}
			else printf("solution not found \n");
}
