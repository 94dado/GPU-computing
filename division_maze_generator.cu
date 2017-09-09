#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <iterator>
#include <stdio.h>
#include <unistd.h>
#include "time.h"
#include "Header/common.h"

using namespace std;

#define HORIZONTAL 1
#define VERTICAL 2

#define MAZE_RESOLUTION 2


int choose_orientation(int width, int height){
	if(width < height)	return HORIZONTAL;
	else if(height < width) return VERTICAL;
	else return rand()%2 + 1;
}

void PrintVector(vector <int> maze){
	for(int i = 0; i < maze.size() - 1; i++){
		cout << maze[i] << " ";
	}
	cout << endl;
}

void fill_border(int *maze, int width, int height){
	for(int a=0; a < height; a+= height-1){
		for(int b = 0; b<width; b++){
			maze[a*width + b] = WALL;
		}
	}
	for(int b=0; b < width; b+= width - 1){
		for(int a = 0; a < height; a++){
			maze[a*width + b] = WALL;
		}
	}
}

vector < int> found_possible_border(int *maze, int width, int height){
	vector <int> stack;
	//top
	int i = 1;
	for(int j=1; j < width-2;j++){
		if(maze[i*width + j] == OPEN){
			stack.push_back(j);
		}
	}

	//left
	for(int j = 1; j < height - 2; j++){
		if(maze[j*width + i] == OPEN){
			stack.push_back(j * width);
		}
	}
	//down
	i = height-2;
	for(int j=1; j < width-2;j++){
		if(maze[i*width + j] == OPEN){
			stack.push_back((i+1) * width + j);
		}
	}
	//right
	for(int j = 1; j < height - 2; j++){
		if(maze[j*width + i] == OPEN){
			stack.push_back(j * width + i + 1);
		}
	}
	return stack;
}

bool isValid(int val, int width, int height){
	int x = val % width;
	int y = val / width;
	return x > 0 && y > 0 && x < width && y < width;
}

bool isValidPath(int *maze, int x, int y, int width, int height){
	return maze[y * width + x] == OPEN && (x == 1 || x == width -2 || y == 1 || y == height -2);
}

void follow_all(int *maze, vector<int> stack, int start, int width, int height){
	int x = start % width;
	int y = start / width;
	if(isValidPath(maze,x,y,width,height)){
		stack.push_back(start);
	}
	//right
	if (isValid(start+1, width, height) && maze[start+1]==OPEN && find(stack.begin(), stack.end(), start+1) != stack.end())
		follow_all(maze,stack,start+1,width,height);
	//left
	if (isValid(start-1, width, height) && maze[start-1]==OPEN && find(stack.begin(), stack.end(), start-1) != stack.end())
		follow_all(maze,stack,start-1,width,height);
	//down
	if (isValid(start + width, width, height) && maze[start+width]==OPEN && find(stack.begin(), stack.end(), start+width) != stack.end())
		follow_all(maze,stack,start+width,width,height);
	//up
	if (isValid(start - width, width, height) && maze[start-width]==OPEN && find(stack.begin(), stack.end(), start-width) != stack.end())
			follow_all(maze,stack,start-width,width,height);

}

void add_objective(int *maze, int width, int height){
	vector <int> possible_border = found_possible_border(maze,width,height);
	bool found = false;
	PrintVector(possible_border);
	while(!found){
		int start = rand() % possible_border.size();
		vector <int> stack_positions;
		follow_all(maze, stack_positions, start, width, height);
		if (stack_positions.size()>0){
			int index = rand() % stack_positions.size();
			int end = maze[stack_positions[index]];
			maze[start] = OBJECTIVE;
			maze[end] = OBJECTIVE;
			found = true;
		}
		std::vector<int>::iterator it = stack_positions.begin();
		for (int i = 0; i < start; i++) next(it);
		stack_positions.erase(it);
	}
}

void recursive_divide(int *maze, int x, int y, int width, int height, int orientation){
//	cout << "------------------" << endl << "recursion started" << endl;
//	cout << "x: " << x << ", y: " << y << ", width: " << width << ", height: " << height << ", orientation: " << orientation << endl;
	int fromX, fromY,holeX,holeY,directionX,directionY,wall_lenght;
	int newX, newY, newWidth, newHeight;
	//check if I don't have to make this iteration
	if(width < MAZE_RESOLUTION || height < MAZE_RESOLUTION){
		return;
	}
	if(orientation == HORIZONTAL){
		//is horizontal
		fromX = x;
		if(height - 2 <= 0){
			fromY = y;
		}else{
			fromY = y + rand()%(height-2);
		}
		holeX = fromX + rand()%width;
		holeY = fromY;
		directionX = 1;
		directionY = 0;
		wall_lenght = width;

	}else{
		//is vertical
		if(width - 2 <= 0){
			fromX = x;
		}else{
			fromX = x + rand()%(width-2);
		}
		fromY = y;
		holeX = fromX;
		holeY = fromY + rand()%height;
		directionX = 0;
		directionY = 1;
		wall_lenght = height;
	}
//	cout << "data calculated:" << endl;
//	cout << "fromX: " << fromX << ", fromY: " << fromY << ", holeX: " << holeX
//			<< ", holeY: " << holeY << ", directionX: " << directionX << ", wall_length:"
//			<< wall_lenght << ", perpendicular: " << perpendicular << endl;
	//create the wall with an hole
	for(int i = 0; i < wall_lenght; i++){
		if(fromX != holeX || fromY != holeY)
		maze[width * fromY + fromX] = WALL;
		fromX += directionX;
		fromY += directionY;
	}
	//first recursive call
	newX = x;
	newY = y;
	if(orientation == HORIZONTAL){
		newWidth = width;
		newHeight = fromY - y + 1;
	}else{
		newWidth = fromX - x + 1;
		newHeight = height;
	}
	recursive_divide(maze,newX, newY, newWidth, newHeight, choose_orientation(newWidth, newHeight));

	//second recursive call
	if(orientation == HORIZONTAL){
		newX = x;
		newY = fromY + 1;
		newWidth = width;
		newHeight = y + height - fromY - 1;
	}else{
		newX = fromX + 1;
		newY = y;
		newWidth = x + width - fromX - 1;
		newHeight = height;
	}
	recursive_divide(maze,newX, newY, newWidth, newHeight, choose_orientation(newWidth, newHeight));
}

void CPU_division_maze_generator(int *maze, int width, int height){
	//set random seed
	srand(time(NULL));
	//first, set everything to OPEN
	FillOpen(maze, width* height);
	//start with the algorithm
	int start_x = rand() % width;
	int start_y = rand() % height;
	recursive_divide(maze, start_x, start_y, width, height, choose_orientation(width, height));
	//fill border
	fill_border(maze,width,height);
	//add objective
	add_objective(maze,width,height);
}

__device__ int device_choose_orientation (int width, int height){
	if(width < height)	return HORIZONTAL;
	else if(height < width) return VERTICAL;
	else return 1;	//i removed random. Hope it doesn't suck at all
}

class StackElement {

	public:
		int x;
		int y;				//coords
		int width;
		int height;			//size of the sub-array
		int orientation;	//orientation for the algorithm
		int random_value;	//a random value

		//constructor
		StackElement(int _x, int _y, int _width, int _height){
			x = _x;
			y = _y;
			width = _width;
			height = _height;
			orientation = choose_orientation(width, height);
			random_value = -1;	//no setup for device compatibility code
		}

		__device__ StackElement(int _x, int _y, int _width, int _height, int useless){
			x = _x;
			y = _y;
			width = _width;
			height = _height;
			orientation = device_choose_orientation(width, height);
			random_value = -1;	//no setup for device compatibility code
		}


		__device__ StackElement(){
			x = y = width = height = orientation = random_value = -1;
		}

		__device__ bool isEmpty(){
			return x == y == width == height == orientation == -1;
		}

		__device__ void copyAttributes(int &_x, int &_y, int &_width, int &_height, int &_orientation, int &_rand){
			_x = x;
			_y = y;
			_width = width;
			_height = height;
			_orientation = orientation;
			_rand = random_value;

		}


};

//function that actually execute the code of the algorithm on the gpu
__global__ void GPU_iterator_divide(int *maze, StackElement *stack, int size_stack, StackElement *new_stack, int *size_new_stack, int *recursive_calls){
	//define parameters for original algorithm
	int x, y, width, height, orientation, rand;
	//get parameters values
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//if it's a valid index
	if(index < size_stack){
		//check if I have data
		if(stack[index].isEmpty()) return;
		//get data from the stack
		stack[index].copyAttributes(x,y,width,height,orientation,rand);
		//start algorithm
		int fromX, fromY,holeX, holeY, directionX, directionY, wall_lenght;
		int newX, newY, newWidth, newHeight;
		//check if I don't have to make this iteration
		if(width < MAZE_RESOLUTION || height < MAZE_RESOLUTION){
			new_stack[2 * index] = StackElement();
			new_stack[2* index + 1] = StackElement();
			atomicAdd(size_new_stack, 2);
			return;
		}
		if(orientation == HORIZONTAL){
			//is horizontal
			fromX = x;
			if(height - 2 <= 0){
				fromY = y;
			}else{
				fromY = y + rand %(height-2);
			}
			holeX = fromX + rand %width;
			holeY = fromY;
			directionX = 1;
			directionY = 0;
			wall_lenght = width;

		}else{
			//is vertical
			if(width - 2 <= 0){
				fromX = x;
			}else{
				fromX = x + rand %(width-2);
			}
			fromY = y;
			holeX = fromX;
			holeY = fromY + rand %height;
			directionX = 0;
			directionY = 1;
			wall_lenght = height;
		}
		//create the wall with an hole
		for(int i = 0; i < wall_lenght; i++){
			if(fromX != holeX || fromY != holeY)
			maze[width * fromY + fromX] = WALL;
			fromX += directionX;
			fromY += directionY;
		}
		//first recursive call
		newX = x;
		newY = y;
		if(orientation == HORIZONTAL){
			newWidth = width;
			newHeight = fromY - y + 1;
		}else{
			newWidth = fromX - x + 1;
			newHeight = height;
		}
		//add call in stack
		new_stack[index * 2] = StackElement(newX, newY, newWidth, newHeight, 0);
		atomicAdd(size_new_stack, 1);

		//second recursive call
		if(orientation == HORIZONTAL){
			newX = x;
			newY = fromY + 1;
			newWidth = width;
			newHeight = y + height - fromY - 1;
		}else{
			newX = fromX + 1;
			newY = y;
			newWidth = x + width - fromX - 1;
			newHeight = height;
		}
		//add call in stack
		new_stack[2* index + 1] = StackElement(newX, newY, newWidth, newHeight, NULL);
		atomicAdd(size_new_stack,1);

		//increment counter of calls
		atomicAdd(recursive_calls, 2);
	}
}

//function that iterate the call of the algorithm in GPU version
void GPU_recursive_divide(int *maze,int width, int height){
	int size_stack = 0;
	int *dev_size_temp;
	int recursive_calls = 0;
	int *dev_recursive;
	int default_size = 0;
	//create stack
	int stack_dimension = width * height * 100;
	StackElement stack[stack_dimension];
	//stack for device
	StackElement *dev_stack, *dev_temp_stack;
	cudaMalloc(&dev_temp_stack, sizeof(StackElement) * stack_dimension);
	cudaMalloc(&dev_size_temp, sizeof(int));
	cudaMalloc(&dev_recursive, sizeof(int));

	bool again = true;
	//setup the first call
	stack[size_stack++] = StackElement(0, 0, width, height);
	while(again){	//till the end of the recursive calls
//		cout << "size_stack: " << size_stack << endl;
		//setup a random value for each stack call
		for(int i = 0; i < size_stack; i++){
			stack[i].random_value = rand();
		}
		int n_blocks = (size_stack / 32 ) + 1;
		//allocate and copy device stack
		cudaMalloc(&dev_stack, sizeof(StackElement) * size_stack);
		cudaMemcpy(dev_stack, stack, sizeof(StackElement) * size_stack, cudaMemcpyHostToDevice);
		//setup size of the new stack and recursive calls
		cudaMemcpy(dev_size_temp, &default_size, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_recursive, &default_size, sizeof(int), cudaMemcpyHostToDevice);
		//call the real algorithm
//		cout << "calling gpu ... ";
		GPU_iterator_divide<<<n_blocks, 32>>>(maze, dev_stack, size_stack, dev_temp_stack, dev_size_temp, dev_recursive);
//		cout << "end!" << endl;
		//wait that every thread finishes
		cudaDeviceSynchronize();
		//copy stack size and stack to host
		cudaMemcpy(&size_stack, dev_size_temp, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(stack, dev_temp_stack, sizeof(StackElement) * size_stack, cudaMemcpyDeviceToHost);
		cudaMemcpy(&recursive_calls, dev_recursive, sizeof(int), cudaMemcpyDeviceToHost);
		//free dev_stack
		cudaFree(dev_stack);
		//check if I have finished
		if(recursive_calls == 0) again = false;
	}
}


void GPU_division_maze_generator(int *maze, int width, int height){
	//define maze on GPU
	int *dev_maze;
	cudaMalloc(&dev_maze, sizeof(int) * width * height);
	cudaMemcpy(dev_maze, maze, sizeof(int) * width * height, cudaMemcpyHostToDevice);
	//set random seed
	srand(time(NULL));
	//first, set everything to OPEN
	GPU_FillOpen<<<height, width>>>(dev_maze, width, width* height);
	//start with the algorithm
	GPU_recursive_divide(dev_maze, width, height);
	//maze generated. copy on host
	cudaMemcpy(maze, dev_maze, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
}

