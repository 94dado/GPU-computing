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

#define SUD 1
#define EST 2

#define MAZE_RESOLUTION 2


int choose_orientation(int width, int height){
	if(width < height)	return HORIZONTAL;
	else if(height < width) return VERTICAL;
	else return rand()%2 + 1;
}

void convert_maze(int *maze, int *newMaze, int width, int height){
	int newWidth = width * 2 + 1;
	bool bottom, south, south2, east;
	for(int i = 0; i < newWidth; i++){
		newMaze[i] = WALL;
	}
	for(int i = 1; i <= height; i++){
		newMaze[i*newWidth] = WALL;
		for(int j = 1; j <= width;j++){
			bottom = (i+1 >= height);
			south = (maze[i*width+j] & SUD) != 0 || bottom;
			south2 = j+1 < width && ((maze[width*i + j + 1] & SUD) != 0 || bottom);
			east = (maze[i*width+j] & EST) != 0 || j+1 > width;

			if(south){
				newMaze[i*newWidth+(j*2) - 1] = WALL;
			}else{
				newMaze[i*newWidth+(j*2) - 1] = OPEN;
			}
			if(east){
				newMaze[i*newWidth+(j*2)] = WALL;
			}else{
				if(south && south2){
					newMaze[i*newWidth+(j*2)] = WALL;
				}else{
					newMaze[i*newWidth+(j*2)] = OPEN;
				}
			}
		}
	}
}

void reduce_maze(int *maze, int *newMaze, int newWidth, int height){
	newWidth += 1;
	int width = (newWidth-2)*2+1;	//width di maze(labirinto grande)
	for(int i = 0; i < height; i++){
		newMaze[i*newWidth] = WALL;
		for(int j = 1; j < width; j++){
			if(i*newWidth + j < newWidth * height){
				int val = maze[i*width+(j*2)];
				int next_val = maze[i*width+(j*2) + 1];
				if(i == 0 || j == 0 || i == height-1 || j == newWidth -1){
					newMaze[i*newWidth + j] = WALL;
				}
				else{
					if(val == WALL && next_val == WALL){
						newMaze[i*newWidth + j] = WALL;
					}
					else{
						newMaze[i*newWidth + j] = OPEN;
					}
				}
			}
		}
	}
}

void add_objective(int *maze, int width, int height){
	vector <int> stack_start, stack_end;
	for(int j = 1; j < width - 1; j++){
		if(maze[j+width] == OPEN){
			stack_start.push_back(j);
		}
		if(maze[width*(height-2) + j] == OPEN){
			stack_end.push_back(width*(height-1)+j);
		}
	}
//	for(int i = 1; i < height; i++){
//		if(maze[i*width + 1] == OPEN){
//			stack_start.push_back(i*width);
//		}
//		if(maze[i*width + width - 2] == OPEN){
//			stack_end.push_back(i*width + width -1);
//		}
//	}
	int start = rand() % stack_start.size();
	start = stack_start[start];
	maze[start] = OBJECTIVE;
	int end = rand() % stack_end.size();
	end = stack_end[end];
	maze[end] = OBJECTIVE;
}

void recursive_divide(int *maze, int x, int y, int width, int height, int real_width, int orientation){
	int wX, wY,pX,pY,dX,dY,lenght,dir;
	int nX, nY, w, h;
	//check if I don't have to make this iteration
	if(width < MAZE_RESOLUTION || height < MAZE_RESOLUTION){
		return;
	}
	bool horizontal = (orientation == HORIZONTAL);
//	cout << "or: " << horizontal << endl;
	if(horizontal){
		//is horizontal
		wX = x;
		wY = y + (height > 2? rand()%(height-2) : 0);
		pX = wX + rand()%width;
		pY = wY;
		lenght = width;
		dir = SUD;

	}else{
		//is vertical
		wX = x + (width > 2? rand()%(width-2) : 0);
		wY = y;
		pX = wX;
		pY = wY + rand()%height;
		lenght = height;
		dir = EST;
	}
	dX = horizontal;
	dY = !horizontal;
//	cout << "data calculated:" << endl;
//	cout << "fromX: " << fromX << ", fromY: " << fromY << ", holeX: " << holeX
//			<< ", holeY: " << holeY << ", directionX: " << directionX << ", wall_length:"
//			<< wall_lenght << ", perpendicular: " << perpendicular << endl;
	//create the wall with an hole

	for(int i = 0; i < lenght; i++){
		if(wX != pX || wY != pY){
			 maze[real_width * wY + wX] = maze[real_width * wY + wX] | dir;
//			cout << wY << "," << wX << "," << real_width*wY+wX << endl;
		}
		wX += dX;
		wY += dY;
	}

	//first recursive call
	nX = x;
	nY = y;
	if(horizontal){
		w = width;
		h = wY - y + 1;
	}else{
		w = wX - x + 1;
		h = height;
	}
	recursive_divide(maze,nX, nY, w, h, real_width, choose_orientation(w, h));

	//second recursive call
	if(horizontal){
		nX = x;
		nY = wY + 1;
		w = width;
		h = y + height - wY - 1;
	}else{
		nX = wX + 1;
		nY = y;
		w = x + width - wX - 1;
		h = height;
	}
//	cout << nX << "," << nY << endl;
	recursive_divide(maze,nX, nY, w, h, real_width, choose_orientation(w, h));
}

void CPU_division_maze_generator(int *maze, int width, int height){
	//to set everything all right
	width --;
	//set random seed
	srand(time(NULL));
	int *first_maze = new int[width * height];
	int *second_maze = new int[(width * 2 + 1) * (height+1)];
	//first, set everything to OPEN
	FillWall(first_maze, width * height);
	//start with the algorithm
	recursive_divide(first_maze, 0, 0, width, height, width, choose_orientation(width, height));
//	PrintMaze(first_maze,width,height);
	convert_maze(first_maze,second_maze,width,height);
	reduce_maze(second_maze,maze,width,height);
	add_objective(maze,width+1,height);

	delete first_maze;
	delete second_maze;
}

__device__ int device_choose_orientation (int width, int height){
	if(width < height)	return HORIZONTAL;
	else if(height < width) return VERTICAL;
	else return HORIZONTAL;	//i removed random. Hope it doesn't suck at all
}

class StackElement {

	public:
		int x;
		int y;				//coords
		int width;
		int height;			//size of the sub-array
		int orientation;	//orientation for the algorithm
		int random_value;	//a random value

		//constructors

		StackElement(){
			x = y = width = height = orientation = random_value = -1;
		}


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


		__device__ StackElement(int useless){
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
__global__ void GPU_iterator_divide(int *maze, int real_width, StackElement *stack, int size_stack, StackElement *new_stack, int *size_new_stack, int *recursive_calls){
	//define parameters for original algorithm
	int x, y, width, height, orientation, rand;
	int wX, wY,pX,pY,dX,dY,lenght,dir;
	int nX, nY, w, h;
	//get parameters values
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//if it's a valid index
	if(index < size_stack){
		//check if I have data
		if(stack[index].isEmpty()) return;
		//get data from the stack
		stack[index].copyAttributes(x,y,width,height,orientation,rand);
		//start algorithm
		//check if I don't have to make this iteration
		if(width < MAZE_RESOLUTION || height < MAZE_RESOLUTION){
			new_stack[2 * index] = StackElement(NULL);
			new_stack[2* index + 1] = StackElement(NULL);
			atomicAdd(size_new_stack, 2);
			return;
		}
		bool horizontal = (orientation == HORIZONTAL);
	//	cout << "or: " << horizontal << endl;
		if(horizontal){
			//is horizontal
			wX = x;
			wY = y + (height > 2? rand %(height-2) : 0);
			pX = wX + rand % width;
			pY = wY;
			lenght = width;
			dir = SUD;

		}else{
			//is vertical
			wX = x + (width > 2? rand %(width-2) : 0);
			wY = y;
			pX = wX;
			pY = wY + rand % height;
			lenght = height;
			dir = EST;
		}
		dX = horizontal;
		dY = !horizontal;
	//	cout << "data calculated:" << endl;
	//	cout << "fromX: " << fromX << ", fromY: " << fromY << ", holeX: " << holeX
	//			<< ", holeY: " << holeY << ", directionX: " << directionX << ", wall_length:"
	//			<< wall_lenght << ", perpendicular: " << perpendicular << endl;
		//create the wall with an hole

		for(int i = 0; i < lenght; i++){
			if(wX != pX || wY != pY){
				 maze[real_width * wY + wX] = maze[real_width * wY + wX] | dir;
	//			cout << wY << "," << wX << "," << real_width*wY+wX << endl;
			}
			wX += dX;
			wY += dY;
		}

		//first recursive call
		nX = x;
		nY = y;
		if(horizontal){
			w = width;
			h = wY - y + 1;
		}else{
			w = wX - x + 1;
			h = height;
		}
		new_stack[index * 2] = StackElement(nX, nY, w, h, NULL);

		//second recursive call
		if(horizontal){
			nX = x;
			nY = wY + 1;
			w = width;
			h = y + height - wY - 1;
		}else{
			nX = wX + 1;
			nY = y;
			w = x + width - wX - 1;
			h = height;
		}
	//	cout << nX << "," << nY << endl;
		new_stack[index * 2 + 1] = StackElement(nX, nY, w, h, NULL);
		//increment counter of calls
		atomicAdd(recursive_calls, 2);
		atomicAdd(size_new_stack,2);
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
	StackElement *stack = new StackElement;
	//stack for device
	StackElement *dev_stack, *dev_temp_stack;
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
		//allocate and copy device stack
		cudaMalloc(&dev_stack, sizeof(StackElement) * size_stack);
		cudaMemcpy(dev_stack, stack, sizeof(StackElement) * size_stack, cudaMemcpyHostToDevice);
		cudaMalloc(&dev_temp_stack, sizeof(StackElement) * size_stack * 2);
		//setup size of the new stack and recursive calls
		cudaMemcpy(dev_size_temp, &default_size, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_recursive, &default_size, sizeof(int), cudaMemcpyHostToDevice);
		//call the real algorithm
//		cout << "calling gpu ... ";
		GPU_iterator_divide<<<1, size_stack>>>(maze, width, dev_stack, size_stack, dev_temp_stack, dev_size_temp, dev_recursive);
//		cout << "end!" << endl;
		//wait that every thread finishes
		cudaDeviceSynchronize();
		//check if I have finished
		cudaMemcpy(&recursive_calls, dev_recursive, sizeof(int), cudaMemcpyDeviceToHost);
		if(recursive_calls == 0){
			again = false;
		}else{
			//copy stack size and stack to host
			cudaMemcpy(&size_stack, dev_size_temp, sizeof(int), cudaMemcpyDeviceToHost);
			delete stack;
			stack = new StackElement[size_stack];
			cudaMemcpy(stack, dev_temp_stack, sizeof(StackElement) * size_stack, cudaMemcpyDeviceToHost);
			//free dev_stack
			cudaFree(dev_stack);
			cudaFree(dev_temp_stack);
		}
	}
	free(stack);
}

__global__ void GPU_convert_maze(int *maze, int *newMaze, int width, int height, int newWidth, int newHeight, int sud, int est){
	printf("dioporco\n");
	int i = blockIdx.x;
	int j = threadIdx.x;
	bool bottom, south, south2, east;bottom = (i+1 >= height);
	south = (maze[i*width+j] & sud) != 0 || bottom;
	south2 = j+1 < width && ((maze[width*i + j + 1] & sud) != 0 || bottom);
	east = (maze[i*width+j] & est) != 0 || j+1 > width;
	__syncthreads();
	if(i == 0 || j == 0){
		newMaze[i*newWidth + j] = WALL;
		return;
	}
//	printf("bottom:%d,south:%d,south2:%d,east:%d\n",bottom,south,south2,east);
	if(south){
		newMaze[i*newWidth+(j*2) - 1] = WALL;
	}else{
		newMaze[i*newWidth+(j*2) - 1] = OPEN;
	}
	if(east){
		newMaze[i*newWidth+(j*2)] = WALL;
	}else{
		if(south && south2){
			newMaze[i*newWidth+(j*2)] = WALL;
		}else{
			newMaze[i*newWidth+(j*2)] = OPEN;
		}
	}
}

__global__ void GPU_reduce_maze(int *maze, int *newMaze, int newWidth, int width, int height){
	int i = blockIdx.x;
	int j = threadIdx.x;
	if(j==0) newMaze[i*newWidth + j] = WALL;
	else if(i*newWidth + j < newWidth * height){
		int val = maze[i*width+(j*2)];
		int next_val = maze[i*width+(j*2) + 1];
		if(i == 0 || j == 0 || i == height-1 || j == newWidth -1){
			newMaze[i*newWidth + j] = WALL;
		}
		else{
			if(val == WALL && next_val == WALL){
				newMaze[i*newWidth + j] = WALL;
			}
			else{
				newMaze[i*newWidth + j] = OPEN;
			}
		}
	}
}

void GPU_division_maze_generator(int *maze, int width, int height){
	//to set everything all right
	width --;
	//set random seed
	srand(time(NULL));
	//define maze on GPU
	int *dev_maze;
	int *dev_first_maze, *dev_second_maze;
	cudaMalloc(&dev_maze, sizeof(int) * (width+1) * height);
	cudaMalloc(&dev_first_maze, sizeof(int) * width * height);
	cudaMalloc(&dev_second_maze, sizeof(int) * (width * 2 + 1) * (height + 1));
	//first, set everything to WALL
	GPU_FillWall<<<height, width>>>(dev_first_maze, width, width * height);
	cudaDeviceSynchronize();
	//start with the algorithm
	GPU_recursive_divide(dev_first_maze, width, height);
//	int *temp = new int[(width * height)];
//	cudaMemcpy(temp,dev_second_maze, sizeof(int) * (width * height), cudaMemcpyDeviceToHost);
//	PrintMaze(temp,width,height);
	GPU_convert_maze<<<height + 1,width * 2 + 1>>>(dev_first_maze, dev_second_maze, width, height, width * 2 + 1, height + 1, SUD, EST);
	cudaDeviceSynchronize();
//	int *temp = new int[(width * 2 + 1) * (height + 1)];
//	cudaMemcpy(temp,dev_second_maze, sizeof(int) * (width * 2 + 1) * (height + 1), cudaMemcpyDeviceToHost);
//	PrintMaze(temp,width * 2 + 1,height + 1);
	GPU_reduce_maze<<<height,width + 1>>>(dev_second_maze, dev_maze, width + 1, (width - 1)*2+1, height);
	cudaDeviceSynchronize();
	cudaMemcpy(maze, dev_maze, sizeof(int) * (width+1) * height, cudaMemcpyDeviceToHost);
	add_objective(maze,width+1,height);
}

