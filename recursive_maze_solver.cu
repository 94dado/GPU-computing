#include <iostream>
#include <unistd.h>
#include "./Header/common.h"

using namespace std;

// Symbols:
// '.' = open
// '#' = blocked
// 'S' = start
// 'G' = goal
// '+' = path
// 'x' = bad path

// start and end of the maze
int pathStart[2];
int pathEnd[2];

void PrintCharMaze(char *array, int width, int height){
	int i,j;
	for(i = 0; i < height; i++){
		for(j = 0; j < width; j++){
			cout << array[i*width + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

void FromCoordToChar(int *mazeInt, char *maze, int width, int height) {
    bool findStart = false;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            // start
            if (mazeInt[i * width + j] == OBJECTIVE && !findStart) {
                pathStart[0] = i;
                pathStart[1] = j;
                maze[i * width + j] = 'S';
                findStart = true;
            }
            // end
            else if (mazeInt[i * width + j] == OBJECTIVE && findStart) {
                pathEnd[0] = i;
                pathEnd[1] = j;
                maze[i * width + j] = 'G';
            }
            // wall
            else if (mazeInt[i * width + j] == WALL) {
                maze[i * width + j] = '#';
            }
            // open
            else {
                maze[i * width + j] = '.';
            }
        }
    }
}

void FromCharToCoord(int *mazeInt, char *maze, int width, int height) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            // start and end
            if ((pathStart[0] == i && pathStart[1] == j) || maze[i * width + j] == 'G') {
                mazeInt[i * width + j] = OBJECTIVE;
            }
            // open
            else if (maze[i * width + j] == '+') {
                mazeInt[i * width + j] = OPEN;
            }
            // wall
            else {
                mazeInt[i * width + j] = WALL;
            }
        }
    }
}

int find_path(char *maze, int x, int y, int width, int height) {
    // If x,y is outside maze, return false.
    if ( x < 0 || x > width - 1 || y < 0 || y > height - 1 ) return false;

    // If x,y is the goal, return true.
    if ( maze[x * width + y] == 'G' ) return true;

    // If x,y is not open, return false.
    if ( maze[x * width + y] != '.' && maze[x * width + y] != 'S' ) return false;

    // Mark x,y part of solution path.
    maze[x * width + y] = '+';

    // If find_path North of x,y is true, return true.
    if ( find_path(maze, x, y - 1, width, height) == true ) return true;

    // If find_path East of x,y is true, return true.
    if ( find_path(maze, x + 1, y, width, height) == true ) return true;

    // If find_path South of x,y is true, return true.
    if ( find_path(maze, x, y + 1, width, height) == true ) return true;

    // If find_path West of x,y is true, return true.
    if ( find_path(maze, x - 1, y, width, height) == true ) return true;

    // Unmark x,y as part of solution path.
    maze[x * width + y] = 'x';

    return false;
}

void CPU_recursive_maze_solver(int *mazeInt, int width, int height) {
    char *mazeChar = new char[width * height];

    FromCoordToChar(mazeInt, mazeChar, width, height);

    find_path(mazeChar, pathStart[0], pathStart[1], width, height);

    FromCharToCoord(mazeInt, mazeChar, width, height);

    delete mazeChar;
}

__global__ void GPU_FromCoordToChar(int *mazeInt, char *maze, int *indexes, int offset) {
    int index = blockDim.x * blockIdx.x + offset + threadIdx.x;
	// start and end
	if (mazeInt[index] == OBJECTIVE) {
		indexes[0] = blockIdx.x;;
		indexes[1] = threadIdx.x;
		maze[index] = 'G';
	}
	// wall
	else if (mazeInt[index] == WALL) {
		maze[index] = '#';
	}
	// open
	else {
		maze[index] = '.';
	}
}

__global__ void GPU_FromCharToCoord(int *mazeInt, char *maze, int *pathStart, int offset) {
	int index = blockDim.x * blockIdx.x + offset + threadIdx.x;
	// start and end
	if ((pathStart[0] == blockIdx.x && pathStart[1] == threadIdx.x) || maze[index] == 'G') {
		mazeInt[index] = OBJECTIVE;
	}
	// open
	else if (maze[index] == '+') {
		mazeInt[index] = OPEN;
	}
	// wall
	else {
		mazeInt[index] = WALL;
	}
}

__global__ void setupArray(int *array, int offset){
	int index = blockDim.x * blockIdx.x + offset + threadIdx.x;
	array[index] = -1;
}

__global__ void GPU_find_path(char *maze, int *result, int height, int offset) {
	int index = blockDim.x * blockIdx.x + offset + threadIdx.x;

    // If x,y is the goal, return true.
    if ( maze[index] == 'G') {
    	result[index] = 1;
    	return;
    }

    // If x,y is not open, return false.
    if ( maze[index] != '.' && maze[index] != 'S' && maze[index] != '+'){
    	result[index] = 0;
    	return;
    }
	// Mark x,y part of solution path.
	maze[index] = '+';
	result[index] = 1;
}

__global__ void setupStart(char *maze, int *pathStart, int width){
	maze[pathStart[0] * width + pathStart[1]] = 'S';
}

__global__ void GPU_clean_no_way(char *maze, int *result, bool *finish, int height, int offset){
	int index = blockDim.x * blockIdx.x + offset + threadIdx.x;
	int x = blockIdx.x;
	int y = offset + threadIdx.x;
	int width = blockDim.x;
	int count = 0;
//	printf("index: %d, width: %d, x: %d, y: %d\n",index,width,x,y);
	if(result[index] == 1){
		if (!( x < 0 || x > width - 1 || y-1 < 0 || y-1 > height - 1) ){
			if(result[x*width + y-1] == 0)
				count ++;
		}
		if (!( x + 1 < 0 || x + 1 > width - 1 || y < 0 || y > height - 1 )){
			if(result[(x+1)*width + y] == 0)
				count ++;
		}
		if (!( x < 0 || x > width - 1 || y + 1 < 0 || y + 1 > height - 1 )){
			if(result[x*width + y+1] == 0)
				count ++;
		}
		if (!( x - 1 < 0 || x - 1 > width - 1 || y < 0 || y > height - 1 )){
			if(result[(x-1)*width + y] == 0)
				count ++;
		}
	//	printf("count: %d\n",count);
		if(count == 3){
			result[index] = 0;
			maze[index] = 'x';
			*finish = false;
		}
	}
}
void GPU_recursive_maze_solver(int *mazeInt, int width, int height){
	char *dev_mazeChar;
	int *dev_mazeInt;
	int *dev_pathStart;

	//three-state: 1 true, 0 false, -1 not yet done
	int *dev_result;
	cudaMalloc(&dev_result, sizeof(int) * width * height);
	int max_rec = width / MAX_THREAD;
	int offset = 0;
	for(int i = 0; i < max_rec; i++){
		setupArray<<<height,MAX_THREAD>>>(dev_result, offset);
		offset = (i + 1) * MAX_THREAD;
	}
	setupArray<<<height,width % MAX_THREAD>>>(dev_result, offset);

	cudaMalloc(&dev_mazeChar, sizeof(char) * width * height);
	cudaMalloc(&dev_mazeInt, sizeof(int) * width * height);
	cudaMalloc(&dev_pathStart,sizeof(int) * 2);

	cudaMemcpy(dev_mazeInt,mazeInt,sizeof(int) * width * height, cudaMemcpyHostToDevice);
	offset = 0;
	for(int i = 0; i < max_rec; i++){
		GPU_FromCoordToChar<<<height,MAX_THREAD>>>(dev_mazeInt, dev_mazeChar, dev_pathStart, offset);
		offset = (i + 1) * MAX_THREAD;
	}
	GPU_FromCoordToChar<<<height,width % MAX_THREAD>>>(dev_mazeInt, dev_mazeChar, dev_pathStart, offset);
	cudaDeviceSynchronize();
	setupStart<<<1,1>>>(dev_mazeChar, dev_pathStart, width);
	cudaDeviceSynchronize();

	offset = 0;
	for(int i = 0; i < max_rec; i++){
		GPU_find_path<<<height,MAX_THREAD>>>(dev_mazeChar,dev_result, height, offset);
		offset = (i + 1) * MAX_THREAD;
	}
	GPU_find_path<<<height,width % MAX_THREAD>>>(dev_mazeChar,dev_result, height, offset);
	cudaDeviceSynchronize();

	bool finish = false;
	bool *dev_finished;
	cudaMalloc(&dev_finished, sizeof(bool));
	while(!finish){
		finish = true;
		cudaMemcpy(dev_finished,&finish,sizeof(bool),cudaMemcpyHostToDevice);
		offset = 0;
		for(int i = 0; i < max_rec; i++){
			GPU_clean_no_way<<<height,MAX_THREAD>>>(dev_mazeChar,dev_result, dev_finished, height, offset);
			offset = (i+1) * MAX_THREAD;
		}
		GPU_clean_no_way<<<height,width % MAX_THREAD>>>(dev_mazeChar,dev_result, dev_finished, height, offset);
		cudaDeviceSynchronize();

		cudaMemcpy(&finish,dev_finished,sizeof(bool),cudaMemcpyDeviceToHost);
	}

	offset = 0;
	for(int i = 0; i < max_rec; i++){
		GPU_FromCharToCoord<<<height, MAX_THREAD>>>(dev_mazeInt, dev_mazeChar, dev_pathStart, offset);
		offset = (i+1) * MAX_THREAD;
	}
	GPU_FromCharToCoord<<<height,width % MAX_THREAD>>>(dev_mazeInt, dev_mazeChar, dev_pathStart, offset);
	cudaMemcpy(mazeInt,dev_mazeInt,sizeof(int) * width * height, cudaMemcpyDeviceToHost);
}
