#include <iostream>
#include <queue>
#include <vector>
#include "Header/common.h"

using namespace std;

// the start and the end of the path
int startPath[2];
int endPath[2];

// queue NodeStruct used in BFS
struct NodeStruct {
    // (x, y) represents matrix cell cordinates
    // dist represent its minimum distance from the source
    int x, y;
    NodeStruct *parent;
    bool isNotWall;
};

// Below arrays details all 4 possible movements from a cell
int row[] = { -1, 0, 0, 1 };
int col[] = { 0, -1, 1, 0 };

// Function to check if it is possible to go to position (row, col)
// from current position. The function returns false if the cell
// not a valid position or has value 0 or it is already visited
bool isValid(NodeStruct *mat, bool *visited, int row, int col, int width, int height) {
    return (row >= 0) && (row < width) && (col >= 0) && (col < height) && mat[row * width + col].isNotWall && !visited[row * width + col];
}

// print the path
void PrintNodeMaze(NodeStruct *array, int *mat, int width, int height){
    int i,j;
    for(i = 0; i < height; i++){
        for(j = 0; j < width; j++){
            if ((array[i*width + j].x == startPath[0] && array[i*width + j].y == startPath[1]) || (array[i*width + j].x == endPath[0] && array[i*width + j].y == endPath[1])) {
//                cout << 2 << " ";
                mat[i*width + j] = 2;
            }
            else {
//                cout << array[i*width + j].isNotWall << " ";
                mat[i*width + j] = array[i*width+j].isNotWall;
            }
        }
//        cout << endl;
    }
//    cout << endl;
}

// Search the shortest path by parent NodeStruct
void ReachPath(NodeStruct *matrix, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        matrix[i].isNotWall = false;
    }
    // the last is the end
    NodeStruct *currentNodeStruct = &(matrix[endPath[0] * width + endPath[1]]);
    // the first is the start
    NodeStruct *startNodeStruct = &(matrix[startPath[0] * width + startPath[1]]);
    bool isReach = false;
    while (!isReach) {
        matrix[currentNodeStruct->x * width + currentNodeStruct->y].isNotWall = true;
        currentNodeStruct = currentNodeStruct->parent;
        if (currentNodeStruct->x == startNodeStruct->x && currentNodeStruct->y == startNodeStruct->y) {
            isReach = true;
        }
    }
}

// it find start and end for the parameter of the BFS algorithm
void FromCoordToNodeStruct(NodeStruct *matrix, int *mat, int width, int height) {
    bool findStart = false;
    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            if (mat[i*width + j] == 1) {
                // is not a wall
                matrix[i * width + j] = {i, j, NULL, true};
            }
            // is the start
            else if (mat[i*width + j] == 2 && !findStart) {
                matrix[i * width + j] = {i, j, NULL, true};
                startPath[0] = i;
                startPath[1] = j;
                findStart = true;
            }
            // is the end
            else if (mat[i*width + j] == 2 && findStart) {
                matrix[i * width + j] = {i, j, NULL, true};
                endPath[0] = i;
                endPath[1] = j;
            }
            else {
                matrix[i * width + j] = {i, j, NULL, false};
            }
        }
    }
}

// Find Shortest Possible Route in a matrix mat from source
// cell (i, j) to destination cell (x, y)
void CPU_bfs_maze_solver(int *mat, int width, int height) {
    // pass to NodeStruct coordinates
    NodeStruct *matrix = new NodeStruct[width * height];
    FromCoordToNodeStruct(matrix, mat, width, height);
    // construct a matrix to keep track of visited cells
    bool *visited = new bool[width * height];

    // initially all cells are unvisited
    for (int k = 0; k < width * height; k++) {
        visited[k] = false;
    }

    // create an empty queue
    queue<NodeStruct> q;

    // mark source cell as visited and enqueue the source NodeStruct
    visited[startPath[0] * width + startPath[1]] = true;
    q.push(matrix[startPath[0] * width + startPath[1]]);

    // run till queue is not empty
    while (!q.empty()) {
        // pop front NodeStruct from queue and process it
        NodeStruct NodeStruct = q.front();
        q.pop();

        // (i, j) represents current cell and dist stores its
        // minimum distance from the source
        int i = NodeStruct.x, j = NodeStruct.y;

        // if destination is found, update min_dist and stop
        if (i == endPath[0] && j == endPath[1]) {
            ReachPath(matrix, width, height);
            PrintNodeMaze(matrix, mat, width, height);
            break;
        }

        // check for all 4 possible movements from current cell
        // and enqueue each valid movement into the queue
        for (int k = 0; k < 4; k++) {
            // check if it is possible to go to position
            // (i + row[k], j + col[k]) from current position
            if (isValid(matrix, visited, i + row[k], j + col[k], width, height)) {
                // mark next cell as visited and enqueue it
                visited[(i + row[k]) * width + (j + col[k])] = true;
                q.push({ i + row[k], j + col[k], &matrix[i * width + j], matrix[i * width + j].isNotWall });
                matrix[(i + row[k]) * width + (j + col[k])].parent = &matrix[i * width + j];
            }
        }
    }
    delete matrix;
    delete visited;
}

__global__ void GPU_FromCoordToNodeStruct(NodeStruct *matrix, int *mat, int width, int height, int *path, int *index, int offset){
	int i = blockIdx.x;
	int j = offset + threadIdx.x;
	if (mat[i*width + j] == 1) {
		// is not a wall
		matrix[i * width + j] = {i, j, NULL, true};
	}
	// is the start
	else if (mat[i*width + j] == 2) {
		matrix[i * width + j] = {i, j, NULL, true};
		path[*index] = i;
		path[*index + 1] = j;
		atomicAdd(index,2);
	}
	else {
		matrix[i * width + j] = {i, j, NULL, false};
	}
}

__global__ void setupVisited(bool *visited, int width, int height, int offset){
	int k = blockIdx.x * width + offset + threadIdx.x;
	visited[k] = false;
}

__global__ void SetWallNode(NodeStruct *matrix, int width, int dimension, bool value){
	int i = blockIdx.x * width +threadIdx.x;
	if(i < dimension) matrix[i].isNotWall = value;
}

__global__ void GPU_PrintNodeMaze(NodeStruct *array, int *mat, int *startPath, int *endPath, int width, int height, int offset){
    int i = blockIdx.x;
    int j = offset + threadIdx.x;
	if ((array[i*width + j].x == startPath[0] && array[i*width + j].y == startPath[1]) || (array[i*width + j].x == endPath[0] && array[i*width + j].y == endPath[1])) {
//                cout << 2 << " ";
		mat[i*width + j] = 2;
	}
	else {
//                cout << array[i*width + j].isNotWall << " ";
		mat[i*width + j] = array[i*width+j].isNotWall;
	}
}

// Search the shortest path by parent NodeStruct
void GPU_ReachPath(NodeStruct *matrix, NodeStruct *dev_matrix, int width, int height) {
	cudaMemcpy(dev_matrix, matrix, sizeof(NodeStruct) * width * height, cudaMemcpyHostToDevice);
    SetWallNode<<<height, width>>>(dev_matrix, width, width * height, false);
    cudaDeviceSynchronize();
    cudaMemcpy(matrix, dev_matrix, sizeof(NodeStruct) * width * height, cudaMemcpyDeviceToHost);
//    PrintNodeMaze(matrix,width,height);
    // the last is the end
    NodeStruct *currentNodeStruct = &(matrix[endPath[0] * width + endPath[1]]);
    // the first is the start
    NodeStruct *startNodeStruct = &(matrix[startPath[0] * width + startPath[1]]);
    bool isReach = false;
    while (!isReach) {
        matrix[currentNodeStruct->x * width + currentNodeStruct->y].isNotWall = true;
//        cout << "new current: x=" << currentNodeStruct->x << ",y=" << currentNodeStruct->y << ",parent=" << currentNodeStruct->parent << ",wall=" << currentNodeStruct->isNotWall << endl;
        currentNodeStruct = currentNodeStruct->parent;
        if (currentNodeStruct->x == startNodeStruct->x && currentNodeStruct->y == startNodeStruct->y) {
            isReach = true;
        }
    }
}

void GPU_bfs_maze_solver(int *mat, int width, int height){
	// pass to NodeStruct coordinates
	NodeStruct *matrix = new NodeStruct[width * height];
	int * dev_mat;
	cudaMalloc(&dev_mat, sizeof(int) * width * height);
	//cuda variable
	NodeStruct *dev_matrix;
	int *dev_path, *dev_index, index = 0;
	//memory allocation
	cudaMalloc(&dev_matrix, sizeof(NodeStruct) * width * height);
	cudaMalloc(&dev_path, sizeof(int) * 4);
	cudaMalloc(&dev_index, sizeof(int));
	//copy data on GPU
	cudaMemcpy(dev_mat, mat, sizeof(int) * width * height, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_index,&index, sizeof(int), cudaMemcpyHostToDevice);
	int max_rec = width / MAX_THREAD;
	int offset = 0;
	for(int i = 0; i < max_rec; i++){
		GPU_FromCoordToNodeStruct<<<height, width>>>(dev_matrix, dev_mat, width, height, dev_path, dev_index, offset);
		offset = (i+1) * MAX_THREAD;
	}
	GPU_FromCoordToNodeStruct<<<height, width % MAX_THREAD>>>(dev_matrix, dev_mat, width, height, dev_path, dev_index, offset);
	cudaDeviceSynchronize();
	//get back all the data
	cudaMemcpy(matrix, dev_matrix, sizeof(NodeStruct) * width * height, cudaMemcpyDeviceToHost);
	cudaMemcpy(startPath, dev_path, sizeof(int) * 2, cudaMemcpyDeviceToHost);
	cudaMemcpy(endPath, dev_path + 2, sizeof(int) * 2, cudaMemcpyDeviceToHost);
//	    PrintNodeMaze(matrix, width, height);
//	    printf("start: %d,%d  end: %d,%d",startPath[0],startPath[1],endPath[0],endPath[1]);
	// construct a matrix to keep track of visited cells
	bool *visited = new bool[width * height];

	bool *dev_visited;
	cudaMalloc(&dev_visited, sizeof(bool) * width * height);
	// initially all cells are unvisited
	offset = 0;
	for(int i = 0; i < max_rec; i++){
		setupVisited<<<height, MAX_THREAD>>>(dev_visited, width, height, offset);
		offset = (i+1) * MAX_THREAD;
	}
   setupVisited<<<height, width % MAX_THREAD>>>(dev_visited, width, height, offset);
   cudaDeviceSynchronize();
   cudaMemcpy(visited, dev_visited, sizeof(bool) * width * height, cudaMemcpyDeviceToHost);
	// create an empty queue
	queue<NodeStruct> q;

	// mark source cell as visited and enqueue the source NodeStruct
	visited[startPath[0] * width + startPath[1]] = true;
	q.push(matrix[startPath[0] * width + startPath[1]]);

	// run till queue is not empty
	while (!q.empty()) {
		// pop front NodeStruct from queue and process it
		NodeStruct NodeStruct = q.front();
		q.pop();

		// (i, j) represents current cell and dist stores its
		// minimum distance from the source
		int i = NodeStruct.x, j = NodeStruct.y;

		// if destination is found, update min_dist and stop
//	        cout << "i: " << i <<", j: " << j << endl;
//	        cout << "endPath: " << endPath[0] << "," << endPath[1] << endl;
		if (i == endPath[0] && j == endPath[1]) {
			GPU_ReachPath(matrix, dev_matrix, width, height);
			int *dev_start,*dev_end;
			cudaMalloc(&dev_start,sizeof(int)*2);
			cudaMalloc(&dev_end,sizeof(int)*2);
			cudaMemcpy(dev_start,startPath,sizeof(int)*2,cudaMemcpyHostToDevice);
			cudaMemcpy(dev_end,endPath,sizeof(int)*2,cudaMemcpyHostToDevice);
			cudaMemcpy(dev_matrix,matrix,sizeof(NodeStruct)*width*height, cudaMemcpyHostToDevice);
			offset = 0;
			for(int i = 0; i < max_rec; i++){
				GPU_PrintNodeMaze<<<height,MAX_THREAD>>>(dev_matrix, dev_mat, dev_start, dev_end, width, height, offset);
				offset = (i+1) * MAX_THREAD;
			}
			GPU_PrintNodeMaze<<<height,width % MAX_THREAD>>>(dev_matrix, dev_mat, dev_start, dev_end, width, height, offset);
			cudaMemcpy(mat,dev_mat,sizeof(int)*width*height,cudaMemcpyDeviceToHost);
			break;
		}

		// check for all 4 possible movements from current cell
		// and enqueue each valid movement into the queue
		for (int k = 0; k < 4; k++) {
			// check if it is possible to go to position
			// (i + row[k], j + col[k]) from current position
			if (isValid(matrix, visited, i + row[k], j + col[k], width, height)) {
				// mark next cell as visited and enqueue it
				visited[(i + row[k]) * width + (j + col[k])] = true;
				q.push({ i + row[k], j + col[k], &matrix[i * width + j], matrix[i * width + j].isNotWall });
				matrix[(i + row[k]) * width + (j + col[k])].parent = &matrix[i * width + j];
//	                cout << "parent setted: " << matrix[(i + row[k]) * width + (j + col[k])].parent << endl;
			}
		}
	}
	delete matrix;
	delete visited;
}

