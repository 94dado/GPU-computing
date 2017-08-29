#include <iostream>
#include <queue>
#include <vector>

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
void PrintNodeMaze(NodeStruct *array, int width, int height){
    int i,j;
    for(i = 0; i < height; i++){
        for(j = 0; j < width; j++){
            if ((array[i*width + j].x == startPath[0] && array[i*width + j].y == startPath[1]) || (array[i*width + j].x == endPath[0] && array[i*width + j].y == endPath[1])) {
                cout << 2 << " ";
            }
            else {
                cout << array[i*width + j].isNotWall << " ";
            }
        }
        cout << endl;
    }
    cout << endl;
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
    NodeStruct matrix[width * height];
    FromCoordToNodeStruct(matrix, mat, width, height);
    // construct a matrix to keep track of visited cells
    bool visited[width * height];

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
//            PrintNodeMaze(matrix, width, height);
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

//    if (min_dist != INT_MAX)
//        cout << "The shortest path from source to destination "
//        "has length " << min_dist << endl;
//    else
//        cout << "Destination can't be reached from given source" << endl;
}

__global__ void GPU_FromCoordToNodeStruct(NodeStruct *matrix, int *mat, int width, int height, int i, int *startPath, int *endPath){
	bool findStart = false;
	int j = blockIdx.x * width + threadIdx.x;
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

__global__ void setupVisited(bool *visited, int width, int height){
	int k = blockIdx.x * width + threadIdx.x;
	visited[k] = false;
}

void GPU_bfs_maze_solver(int *mat, int width, int height){
	// pass to NodeStruct coordinates
	    NodeStruct matrix[width * height];

	    //cuda variable
	    NodeStruct *dev_matrix;
	    int *dev_startPath, *dev_endPath;
	    //memory allocation
	    cudaMalloc(&dev_matrix, sizeof(NodeStruct) * width * height);
	    cudaMalloc(&dev_startPath, sizeof(int) * 2);
	    cudaMalloc(&dev_endPath, sizeof(int) * 2);
	    //copy data on GPU
	    cudaMemcpy(&dev_matrix, matrix, sizeof(NodeStruct) * width * height, cudaMemcpyHostToDevice);

	    for(int i = 0; i < width; i++){
	    	GPU_FromCoordToNodeStruct<<<width/32, 32>>>(matrix, mat, width, height, i, dev_startPath, dev_endPath);
	    }
	    cudaDeviceSynchronize();
	    //get back all the data
	    cudaMemcpy(matrix, dev_matrix, sizeof(NodeStruct) * width * height, cudaMemcpyDeviceToHost);
	    cudaMemcpy(startPath, dev_startPath, sizeof(int) * 2, cudaMemcpyDeviceToHost);
	    cudaMemcpy(endPath, dev_endPath, sizeof(int) * 2, cudaMemcpyDeviceToHost);

	    // construct a matrix to keep track of visited cells
	    bool visited[width * height];

	    bool *dev_visited;
	    cudaMalloc(&dev_visited, sizeof(bool) * width * height);
	    // initially all cells are unvisited
	   setupVisited<<<width, height>>>(dev_visited, width, height);
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
	        if (i == endPath[0] && j == endPath[1]) {
	            ReachPath(matrix, width, height);
	            PrintNodeMaze(matrix, width, height);
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
}

