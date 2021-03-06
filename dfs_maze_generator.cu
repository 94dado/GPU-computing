#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "Header/common.h"

#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

using namespace std;

int start_axis;
int start_side;

vector< vector< int > > dfs_path;
/*
 * Structure of the maze vector:
 *                     |--> Filled in?
 *   Row --> Column --|
 *                     |--> Has been visited?
 */
vector< vector< vector< bool > > > maze;

// Select a random direction based on our options, append it to the current path, and move there
bool randomMove(int *maze_size, bool first_move){
    int random_neighbor;
    vector< vector< int > > unvisited_neighbors;

    for(int direction = 0; direction < 4; direction++){
        int possible_pmd[2] = {0, 0};

        if(direction == UP){
            possible_pmd[1] = -1;
		}
		else if(direction == DOWN){
            possible_pmd[1] = 1;
		}
		else if(direction == LEFT){
            possible_pmd[0] = -1;
		}
		else {
            possible_pmd[0] = 1;
        }

        if(dfs_path.back()[0] + possible_pmd[0] * 2 > 0 &&
           dfs_path.back()[0] + possible_pmd[0] * 2 < maze_size[0] - 1 &&
           dfs_path.back()[1] + possible_pmd[1] * 2 > 0 &&
           dfs_path.back()[1] + possible_pmd[1] * 2 < maze_size[1] - 1){
            if(!maze[dfs_path.back()[1] + possible_pmd[1] * 2][dfs_path.back()[0] + possible_pmd[0] * 2][1]){
                vector< int > possible_move_delta = {possible_pmd[0], possible_pmd[1]};

                unvisited_neighbors.push_back(possible_move_delta);
            }
        }
    }

    if(unvisited_neighbors.size() > 0){
        random_neighbor = rand() % unvisited_neighbors.size();

        for(int a = 0; a < !first_move + 1; a++){
            vector< int > new_location;

            new_location.push_back(dfs_path.back()[0] + unvisited_neighbors[random_neighbor][0]);
            new_location.push_back(dfs_path.back()[1] + unvisited_neighbors[random_neighbor][1]);

            dfs_path.push_back(new_location);

            maze[dfs_path.back()[1]][dfs_path.back()[0]][0] = false;
            maze[dfs_path.back()[1]][dfs_path.back()[0]][1] = true;
        }

        return true;
    } else {
        return false;
    }
}

// The fun part ;)
void generateMaze(int *maze_size){
    bool first_move = true;
    bool success = true;

    while((int) dfs_path.size() > 1 - first_move){
        if(!success){
            dfs_path.pop_back();

            if(!first_move && dfs_path.size() > 2){
                dfs_path.pop_back();
			}
			else {
                break;
            }

            success = true;
        }

        while(success){
            success = randomMove(maze_size, first_move);

            if(first_move){
                first_move = false;
            }
        }
    }
}

// Initialize the maze vector with a completely-filled grid with the size the user specified
void initializeMaze(int *maze_size){
    for(int a = 0; a < maze_size[1]; a++){
        for(int b = 0; b < maze_size[0]; b++){
            bool is_border;

            if(a == 0 || a == maze_size[1] - 1 ||
               b == 0 || b == maze_size[0] - 1){
                is_border = true;
			}
			else {
                is_border = false;
            }

            vector< bool > new_cell = {true, is_border};

            if((unsigned int) a + 1 > maze.size()){
                vector< vector< bool > > new_row = {new_cell};

                maze.push_back(new_row);
			}
			else {
                maze[a].push_back(new_cell);
            }
        }
    }
}

int *DFSToCoord(int *coordMaze){
    int width = maze.size();
    for(int a = 0; a < width; a++){
        for(int b = 0; b < maze[a].size(); b++){
            if ((a == 0 && maze[a][b][0] == 0) || (a == width-1 && maze[a][b][0] == 0) || (b == 0 && maze[a][b][0] == 0) || (b == maze[a].size()-1 && maze[a][b][0] == 0)) {
                coordMaze[a*width + b] = OBJECTIVE;
            }
            else {
                if (maze[a][b][0]) {
                    coordMaze[a*width + b] = WALL;
                }
                else {
                    coordMaze[a*width + b] = OPEN;
                }
            }
        }
	}
	return coordMaze;
}

// Set a random point (start or end)
void randomPoint(int *maze_size, bool part){
    int axis;
    int side;

    if(!part){
        axis = rand() % 2;
        side = rand() % 2;

        start_axis = axis;
        start_side = side;
	}
	else {
        bool done = false;

        while(!done){
            axis = rand() % 2;
            side = rand() % 2;

            if(axis != start_axis ||
               side != start_side){
                done = true;
            }
        }
    }

    vector< int > location = {0, 0};

    if(!side){
        location[!axis] = 0;
	}
	else {
        location[!axis] = maze_size[!axis] - 1;
    }

    location[axis] = 2 * (rand() % ((maze_size[axis] + 1) / 2 - 2)) + 1;

    if(!part){
        dfs_path.push_back(location);
    }

    maze[location[1]][location[0]][0] = false;
    maze[location[1]][location[0]][1] = true;
}

// The width and height must be greater than or equal to 4 or it won't work
// The width and height must be odd or else it won't work
void CPU_dfs_maze_generator(int *coordMaze, int width, int height){
	srand(time(NULL));

	int maze_size[2] = {width, height};

    // The width and height must be greater than or equal to 5 or it won't work (still not working)
    // The width and height must be odd or else we will have extra walls
    for(int a = 0; a < 2; a++){
        if(maze_size[a] < 5){
            maze_size[a] = 5;
		}
		else if(maze_size[a] % 2 == 0){
            maze_size[a]--;
        }
    }

    initializeMaze(maze_size);
    randomPoint(maze_size, false);
    randomPoint(maze_size, true);
    generateMaze(maze_size);

    DFSToCoord(coordMaze);
}




vector< vector< vector< bool>>> arrayToVec( bool *maze, int width, int height);
void PrintBoolMaze(bool *array, int width, int height);
void PrintPath();




// Initialize the maze vector with a completely-filled grid with the size the user specified
__global__ void GPU_initializeMaze(bool *maze, int width, int height, int offset){
	int b = offset + threadIdx.x;	//width
	int a = blockIdx.x;				//height
	bool is_border;

	if(a == 0 || a == height - 1 ||
	   b == 0 || b == width - 1){
		is_border = true;
	}
	else {
		is_border = false;
	}
	maze[a * width * 2 + (2*b)] = true;
	maze[a * width * 2 + (2*b) + 1] = is_border;
//	printf("a:%d, b:%d, values:(%d,%d)\n",a,b,maze[a * width * 2 + (2*b)],maze[a * width * 2 + (2*b) + 1]);

}

// Set a random point (start or end)
void GPU_randomPoint(bool *maze, int *maze_size, bool part){
    int axis;
    int side;

    if(!part){
        axis = rand() % 2;
        side = rand() % 2;

        start_axis = axis;
        start_side = side;
	}
	else {
        bool done = false;

        while(!done){
            axis = rand() % 2;
            side = rand() % 2;

            if(axis != start_axis ||
               side != start_side){
                done = true;
            }
        }
    }

    vector <int>location = {0, 0};

    if(!side){
        location[!axis] = 0;
	}
	else {
        location[!axis] = maze_size[!axis] - 1;
    }

    location[axis] = 2 * (rand() % ((maze_size[axis] + 1) / 2 - 2)) + 1;

    if(!part){
        dfs_path.push_back(location);
    }
    maze[location[1] * maze_size[0] * 2 + (2 * location[0])] = false;
    maze[location[1] * maze_size[0] * 2 + (2 * location[0] + 1)] = true;
}

// Select a random direction based on our options, append it to the current path, and move there
bool GPU_randomMove(bool *maze, int *maze_size, bool first_move){
    int random_neighbor;
    vector< vector< int > > unvisited_neighbors;

    for(int direction = 0; direction < 4; direction++){
        int possible_pmd[2] = {0, 0};

        if(direction == UP){
            possible_pmd[1] = -1;
		}
		else if(direction == DOWN){
            possible_pmd[1] = 1;
		}
		else if(direction == LEFT){
            possible_pmd[0] = -1;
		}
		else {
            possible_pmd[0] = 1;
        }

        if(dfs_path.back()[0] + possible_pmd[0] * 2 > 0 &&
           dfs_path.back()[0] + possible_pmd[0] * 2 < maze_size[0] - 1 &&
           dfs_path.back()[1] + possible_pmd[1] * 2 > 0 &&
           dfs_path.back()[1] + possible_pmd[1] * 2 < maze_size[1] - 1){
            if(!maze[(dfs_path.back()[1] + possible_pmd[1] * 2) * maze_size[0] * 2 + (dfs_path.back()[0] + possible_pmd[0] * 2) * 2 + 1]){
                vector< int > possible_move_delta = {possible_pmd[0], possible_pmd[1]};

                unvisited_neighbors.push_back(possible_move_delta);
            }
        }
    }

    if(unvisited_neighbors.size() > 0){
        random_neighbor = rand() % unvisited_neighbors.size();

        for(int a = 0; a < !first_move + 1; a++){
            vector< int > new_location;

            new_location.push_back(dfs_path.back()[0] + unvisited_neighbors[random_neighbor][0]);
            new_location.push_back(dfs_path.back()[1] + unvisited_neighbors[random_neighbor][1]);

            dfs_path.push_back(new_location);
//            cout << "signed:(" << dfs_path.back()[0];
//            cout << "," << (dfs_path.back()[1]);
//            cout << ")" << endl;
            maze[(dfs_path.back()[1] * 2 * maze_size[0]) + (2 * dfs_path.back()[0])] = false;
            maze[(dfs_path.back()[1] * 2 * maze_size[0]) + (2 * dfs_path.back()[0]) + 1] = true;
        }

        return true;
    } else {
        return false;
    }
}

// The fun part ;)
void GPU_generateMaze(int *maze_size, bool *maze){
    bool first_move = true;
    bool success = true;

    while((int) dfs_path.size() > 1 - first_move) {
//    	cout << "path:" << endl;
//    	PrintPath();
        if(!success){
            dfs_path.pop_back();

            if(!first_move && dfs_path.size() > 2){
                dfs_path.pop_back();
			}
			else {
                break;
            }

            success = true;
        }

        while(success){
//        	cout << "before random" << endl;
//        	PrintBoolMaze(maze,maze_size[0],maze_size[1]);
            success = GPU_randomMove(maze, maze_size, first_move);
//            cout << "after random" << endl;
//            PrintBoolMaze(maze,maze_size[0],maze_size[1]);
            if(first_move){
                first_move = false;
            }
        }
    }
}

__global__ void GPU_putCoord(bool *maze, int *coordMaze, int width, int height, int offset){
	int a = blockIdx.x;			//height
	int b = offset + threadIdx.x;		//width
	if ((a == 0 && maze[a * width * 2 + (2*b)] == 0) || (a == height-1 && maze[a * width * 2 + (2*b)] == 0) || (b == 0 && maze[a * width * 2 + (2*b)] == 0) || (b == width -1 && maze[a * width * 2 + (2*b)] == 0)) {
//		printf("a:%d, b:%d\n",a,b);
		coordMaze[a*width + b] = OBJECTIVE;
	}
	else {
		if (maze[a * width * 2 +(2*b)] == 1) {
			coordMaze[a*width + b] = WALL;
		}
		else {
			coordMaze[a*width + b] = OPEN;
		}
	}
}

// The width and height must be greater than or equal to 4 or it won't work
// The width and height must be even or else it won't work
void GPU_dfs_maze_generator(int *coordMaze, int width, int height){
	srand(time(NULL));
	if (width %2 == 0) width--;
	if (height%2 == 0) height--;
	int mazeSize[] = {width, height};
	bool *maze = new bool[height * width * 2];
	bool *dev_maze;
	int *dev_coordMaze;

	cudaMalloc(&dev_maze, sizeof(bool) * 2 * width * height);
	cudaMalloc(&dev_coordMaze, sizeof(int) * width * height);
	int max_rec = width / MAX_THREAD;
	int offset = 0;
	for(int i = 0; i < max_rec; i++){
		GPU_initializeMaze<<<height,MAX_THREAD>>>(dev_maze, width, height, offset);
		offset = (i + 1) * MAX_THREAD;
	}
	GPU_initializeMaze<<<height,width % MAX_THREAD>>>(dev_maze, width, height, offset);
	cudaDeviceSynchronize();
	//copy data on cpu
	cudaMemcpy(maze,dev_maze, sizeof(bool) * 2 * width * height, cudaMemcpyDeviceToHost);
//	PrintBoolMaze(maze, width, height);
	GPU_randomPoint(maze, mazeSize, false);
	GPU_randomPoint(maze, mazeSize, true);

	GPU_generateMaze(mazeSize, maze);
	cudaMemcpy(dev_maze, maze, sizeof(bool) * 2 * width * height, cudaMemcpyHostToDevice);
	offset = 0;
	for(int i = 0; i < max_rec; i++){
		GPU_putCoord<<<height,MAX_THREAD>>>(dev_maze, dev_coordMaze, width, height, offset);
		offset = (i + 1) * MAX_THREAD;
	}
	GPU_putCoord<<<height,width % MAX_THREAD>>>(dev_maze, dev_coordMaze, width, height, offset);
	cudaMemcpy(coordMaze, dev_coordMaze, sizeof(int) * width * height, cudaMemcpyDeviceToHost);

	delete maze;
}

vector< vector< vector< bool>>> arrayToVec( bool *maze, int width, int height){
	vector< vector< vector< bool>>> dest;
	for(int i = 0; i< width; i++){
		dest[i].push_back({});
		for(int j=0; j<height;j++){
			dest[i][j].push_back(maze[i*width*2 + j*2]);
			dest[i][j].push_back(maze[i*width*2 + j*2 + 1]);
		}
	}
	return dest;
}

void PrintBoolMaze(bool *array, int width, int height){
	int i,j;
	for(i = 0; i < height; i++){
		for(j = 0; j < width; j++){
			cout << "(" << array[i*height*2 + j*2] << "," << array[i*height*2 + j*2 + 1] << ") ";
		}
		cout << endl;
	}
	cout << endl;
}

void PrintPath(){
	for(int i = 0; i < dfs_path.size(); i++){
		cout << "(";
		for(int j=0; j < dfs_path[i].size(); j++){
			cout << dfs_path[i][j] << " ";
		}
		cout << ")" << endl;
	}
}
