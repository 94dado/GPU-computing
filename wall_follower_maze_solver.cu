#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "./Header/common.h"

void CPU_wall_follower_maze_solver(int *maze, int length, int row_length){
	//directions
	int north = -row_length, south = row_length, west = -1, east = 1;
	int directions[] = {east,south,west,north};
	int current;
	//inizializzo lista celle soluzione
	ListNode solutions;
	init_list(&solutions);

	srand(time(NULL));
	//cerco la cella da cui iniziare
	int i = 0;
	bool again = true;
	while(again && i < length){
		if(maze[i] == OBJECTIVE){
			again = false;
		}
	}
	current = i;
}
