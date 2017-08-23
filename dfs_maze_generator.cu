#include <cuda_runtime.h>
#include <iostream>
#include "./Header/common.h"

using namespace std;

void init(Node *nodes, int width, int height) {
	int i, j;
	struct Node *n;

	//Setup crucial nodes
	for (i = 0; i < width; i++) {
		for (j = 0; j < height; j++) {
			n = nodes + i + j * width;
			if (i * j % 2) {
				n->x = i;
                n->y = j;
                //Assume that all directions can be explored (4 youngest bits set)
				n->dirs = 15;
				n->isSpace = OPEN;
            }
            //Add walls between nodes
			else {
                n->isSpace = WALL;
            }
		}
	}
}

Node *link(Node *n, Node *nodes, int width, int height) {
	//Connects node to random neighbor (if possible) and returns
	//address of next node that should be visited

	int x, y;
	char dir;
	struct Node *dest;

	//Nothing can be done if null pointer is given - return
	if (n == NULL) {
        return NULL;
    }

	//While there are directions still unexplored
	while (n->dirs) {
		//Randomly pick one direction
		dir = (1 << (rand() % 4));

		//If it has already been explored - try again
		if (~n->dirs & dir) continue;

		//Mark direction as explored
		n->dirs &= ~dir;

		//Depending on chosen direction
		switch (dir) {
			//Check if it's possible to go right
			case 1:
				if (n->x + 2 < width) {
					x = n->x + 2;
					y = n->y;
				}
				else continue;
				break;

			//Check if it's possible to go down
			case 2:
				if (n->y + 2 < height) {
					x = n->x;
					y = n->y + 2;
				}
				else continue;
				break;

			//Check if it's possible to go left
			case 4:
				if (n->x - 2 >= 0) {
					x = n->x - 2;
					y = n->y;
				}
				else continue;
				break;

			//Check if it's possible to go up
			case 8:
				if (n->y - 2 >= 0) {
					x = n->x;
					y = n->y - 2;
				}
				else continue;
				break;
		}

		//Get destination node into pointer (makes things a tiny bit faster)
		dest = nodes + x + y * width;

		//Make sure that destination node is not a wall
		if (dest->isSpace == OPEN) {
			//If destination is a linked node already - abort
			if (dest->parent != NULL) continue;

			//Otherwise, adopt node
			dest->parent = n;

			//Remove wall between nodes
			nodes[n->x + (x - n->x) / 2 + (n->y + (y - n->y) / 2) * width].isSpace = OPEN;

			//Return address of the child node
			return dest;
		}
	}

	//If nothing more can be done here - return parent's address
	return n->parent;
}

// dimensions must be odd and greater than 0
void dfs_maze_generator (int *maze, int width, int height) {
    // Nodes array
    struct Node nodes[width * height];
    int i, badarg;
	long seed;
	struct Node *start, *last;

	//Initialize maze
	init(nodes, width, height);

	//Setup start node
	start = nodes + 1 + width;
	start->parent = start;
	last = start;

	//Connect nodes until start node is reached and can't be left
    while ((last = link(last, nodes, width, height)) != start);

    // move to grid from node
    FromNodeToGrid(nodes, maze, width, height);

    // print grid on terminal
    PrintMaze(maze, width, height);
}

int main(int argc, char **argv) {
    int width = 99;
    int height = 99;
    int maze [width * height];
    dfs_maze_generator(maze, width, height);
}
