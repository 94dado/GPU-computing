#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "./Header/common.h"
#include <vector>
#include <iostream>

using namespace std;

// array with all nodes
vector<vector<Node> > nodes;
// saves father of node at index
vector<Node> father;
// all safe edges
vector<Edge> safe;
// array with all nodes and edges of mst
vector<Node> mst;

// initialize vectors
void init(int width, int height) {
    for (int i = 0; i < height; i++) {
        nodes.push_back(vector<Node>(width));
    }
    for (int i = 0; i < width * height; i++) {
        father.push_back(Node());
    }
}

// fills array with nodes and sets up all edges.
void setup (int width, int height) {
    // loops through array to fill with nodes.
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            nodes[i][j] = Node(j, i, i * height + j);
        }
    }

    // Loops through array and sets up all horizontal Edges.
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height - 1; j++) {
            // Creates a new Edge between a Node and its right neighbor and puts it in the Node's edArray
            nodes[i][j].ed[0] = Edge(&nodes[i][j], &nodes[i][j + 1], NULL);
            // Creates a new Edge between the same Nodes with the same weight and puts it in the other Node's edArray
            nodes[i][j + 1].ed[2] = Edge(&nodes[i][j + 1], &nodes[i][j], &nodes[i][j].ed[0]);
        }
    }

    // Loops through array and sets up all vertical Edges.
    for (int i = 0; i < width - 1; i++) {
        for (int j = 0; j < height; j++) {
            // Creates a new Edge between a Node and its lower neighbor and puts it in the Node's edArray
            nodes[i][j].ed[1] = Edge(&nodes[i][j], &nodes[i + 1][j], NULL);
            // Creates a new Edge between the same Nodes with the same weight and puts it in the other Node's edArray
            nodes[i + 1][j].ed[3] = Edge(&nodes[i + 1][j], &nodes[i][j], &nodes[i][j].ed[1]);
        }
    }

    father[0] = nodes[0][0]; // father of start node is start node

    safe.push_back(nodes[0][0].ed[0]);	// all edges of start node are safe.
    safe.push_back(nodes[0][0].ed[1]);

    nodes[0][0].isSpanning = true; // start node is already part of minimal spanning tree
    nodes[0][0].path = true; // start node also part of path to node at upper right corner

    mst.push_back(nodes[0][0]);
}

void prim(int width, int height) {
    for (int i = 0; i != width * height; i++) {
    	int count = 0;
        Edge *min = new Edge();

        // with each iteration (since a timer is used, rather with each call of this function)
        // the newly added node (min.b) is drawn onto the canvas with red color, to show how the algorithm works.
        // however, when a new node is added the one that used to be the most recent one will be colored white again.
        Node oldNode = mst[mst.size()-1];
        oldNode.current = false;

        /*
         * loops through all edges of array "safe". compares weight of all safe edges to get edge with least weight
         * edge also has to lead to a node that isn't part of MST (minimal spanning tree)
         * breaks if edge with minimum weight of 0 has been found.
         *
         * First if() to reduce size of list for runtime optimization.
         * Loops through list to find edges that dont have non-MST nodes anymore and removes them from list.
         */
        for (int l = 0; l < safe.size(); l++) {

            if (safe[l].a->isSpanning && safe[l].b->isSpanning) {
                safe.erase(safe.begin() + l);
                continue;
            }

            if ((safe[l].weight < min->weight) && (!(safe[l].b->isSpanning))) {
                min = &safe[l];
                if (min->weight == 0)
                    break;
            }
        }

        /*
         * all edges of the new node from the minimal edge are put into the safe list. light edges at front, heavy ones at end.
         */
        for (int i = 0; i < 4; i++) {
            if (min->b->ed[i].b != NULL && !(min->b->ed[i].b->isSpanning)) {
                if (min->b->ed[i].weight > 0)
                    // on tale
                    safe.push_back(min->b->ed[i]);
                else
                    // on head
                    safe.insert(safe.begin(), min->b->ed[i]);
            }
        }

        min->b->isSpanning = true;	         // new node from minimal edge ist part of MST
        min->b->current = true;		 // min.b is newest node of mst, is colored red.
        min->minimal = true;		     // edge is part of MST

        // add edge and node to mst
//        mst.push_back(min); non si puo' fare in c++
        mst.push_back(*min->b);
        //safe.remove(min);			 // removes the added edge from list for unneccessary further comparisons
        father[min->b->pos] = *min->a; // node a of edge is father of node b from the same edge
        count ++;						// addition of edge to MST
        if (count == (width * height) - 1) {    // algorithm complete. stop execution.
            min->b->current = false;
        }
    }
}

// solve the maze
void CPU_prim_solver(int u, int v) {
    Node k = nodes[u][v];
    while (father[k.pos].pos != k.pos) {
        k.path = true;
        k = father[k.pos];
    }
    nodes[0][0].path = true;
    //for (int i = 0; i < mst.size(); i++)
    //    cout << mst[i].pos << endl;
}

void CPU_prim_maze_generator(int *maze, int width, int height) {
    srand((unsigned)time(NULL));
    // initialize vectors
    init(height, width);
    // first function to get called. Calls all other needed functions-.
    // set end coord node
    int u = width -1;
    int v = height -1;
    setup(width, height);
    prim(width, height);
    CPU_prim_solver(u,v);
}

int main() {
    int width = 3;
    int height = 4;
    int maze[width * height];
    CPU_prim_maze_generator(maze, width, height);
    return 0;
}


