#ifndef _node_
#define _node_ 1

#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

class Node;

// edge with weight and two nodes
class Edge {
public:
    Node *a;             // node at the end of edge
    Node *b;             // node at the end of edge
    int weight;            // weight of edge. either 0 or 1
    bool minimal = false;   // true if part of minimal spanning tree


    // constructor
    Edge(Node *_a, Node *_b, Edge *_edge) {
        if (_a == NULL && _b == NULL && _edge == NULL)
            weight = 100;
        else if (_edge == NULL) {
            weight = round(rand());
            weight = weight % 2;
            a = _a;
            b = _b;
        }
        else {
            weight = _edge->weight;
            a = _a;
            b = _b;
        }
    }

    Edge(){
        weight = 100;
    }
};

// Node with edges
class Node {
    
public:
    int x;         // coords
    int y;
    int pos; // position in a simulated one dimensional array

    bool isSpanning = false;  // true when it becomes part of minimal spanning tree
    bool path = false;  // true when node is on path from start node to end node
    bool current = false;  // if true, node is the newest node of mst
    vector<Edge> ed;      // array with all of node's edges

    // constructor
    Node(int _x, int _y, int _pos) {
        x = _x;
        y = _y;
        pos = _pos;
        for (int i = 0; i < 4; i++) {
            ed.push_back(Edge());
        }
    }
    
    Node(){
        pos = -1;
    }
};

#endif

