#include <iostream>
#include <queue>
#include <vector>
#include "./Header/common.h"

using namespace std;

// the start and the end of the path
int startPath[2];
int endPath[2];

// queue node used in BFS
struct Node {
    // (x, y) represents matrix cell cordinates
    // dist represent its minimum distance from the source
    int x, y, dist;
    Node *node;
    bool isNotWall;
};

// Below arrays details all 4 possible movements from a cell
int row[] = { -1, 0, 0, 1 };
int col[] = { 0, -1, 1, 0 };

// Function to check if it is possible to go to position (row, col)
// from current position. The function returns false if the cell
// not a valid position or has value 0 or it is already visited
bool isValid(Node *mat, bool *visited, int row, int col, int width, int height) {
    return (row >= 0) && (row < width) && (col >= 0) && (col < height) && mat[row * width + col].isNotWall && !visited[row * width + col];
}

// print the path
void PrintMaze(Node *array, int width, int height){
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

// Search the shortest path by parent node
void ReachPath(int maxDist, Node *matrix, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        matrix[i].isNotWall = WALL;
    }
    // the last is the end
    Node *currentNode = &matrix[endPath[0] * width + endPath[1]];
    // the first is the start
    Node *startNode = &matrix[startPath[0] * width + startPath[1]];
    while ((currentNode->x != startNode->x) && currentNode->y != startNode->y) {
        matrix[currentNode->x * width + currentNode->y].isNotWall = OPEN;
        currentNode = currentNode->node;
    }
}

// it find start and end for the parameter of the BFS algorithm
void FromCoordToNode(Node *matrix, int *mat, int width, int height) {
    bool findStart = false;
    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            if (mat[i*width + j] == OPEN) {
                // is not a wall
                matrix[i * width + j] = {i, j, 0, NULL, OPEN};
            }
            // is the start
            else if (mat[i*width + j] == OBJECTIVE && !findStart) {
                matrix[i * width + j] = {i, j, 0, NULL, OPEN};
                startPath[0] = i;
                startPath[1] = j;
                findStart = true;
            }
            // is the end
            else if (mat[i*width + j] == OBJECTIVE && findStart) {
                matrix[i * width + j] = {i, j, 0, NULL, OPEN};
                endPath[0] = i;
                endPath[1] = j;
            }
            else {
                matrix[i * width + j] = {i, j, 0, NULL, WALL};
            }
        }
    }
}

// Find Shortest Possible Route in a matrix mat from source
// cell (i, j) to destination cell (x, y)
void CPU_bfs_maze_solver(int *mat, int width, int height) {
    // pass to node coordinates
    Node matrix[width * height];
    FromCoordToNode(matrix, mat, width, height);
    // construct a matrix to keep track of visited cells
    bool visited[width * height];

    // initially all cells are unvisited
    for (int k = 0; k < width * height; k++) {
        visited[k] = WALL;
    }

    // create an empty queue
    queue<Node> q;

    // mark source cell as visited and enqueue the source node
    visited[startPath[0] * width + startPath[1]] = OPEN;
    q.push(matrix[startPath[0] * width + startPath[1]]);

    // stores length of longest path from source to destination
    int min_dist = INT_MAX;

    // run till queue is not empty
    while (!q.empty()) {
        // pop front node from queue and process it
        Node node = q.front();
        q.pop();

        // (i, j) represents current cell and dist stores its
        // minimum distance from the source
        int i = node.x, j = node.y, dist = node.dist;

        // if destination is found, update min_dist and stop
        if (i == endPath[0] && j == endPath[1]) {
            min_dist = dist;
            ReachPath(min_dist, matrix, width, height);
            PrintMaze(matrix, width, height);
            break;
        }

        // check for all 4 possible movements from current cell
        // and enqueue each valid movement into the queue
        for (int k = 0; k < 4; k++) {
            // check if it is possible to go to position
            // (i + row[k], j + col[k]) from current position
            if (isValid(matrix, visited, i + row[k], j + col[k], width, height)) {
                // mark next cell as visited and enqueue it
                visited[(i + row[k]) * width + (j + col[k])] = OPEN;
                q.push({ i + row[k], j + col[k], dist + 1, &matrix[i * width + j], matrix[i * width + j].isNotWall });
                matrix[(i + row[k]) * width + (j + col[k])].node = &matrix[i * width + j];
            }
        }
    }

    if (min_dist != INT_MAX)
        cout << "The shortest path from source to destination "
        "has length " << min_dist << endl;
    else
        cout << "Destination can't be reached from given source" << endl;
}
