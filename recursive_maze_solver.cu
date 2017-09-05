#include <iostream>
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
            // start
            if (i == pathStart[0] && j == pathStart[1]) {
                mazeInt[i * width + j] = OBJECTIVE;
            }
            // end
            else if (maze[i * width + j] == 'G') {
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
    char mazeChar[width * height];

    FromCoordToChar(mazeInt, mazeChar, width, height);

    find_path(mazeChar, pathStart[0], pathStart[1], width, height);

    FromCharToCoord(mazeInt, mazeChar, width, height);

    PrintMaze(mazeInt, width, height);
}

void GPU_recursive_maze_solver(int *mazeInt, int width, int height){

}
