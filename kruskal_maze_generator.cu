#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "./Header/common.h"

using namespace std;

// We are going to represent the two-dimensinal array as a one-dimensional
// vector. This will allow us to represent the edges in a shorter way.
// In this case, manual calculations are necessary.
inline int Pos2Idx (int row, int column, int width) {
  return row * width + column;
}

inline pair<int, int> Idx2Pos(int idx, int width) {
  int row = idx / width;
  int column = idx % width;
  return pair<int, int>(row, column);
}

// pair < node A, node B >
void CreateEdges(vector<pair<int, int> > *edges, int width, int height) {
  for (int row = 1; row < height; ++row) {
    for (int column = 1; column < width; ++column) {
      edges->push_back(pair<int, int>(
          Pos2Idx(row, column, width), Pos2Idx(row, column - 1, width)));
      edges->push_back(pair<int, int>(
          Pos2Idx(row, column, width), Pos2Idx(row - 1, column, width)));
    }
  }

  for (int column = 1; column < width; ++column) {
    edges->push_back(pair<int, int>(
          Pos2Idx(0, column, width), Pos2Idx(0, column - 1, width)));
  }

  for (int row = 1; row < height; ++row) {
    edges->push_back(pair<int, int>(
          Pos2Idx(row, 0, width), Pos2Idx(row - 1, 0, width)));
  }
}

void Kruskal(const vector<pair<int, int> > &all_edges, vector<pair<int, int> > * mst_edges, int width, int height) {
  Union_Find uf;
  uf.Reset(width * height);
  for (int i = 0; i < all_edges.size(); ++i) {
    if (uf.Union(all_edges[i].first, all_edges[i].second)) {
      mst_edges->push_back(all_edges[i]);
    }
  }
}

// The code below is for the drawing part.

// Convert each edge to the postion in the matrix that must be drawn.
void ConvertToPaintedPoint(const vector<pair<int, int> > &edges, vector<pair<int, int> > *painted_points, int width) {
  for (int i = 0; i < edges.size(); ++i) {
    pair<int, int> node_a = Idx2Pos(edges[i].first, width);
    pair<int, int> node_b = Idx2Pos(edges[i].second, width);

    // Convert the vertices positions to their drawn version.
    node_a.first *= 2;
    node_a.second *= 2;
    node_b.first *= 2;
    node_b.second *= 2;

    // Calculate the position of their edge.
    pair<int, int> edge;
    edge.first = (node_a.first + node_b.first) / 2;
    edge.second = (node_a.second + node_b.second) / 2;

    painted_points->push_back(edge);
  }
  sort(painted_points->begin(), painted_points->end());
}

// Convert Kruskal to maze
int *GenerateMaze(const vector<pair<int, int> > &painted_points, int *largeMaze, int width, int height, int size) {
  int count = 0;
  int pointer = 0;
  largeMaze[pointer] = WALL;
  largeMaze[pointer+1] = OBJECTIVE;
  largeMaze[pointer+2] = OPEN;
  largeMaze[pointer+3] = OPEN;
  pointer = 4;
  for (int column = 0; column < (width * 2) - 1; ++column) {
    largeMaze[pointer] = WALL;
    largeMaze[pointer + 1] = OPEN;
    pointer += 2;
  }
  for (int column = 0; column < size-1; ++column) {
    largeMaze[pointer] = OPEN;
    pointer ++;
  }
  for (int row = 0; row < (height * 2) - 1; ++row) {
    largeMaze[pointer] = WALL;
    largeMaze[pointer + 1] = OPEN;
    pointer += 2;
    for (int column = 0; column < (width * 2) - 1; ++column) {
      // If it's a vertex position
      if ((row & 1) == 0 && (column & 1) == 0) {
        largeMaze[pointer] = OPEN;
        largeMaze[pointer + 1] = OPEN;
        pointer += 2;
      }
      else {
        if (count < painted_points.size() && painted_points[count].first == row && painted_points[count].second == column) {
          largeMaze[pointer] = OPEN;
          pointer++;
          count++;
        }
        else {
          largeMaze[pointer] = WALL;
          pointer++;
        }
        largeMaze[pointer] = OPEN;
        pointer++;
      }
    }
    largeMaze[pointer] = WALL;
    pointer++;
    for (int column = 0; column < size; ++column) {
      largeMaze[pointer] = OPEN;
      pointer ++;
    }
  }
  for (int column = 0; column < (width * 2) - 1; ++column) {
    largeMaze[pointer] = WALL;
    largeMaze[pointer + 1] = OPEN;
    pointer += 2;
  }
  // add end node
  largeMaze[pointer] = OPEN;
  largeMaze[pointer+1] = OBJECTIVE;
  largeMaze[pointer+2] = WALL;
  return largeMaze;
}

// Convert maze to matrix
int *FromMazeToGrid(int *largeMaze, int *maze, int width, int size) {
  int count = 2;
  maze[0] = WALL;
  maze[1] = OBJECTIVE;
  for (int i = 0; i < width-2; i++) {
    maze[count] = WALL;
    count++;
  }
  for (int i = 2; i < size-2; i += 2) {
    for (int j = 0; j < size; j += 2) {
      if (j+1 == size) {
        maze[count] = WALL;
      }
      else {
        // if is open
        if (largeMaze[i * size + j] == OPEN) {
          maze[count] = OPEN;
        }
        else {
          maze[count] = WALL;
        }
      }
      count++;
    }
  }
  for (int i = 0; i < width-2; i++) {
    maze[count] = WALL;
    count++;
  }
  maze[count] = OBJECTIVE;
  maze[count+1] = WALL;
  return maze;
}

// width and height more then 2
void CPU_kruskal_maze_generator(int *maze, int width, int height) {
  srand(time(0));
  int newWidth = (width -1)/2;
  int newHeight = (height -1)/2;
  vector<pair<int, int> > all_edges;
  vector<pair<int, int> > mst_edges;
  vector<pair<int, int> > points;
  CreateEdges(&all_edges, newWidth, newHeight);
  // Please, note that giving random weights and sorting a list of edges is
  // the same as just shuffling it. The later is O(E) against O(E log E) of
  // the former, though.
  // Also, note that if we are making a maze using a four-sided grid,
  // E ~= H * W * 4.
  // Another important note is that the grid to be drawn is twice the size of
  // the "actual" grid. All positions of the type (even, even) are vertices,
  // while the remaining ones can be edges (or the lack of edges).
  // Sort the edges to draw them.
  ShuffleVector(&all_edges);
  Kruskal(all_edges, &mst_edges, newWidth, newHeight);
  ConvertToPaintedPoint(mst_edges, &points, newWidth);
  int size = ((newWidth * 4) + 1);
  int largeMaze[size * size];
  GenerateMaze(points, largeMaze, newWidth, newHeight, size);
  FromMazeToGrid(largeMaze, maze, width, size);
  PrintMaze(maze, width, height);
}
