#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "Header/common.h"

using namespace std;

// Shuffle a vector
void ShuffleVector (vector< pair<int,int> > *vector) {
	for (int i = 0; i < vector->size(); ++i) {
		swap((*vector)[i], (*vector)[rand() % vector->size()]);
	}
}

// We are going to represent the two-dimensinal array as a one-dimensional
// vector. This will allow us to represent the edges in a shorter way.
// In this case, manual calculations are necessary.
int Pos2Idx (int row, int column, int width) {
  return row * width + column;
}

pair<int, int> Idx2Pos(int idx, int width) {
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
//	cout << "size: " << edges.size() << endl;
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
//  cout << "size2: " << painted_points->size() << endl;
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
}

__device__ int GPU_Pos2Idx (int row, int column, int width) {
  return row * width + column;
}

__device__ void GPU_Idx2Pos(int idx, int width, pair<int, int> *point) {
  int row = idx / width;
  int column = idx % width;
  point->first = row;
  point->second = column;
}

__global__ void CreateFirstEdges(pair<int, int> *edges, int row, int height, int width) {
	int column = threadIdx.x;
	if (column == 0) {
		return;
	}
//	printf("column:%d, index:%d\n",column,(row-1) * height * 2 + (column-1) * 2);
	edges[(row-1) * height * 2 + (column-1) * 2].first = GPU_Pos2Idx(row, column, width);
	edges[(row-1) * height * 2 + (column-1) * 2].second = GPU_Pos2Idx(row, column - 1, width);
	edges[(row-1) * height * 2 + (column-1) * 2 + 1].first = GPU_Pos2Idx(row, column, width);
	edges[(row-1) * height * 2 + (column-1) * 2 + 1].second = GPU_Pos2Idx(row - 1, column, width);
}

__global__ void CreateSecondEdges(pair<int, int> *edges, int height, int width) {
	int column = threadIdx.x;
	if (column == 0) {
		return;
	}
//	printf("column2:%d, index:%d\n",column, column + (width-1) * (height-1));
	edges[column + (width-1) * (height-1)].first = GPU_Pos2Idx(0, column, width);
	edges[column + (width-1) * (height-1)].second = GPU_Pos2Idx(0, column - 1, width);
}

__global__ void CreateThirdEdges(pair<int, int> *edges, int height, int width) {
	int row = threadIdx.x;
	if (row == 0) {
		return;
	}
//	printf("row:%d, index:%d\n",row, row + (width-1) * (height-1) + (width -1));
	edges[row + (width-1) * (height-1) + (width -1)].first = GPU_Pos2Idx(row, 0, width);
	edges[row + (width-1) * (height-1) + (width -1)].second = GPU_Pos2Idx(row - 1, 0, width);
}

// Shuffle a matrix
void GPU_ShuffleMatrix (pair<int,int> *edges, int width, int height) {
	for(int index = 0; index < width * height * 2; index++){
		int random = rand() %(width * height * 2);
		pair<int,int> temp = edges[index];
		edges[index] = edges[random];
		edges[random] = temp;
	}
}

// Convert each edge to the postion in the matrix that must be drawn.
__global__ void GPU_ConvertToPaintedPoint(pair<int, int> *edges, pair<int, int> *painted_points, int width, int dim, pair<int, int> *node_a,  pair<int, int> *node_b,  pair<int, int> *edge) {
	int index = threadIdx.x;
	if (index >= dim) {
		return;
	}
    GPU_Idx2Pos(edges[index].first, width, &node_a[index]);
    GPU_Idx2Pos(edges[index].second, width, &node_b[index]);

    // Convert the vertices positions to their drawn version.
    node_a[index].first *= 2;
    node_a[index].second *= 2;
    node_b[index].first *= 2;
    node_b[index].second *= 2;

    // Calculate the position of their edge.
    edge[index].first = (node_a[index].first + node_b[index].first) / 2;
    edge[index].second = (node_a[index].second + node_b[index].second) / 2;

    painted_points[index].first = edge[index].first;
    painted_points[index].second = edge[index].second;
}

// Convert Kruskal to maze
int *GPU_GenerateMaze(pair<int, int> *painted_points, int *largeMaze, int width, int height, int size, int dim) {
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
        if (count < dim && painted_points[count].first == row && painted_points[count].second == column) {
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

// pair < node A, node B >
void GPU_CreateEdges(pair<int, int> *dev_edges, int width, int height) {
	for (int row = 1; row < height; ++row) {
//		cout << "width: " << width << endl;
		CreateFirstEdges<<<1, width>>>(dev_edges, row, height, width);
	}

	CreateSecondEdges<<<1, width>>>(dev_edges, height, width);

	CreateThirdEdges<<<1, height>>>(dev_edges, height, width);

	cudaDeviceSynchronize();
}

int GPU_Kruskal(pair<int, int> *all_edges, pair<int, int> *mst_edges, int width, int height) {
  Union_Find uf;
  int count = 0;
  uf.Reset(width * height);
  for (int i = 0; i < width * height * 2; ++i) {
//	  cout << all_edges[i].first << " " << all_edges[i].second << endl;
    if (uf.Union(all_edges[i].first, all_edges[i].second)) {
      mst_edges[count] = all_edges[i];
//      cout << "mst: " << mst_edges[count].first << "," << mst_edges[count].second << endl;
      count++;
    }
  }
  return count;
}

// width and height more then 2
void GPU_kruskal_maze_generator(int *maze, int width, int height) {
	srand(time(0));
	int newWidth = (width -1)/2;
	int newHeight = (height -1)/2;
	pair<int, int> all_edges[newWidth * newHeight * 2];
	pair<int, int> mst_edges[newWidth * newHeight * 2];

	// initialize all edges matrix
	pair<int, int> *dev_edges;
	cudaMalloc(&dev_edges, sizeof(pair<int, int>) * newWidth * newHeight * 2);
	cudaMemcpy(dev_edges, all_edges, sizeof(pair<int, int>) * newWidth * newHeight * 2, cudaMemcpyHostToDevice);
	GPU_CreateEdges(dev_edges, newWidth, newHeight);

	// copy device edge matrix in host matrix
	cudaMemcpy(all_edges, dev_edges, sizeof(pair<int, int>) * newWidth * newHeight * 2, cudaMemcpyDeviceToHost);

	GPU_ShuffleMatrix(all_edges, newWidth, newHeight);

	int count = GPU_Kruskal(all_edges, mst_edges, newWidth, newHeight);
//	cout << count << endl;
	// initialize mst edges matrix
	pair<int, int> *dev_mst;
	cudaMalloc(&dev_mst, sizeof(pair<int, int>) * count);
	cudaMemcpy(dev_mst, mst_edges, sizeof(pair<int, int>) * count, cudaMemcpyHostToDevice);

	// initialize points matrix
	pair<int, int> points[count];
	pair<int, int> *dev_points;
	cudaMalloc(&dev_points, sizeof(pair<int, int>) * count);

	// initialize nodes
	pair<int, int> *node_a;
	pair<int, int> *node_b;
	pair<int, int> *edge;
	cudaMalloc(&node_a, sizeof(pair<int, int>) * count);
	cudaMalloc(&node_b, sizeof(pair<int, int>) * count);
	cudaMalloc(&edge, sizeof(pair<int, int>) * count);

	GPU_ConvertToPaintedPoint<<<1, count>>>(dev_mst, dev_points, newWidth, count, node_a, node_b, edge);

	// copy device edge matrix in host matrix
	cudaMemcpy(points, dev_points, sizeof(pair<int, int>) * count, cudaMemcpyDeviceToHost);

	sort(&points[0], &points[count]);

	int size = ((newWidth * 4) + 1);
	int largeMaze[size * size];
	GPU_GenerateMaze(points, largeMaze, newWidth, newHeight, size, count);
	FromMazeToGrid(largeMaze, maze, width, size);
}
