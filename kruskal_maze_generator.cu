// C libraries
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "./Header/common.h"

int WIDTH = 10;
int HEIGHT = 10;

// Please, note that giving random weights and sorting a list of edges is
// the same as just shuffling it. The later is O(E) against O(E log E) of
// the former, though.
// Also, note that if we are making a maze using a four-sided grid,
// E ~= H * W * 4.
// Another important note is that the grid to be drawn is twice the size of
// the "actual" grid. All positions of the type (even, even) are vertices,
// while the remaining ones can be edges (or the lack of edges).

// Shuffle a vector
void shuffle_vector (Vector *vector) {
    for (int i = 0; i < vector_count(vector); ++i) {
        int x = rand() % vector_count(vector);
        IntPair *tmp1 = vector->data[i];
        vector->data[i] = vector->data[x];
        vector->data[x] = tmp1;
    }
}

// We are going to represent the two-dimensinal array as a one-dimensional
// vector. This will allow us to represent the edges in a shorter way.
// In this case, manual calculations are necessary.
int Pos2Idx (int row, int column) {
    return row * WIDTH + column;
}

IntPair *Idx2Pos(int idx) {
    int row = idx / WIDTH;
    int column = idx % WIDTH;
    IntPair *p;
    pair_init(p, row, column);
    return p;
}

// std::pair < node A, node B >
void CreateEdges(Vector *edges) {
    for (int row = 1; row < HEIGHT; ++row) {
        for (int column = 1; column < WIDTH; ++column) {
            IntPair *p;
            pair_init(p, Pos2Idx(row, column), Pos2Idx(row, column - 1));
            vector_add(edges, p);
            IntPair *q;
            pair_init(q, Pos2Idx(row, column), Pos2Idx(row - 1, column));
            vector_add(edges, q);
        }
    }

    for (int column = 1; column < WIDTH; ++column) {
        IntPair *p;
        pair_init(p, Pos2Idx(0, column), Pos2Idx(0, column - 1));
        vector_add(edges, p);
    }

    for (int row = 1; row < HEIGHT; ++row) {
        IntPair *p;
        pair_init(p, Pos2Idx(row, 0), Pos2Idx(row - 1, 0));
        vector_add(edges, p);
    }
}

void Kruskal(Vector *all_edges, Vector *mst_edges) {
    UnionFind *uf;
    reset(uf, WIDTH * HEIGHT);
    for (int i = 0; i < vector_count(all_edges); ++i) {
        if (union_set(uf, all_edges->data[i]->a, all_edges->data[i]->b)) {
            vector_add(mst_edges, all_edges->data[i]);
        }
    }
}

// The code below is for the drawing part.

// Convert each edge to the postion in the matrix that must be drawn.
void ConvertToPaintedPoint(Vector *edges, Vector *painted_points) {
    for (int i = 0; i < vector_count(edges); ++i) {
        IntPair *node_a = Idx2Pos(edges->data[i]->a);
        IntPair *node_b = Idx2Pos(edges->data[i]->b);

        // Convert the vertices positions to their drawn version.
        node_a->a *= 2;
        node_a->b *= 2;
        node_b->a *= 2;
        node_b->b *= 2;

        // Calculate the position of their edge.
        IntPair *edge;
        edge->a = (node_a->a + node_b->a) / 2;
        edge->b = (node_a->b + node_b->b) / 2;

        vector_add(painted_points, edge);
    }

    // Sort the edges to draw them.
    bubble_sort(painted_points);
}

void Draw(Vector *painted_points) {
    int cpp = 0; // cur painted point
    printf("*   *\n*   *\n");
    printf("*   ");
    for (int column = 0; column < (WIDTH * 2) - 1; ++ column) {
        putchar('*'); putchar(' ');
    }
    putchar('\n');
    for (int row = 0; row < (HEIGHT * 2) - 1; ++row) {
        putchar('*');
        putchar(' ');
        for (int column = 0; column < (WIDTH * 2) - 1; ++ column) {
            // If it's an vertex position
            if ((row & 1) == 0 && (column & 1) == 0) {
                putchar(' ');
                putchar(' ');
            }
            else {
                if (cpp < vector_count(painted_points)
                    && painted_points->data[cpp]->a == row
                    && painted_points->data[cpp]->b == column) {
                    putchar(' ');
                    cpp++;
                }
                else {
                    putchar('*');
                }
            putchar(' ');
            }
        }
        putchar('*');
        putchar('\n');
    }
    for (int column = 0; column < (WIDTH * 2) - 1; ++ column) {
        putchar('*'); putchar(' ');
    }
    printf("  *\n");
    for (int column = 0; column < (WIDTH * 2) - 2; ++ column) {
        putchar(' '); putchar(' ');
    }
    printf("*   *\n");
    for (int column = 0; column < (WIDTH * 2) - 2; ++ column) {
        putchar(' '); putchar(' ');
    }
    printf("*   *\n");
}

int main(int argc, char **argv) {
    Vector *all_edges;
    vector_init(all_edges);
    Vector *mst_edges;
    vector_init(mst_edges);
    Vector *points;
    vector_init(points);

    srand(time(0));

    CreateEdges(all_edges);
    shuffle_vector(all_edges);
    Kruskal(all_edges, mst_edges);
    ConvertToPaintedPoint(mst_edges, points);
    Draw(points);
    return 0;
}
