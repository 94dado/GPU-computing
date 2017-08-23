#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct intStruct {
    int val;
} IntStruct;

typedef struct vectorInt {
    IntStruct **data;
    int size;
    int count;
} VectorInt;

void vectorInt_init(VectorInt *v) {
    v->size = 0;
    v->count = 0;
}

int vectorInt_count(VectorInt *v) {
    return v->count;
}

void vectorInt_add(VectorInt *v, IntStruct *e) {
    if (v->size == 0) {
        v->size = 10;
        v->data = (IntStruct**)malloc(sizeof(IntStruct*) * v->size);
        memset(v->data, '\0', sizeof(IntStruct*) * v->size);
    }

    if (v->size == v->count) {
        v->size *= 2;
        v->data = (IntStruct**)realloc(v->data, sizeof(IntStruct*) * v->size);
    }

    v->data[v->count]->val = e->val;
    v->count++;
}

void vectorInt_set(VectorInt *v, int index, IntStruct *e) {
    if (index >= v->count) {
        return;
    }

    v->data[index] = e;
}

int vectorInt_get(VectorInt *v, int index) {
    if (index >= v->count) {
        return -1;
    }

    return v->data[index]->val;
}

void vectorInt_delete(VectorInt *v, int index) {
    if (index >= v->count) {
        return;
    }

    for (int i = index, j = index; i < v->count; i++) {
        v->data[j] = v->data[i];
        j++;
    }

    v->count--;
}

void vectorInt_free(VectorInt *v) {
    free(v->data);
}

int find_vectorInt_index(VectorInt *v, IntStruct *elem) {
	int found = 0;
	for(int i=0; i < v->count; ++i) {
        if(v->data[i] == elem) {
            return i;
        }
    }
    return -1;
}

// A Union-Find structure is needed by Kruskal's algorithm.
// Basicly, it efficiently manages disjoint sets, making it possible to
// join them and check if two elements belong to the same set.
// find(A) = find(B) IFF A and B belong to the same set.
typedef struct union_find {
    VectorInt *_sets;
    int _number_of_sets;
} UnionFind;

void union_init(UnionFind *u, int size) {
    u->_number_of_sets = 0;
    vectorInt_init(u->_sets);
}

void reset (UnionFind *u, int number_of_sets) {
    vectorInt_init(u->_sets);
    u->_number_of_sets = number_of_sets;
}

int find(UnionFind *u, int element) {
    IntStruct *element_struct;
    element_struct->val = element;
    if (vectorInt_get(u->_sets, find_vectorInt_index(u->_sets, element_struct)) < 0) return element;
    IntStruct *el_struct;
    el_struct->val = find(u, vectorInt_get(u->_sets, find_vectorInt_index(u->_sets, element_struct)));
    vectorInt_set(u->_sets, find_vectorInt_index(u->_sets, element_struct), el_struct);
    return (vectorInt_get(u->_sets, find_vectorInt_index(u->_sets, element_struct)));
}

int union_set(UnionFind *u, int set_a, int set_b) {
    int root_a = find(u, set_a), root_b = find(u, set_b);
    if (root_a == root_b) return 0;
    IntStruct *root_a_struct;
    root_a_struct->val = root_a;
    IntStruct *root_b_struct;
    IntStruct *sum_struct;
    root_b_struct->val = root_b;
    sum_struct->val = vectorInt_get(u->_sets, find_vectorInt_index(u->_sets, root_a_struct) + vectorInt_get(u->_sets, find_vectorInt_index(u->_sets, root_b_struct)));
    vectorInt_set(u->_sets, find_vectorInt_index(u->_sets, root_a_struct), sum_struct);
    vectorInt_set(u->_sets, find_vectorInt_index(u->_sets, root_b_struct), root_a_struct);
    u->_number_of_sets--;
    return 1;
}

int number_set(UnionFind *u) {
    return u->_number_of_sets;
}