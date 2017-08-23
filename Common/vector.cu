#include <stdlib.h>
#include <string.h>
#include "../Header/vector.h"

// define a pair of int

void pair_init(IntPair *p, int x, int y) {
    p->a = x;
    p->b = y;
}

void vector_init(Vector *v) {
	v->data = NULL;
    v->size = 0;
    v->count = 0;
}

int vector_count(Vector *v) {
    return v->count;
}

void vector_add(Vector *v, IntPair *e) {
    if (v->size == 0) {
        v->size = 10;
        v->data = (IntPair**)malloc(sizeof(IntPair*) * v->size);
        memset(v->data, '\0', sizeof(IntPair*) * v->size);
    }

    if (v->size == v->count) {
        v->size *= 2;
        v->data = (IntPair**)realloc(v->data, sizeof(IntPair*) * v->size);
    }

    v->data[v->count] = e;
    v->count++;
}

void vector_set(Vector *v, int index, IntPair *e) {
    if (index >= v->count) {
        return;
    }

    v->data[index] = e;
}

void *vector_get(Vector *v, int index) {
    if (index >= v->count) {
        return NULL;
    }

    return v->data[index];
}

void vector_delete(Vector *v, int index) {
    if (index >= v->count) {
        return;
    }

    for (int i = index, j = index; i < v->count; i++) {
        v->data[j] = v->data[i];
        j++;
    }

    v->count--;
}

void vector_free(Vector *v) {
    free(v->data);
}
