#ifndef _vector_
#define _vector_ 1

typedef struct intPair {
    int a;
    int b;
} IntPair;


void pair_init(IntPair *p, int x, int y);

typedef struct vector {
    IntPair **data;
    int size;
    int count;
} Vector;

void vector_init(Vector *v);
int vector_count(Vector *v);
void vector_add(Vector *v, IntPair *e);
void vector_set(Vector *v, int index, IntPair *e);
void *vector_get(Vector *v, int index);
void vector_delete(Vector *v, int index);
void vector_free(Vector *v);

#endif
