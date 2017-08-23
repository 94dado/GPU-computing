#ifndef _union_
#define _union_ 1

typedef struct intStruct {
    int val;
} IntStruct;

typedef struct vectorInt {
    IntStruct **data;
    int size;
    int count;
} VectorInt;



void vectorInt_init(VectorInt *v);
void vectorInt_init(VectorInt *v);
int vectorInt_count(VectorInt *v);
void vectorInt_add(VectorInt *v, IntStruct *e);
void vectorInt_set(VectorInt *v, int index, IntStruct *e);
int vectorInt_get(VectorInt *v, int index);
void vectorInt_delete(VectorInt *v, int index);
void vectorInt_free(VectorInt *v);
int find_vectorInt_index(VectorInt *v, IntStruct *elem);


typedef struct union_find {
    VectorInt *_sets;
    int _number_of_sets;
} UnionFind;


void union_init(UnionFind *u, int size);
void reset (UnionFind *u, int number_of_sets);
int find(UnionFind *u, int element);
int union_set(UnionFind *u, int set_a, int set_b);
int number_set(UnionFind *u);

#endif
