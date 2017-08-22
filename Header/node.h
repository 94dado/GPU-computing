#include <stdlib.h>
#include <stdio.h>

typedef struct node {
    //Node position - little waste of memory, but it allows faster generation
    int x, y;
    //Pointer to parent node
    void *parent;
    //Character to be displayed
    int isSpace;
    //Directions that still haven't been explored
	char dirs;
} Node;
