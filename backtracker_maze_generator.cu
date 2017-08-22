#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"


/*  Carve the maze starting at x, y. */
void CarveMaze(int *maze, int width, int height, int x, int y) {

   int x1, y1;
   int x2, y2;
   int dx, dy;
   int dir, count;

   dir = rand() % 4;
   count = 0;
   while(count < 4) {
      dx = 0; dy = 0;
      switch(dir) {
      case 0:  dx = 1;  break;
      case 1:  dy = 1;  break;
      case 2:  dx = -1; break;
      default: dy = -1; break;
      }
      x1 = x + dx;
      y1 = y + dy;
      x2 = x1 + dx;
      y2 = y1 + dy;
      if(   x2 > 0 && x2 < width && y2 > 0 && y2 < height
         && maze[y1 * width + x1] == WALL && maze[y2 * width + x2] == WALL) {
         maze[y1 * width + x1] = OPEN;
         maze[y2 * width + x2] = OPEN;
         x = x2; y = y2;
         dir = rand() % 4;
         count = 0;
      } else {
         dir = (dir + 1) % 4;
         count += 1;
      }
   }

}

/* Generate maze in matrix maze with size width, height. */
void CPU_backtracker_maze_generator(int *maze, int width, int height) {
   int x, y;

   /* Initialize the maze. */
   for(x = 0; x < width * height; x++) {
      maze[x] = WALL;
   }
   maze[1 * width + 1] = OPEN;

   /* Carve the maze. */
   for(y = 1; y < height; y += 2) {
      for(x = 1; x < width; x += 2) {
         CarveMaze(maze, width, height, x, y);
      }
   }

   /* Set up the entry and exit. */
   maze[0 * width + 1] = OPEN;
   maze[(height - 1) * width + (width - 2)] = OPEN;
}

//int main(){
//	int maze[100];
//	CPU_backtracker_maze_generator(maze,10,10);
//	print_maze(maze,10,10);
//	return 0;
//}


/* Solve the maze.
void SolveMaze(char *maze, int width, int height) {

   int dir, count;
   int x, y;
   int dx, dy;
   int forward;

   // Remove the entry and exit.
   maze[0 * width + 1] = 1;
   maze[(height - 1) * width + (width - 2)] = 1;

   forward = 1;
   dir = 0;
   count = 0;
   x = 1;
   y = 1;
   while(x != width - 2 || y != height - 2) {
      dx = 0; dy = 0;
      switch(dir) {
      case 0:  dx = 1;  break;
      case 1:  dy = 1;  break;
      case 2:  dx = -1; break;
      default: dy = -1; break;
      }
      if(   (forward  && maze[(y + dy) * width + (x + dx)] == 0)
         || (!forward && maze[(y + dy) * width + (x + dx)] == 2)) {
         maze[y * width + x] = forward ? 2 : 3;
         x += dx;
         y += dy;
         forward = 1;
         count = 0;
         dir = 0;
      } else {
         dir = (dir + 1) % 4;
         count += 1;
         if(count > 3) {
            forward = 0;
            count = 0;
         }
      }
   }

   // Replace the entry and exit.
   maze[(height - 2) * width + (width - 2)] = 2;
   maze[(height - 1) * width + (width - 2)] = 2;
}
*/
