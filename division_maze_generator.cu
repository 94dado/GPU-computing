#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "time.h"
#include "Header/common.h"

#define HORIZONTAL 1
#define VERTICAL 2

#define MAZE_RESOLUTION 2

#define PERPENDICULAR_HORIZONTAL 1
#define PERPENDICULAR_VERTICAL 2

int choose_orientation(int width, int height){
	if(width < height)	return HORIZONTAL;
	else if(height < width) return VERTICAL;
	else return rand()%2 + 1;
}

void recursive_divide(int *maze, int x, int y, int width, int height, int orientation){
	cout << "------------------" << endl << "recursion started" << endl;
	cout << "x: " << x << ", y: " << y << ", width: " << width << ", height: " << height << ", orientation: " << orientation << endl;
	int fromX, fromY,holeX,holeY,directionX,directionY,wall_lenght, perpendicular;
	int newX, newY, newWidth, newHeight;
	//check if I don't have to make this iteration
	if(width < MAZE_RESOLUTION || height < MAZE_RESOLUTION){
		cout << "nope!" << endl;
		return;
	}
	if(orientation == HORIZONTAL){
		//is horizontal
		fromX = x;
		if(height - 2 <= 0){
			fromY = y;
		}else{
			fromY = y + rand()%(height-2);
		}
		holeX = fromX + rand()%width;
		holeY = fromY;
		directionX = 1;
		directionY = 0;
		wall_lenght = width;
		perpendicular = PERPENDICULAR_HORIZONTAL;

	}else{
		//is vertical
		if(width - 2 <= 0){
			fromX = x;
		}else{
			fromX = x + rand()%(width-2);
		}
		fromY = y;
		holeX = fromX;
		holeY = fromY + rand()%height;
		directionX = 0;
		directionY = 1;
		wall_lenght = height;
		perpendicular = PERPENDICULAR_VERTICAL;
	}
	cout << "data calculated:" << endl;
	cout << "fromX: " << fromX << ", fromY: " << fromY << ", holeX: " << holeX
			<< ", holeY: " << holeY << ", directionX: " << directionX << ", wall_length:"
			<< wall_lenght << ", perpendicular: " << perpendicular << endl;
	//create the wall with an hole
	for(int i = 0; i < wall_lenght; i++){
		if(fromX != holeX || fromY != holeY)
		maze[width * fromY + fromX] = WALL;
		fromX += directionX;
		fromY += directionY;
	}
	//first recursive call
	newX = x;
	newY = y;
	if(orientation == HORIZONTAL){
		newWidth = width;
		newHeight = fromY - y + 1;
	}else{
		newWidth = fromX - x + 1;
		newHeight = height;
	}
	recursive_divide(maze,newX, newY, newWidth, newHeight, choose_orientation(newWidth, newHeight));

	//second recursive call
	if(orientation == HORIZONTAL){
		newX = x;
		newY = fromY + 1;
		newWidth = width;
		newHeight = y + height - fromY - 1;
	}else{
		newX = fromX + 1;
		newY = y;
		newWidth = x + width - fromX - 1;
		newHeight = height;
	}
	recursive_divide(maze,newX, newY, newWidth, newHeight, choose_orientation(newWidth, newHeight));
}

void CPU_division_maze_generator(int *maze, int width, int height){
	//set random seed
	srand(time(NULL));
	//first, set everything to OPEN
	FillOpen(maze, width* height);
	//start with the algorithm
	recursive_divide(maze, 0, 0, width, height, choose_orientation(width, height));
}
