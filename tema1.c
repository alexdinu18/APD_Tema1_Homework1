/*
	Dinu Marian Alexandru
	Grupa 334 CC
	Tema 1
	APD
*/
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

void plan(int H, int W, int matrix[H][W], int N);
void toroid(int H, int W, int matrix[H][W], int N);

int main(int argc, char** argv)
{
	int i, j, W_new = 0, H_new = 0, W, H, W_map, H_map, N, nr_threads, temp;
	char type, line[80];
	FILE *in;
	FILE *out;
	
	if(argc != 5) {
		printf("Wrong number of parameters!\n");
		exit(1);
	}
	
	nr_threads = atoi(argv[1]);
	omp_set_num_threads(nr_threads);
	N = atoi(argv[2]);
	in = fopen(argv[3], "r");	
	
	if (in != NULL) {
		fgets(line, 80, in);
		sscanf(line, "%c %d %d %d %d", &type, &W_map, &H_map, &W, &H);
	}
	else {
		printf("Error opening file!\n");
		exit(1);
	}
	
	int matrix[H][W];
	
	// fill matrix with zeros
	#pragma omp parallel for collapse(2)
	for (i = 0; i < H; i++)
		for (j = 0; j < W; j++)
			matrix[i][j] = 0;
	
	// read matrix from file
	if (H < H_map) {
		for (i = 0; i < H; i++) {
			if (W < W_map) {
				for (j = 0; j < W; j++)	{
						fscanf(in, "%d", &matrix[i][j]);
				}
				for (j = W; j < W_map; j++)
					fscanf(in, "%d", &temp);
			}
			else
				for (j = 0; j < W_map; j++)	{
					fscanf(in, "%d", &matrix[i][j]);
				}
		}
	}
	else {
		for (i = 0; i < H_map; i++) 
			for (j = 0; j < W_map; j++)	
				fscanf(in, "%d", &matrix[i][j]);
	}
	
	
	if (W < W_map)
		W_map = W;
	if (H < H_map)
	    H_map = H;
	
	if (type == 'P')
		plan(H, W, matrix, N);
	else if (type == 'T')
		toroid(H, W, matrix, N);

	// crop matrix
	#pragma omp parallel for collapse(2)
	for (i = 0; i < H; i++) {
		for (j = 0; j < W; j++) { 
			if (matrix[i][j] && W_new < j)
				W_new = j;
			if (matrix[i][j] && H_new < i)
				H_new = i;
		}
	}
	
	// write matrix to file
	out = fopen(argv[4], "w");
	fprintf(out, "%c %d %d %d %d\n", type, W_new + 1, H_new + 1, W, H);
	for (i = 0; i <= H_new; i++) {
		for (j = 0; j <= W_new; j++) {
			fprintf(out, "%d ", matrix[i][j]);
		}
		fprintf(out, "\n");
	}
	
	fclose(in);
	fclose(out);
	
	return 0;
}


void plan(int H, int W, int matrix[H][W], int N) {
	int i, j, iterations = N;
	int previous[H][W];
	
	while(iterations) {
		iterations--;
		
		// update previous version of the matrix
		#pragma omp parallel for collapse(2)
		for (i = 0; i < H; i++)
			for (j = 0; j < W; j++)
				previous[i][j] = matrix[i][j];
				
		#pragma omp parallel for collapse(2) private(i, j)		
		for (i = 0; i < H; i++)
			for (j = 0; j < W; j++) {
				int neighbours = 0;
				
				// counting neighbours
				if (i - 1 >= 0 && j - 1 >= 0)
					neighbours += previous[i - 1][j - 1];
				if (i - 1 >= 0)
					neighbours += previous[i - 1][j];
				if (i - 1 >= 0 && j + 1 < W)
					neighbours += previous[i - 1][j + 1];
				if (j - 1 >= 0)
					neighbours += previous[i][j - 1];
				if (j + 1 < W)
					neighbours += previous[i][j + 1];
				if (i + 1 < H && j - 1 >= 0)
					neighbours += previous[i + 1][j - 1];
				if (i + 1 < H)
					neighbours += previous[i + 1][j];
				if (i + 1 < H && j + 1 < W)
					neighbours += previous[i + 1][j + 1];
					
				if (previous[i][j] && (neighbours < 2 || neighbours > 3))
					matrix[i][j] = 0;
				if (!previous[i][j] && neighbours == 3)
					matrix[i][j] = 1;
			}
	}
}

void toroid(int H, int W, int matrix[H][W], int N) {
	int i, j, iterations = N;
	int previous[H][W];
	
	while(iterations) {
		iterations--;
		
		// update previous version of the matrix
		#pragma omp parallel for collapse(2)
		for (i = 0; i < H; i++)
			for (j = 0; j < W; j++) {
				previous[i][j] = matrix[i][j];
			}
		
		#pragma omp parallel for collapse(2) private(i, j)
		for (i = 0; i < H; i++)
			for (j = 0; j < W; j++) {
				int neighbours = 0;
				int i_new_minus, i_new_plus, j_new_minus, j_new_plus;
				
				i_new_minus = i - 1;
				i_new_plus = i + 1;
				j_new_minus = j - 1;
				j_new_plus = j + 1;
				
				// computing new indexes
				if (i_new_minus < 0)
					i_new_minus += H;
				if (i + 1 >= H)
					i_new_plus -= H;
				if (j - 1 < 0)
					j_new_minus += W;
				if (j + 1 >= W)
					j_new_plus -= W;
				
				neighbours += previous[i_new_minus][j_new_minus];
				neighbours += previous[i_new_minus][j];
				neighbours += previous[i_new_minus][j_new_plus];
				neighbours += previous[i][j_new_minus];
				neighbours += previous[i][j_new_plus];
				neighbours += previous[i_new_plus][j_new_minus];
				neighbours += previous[i_new_plus][j];
				neighbours += previous[i_new_plus][j_new_plus];
								
				if (previous[i][j] && (neighbours < 2 || neighbours > 3))
					matrix[i][j] = 0;
				if (!previous[i][j] && neighbours == 3)
					matrix[i][j] = 1;
			}
	}
}
