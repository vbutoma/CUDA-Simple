#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "addition.cuh"
#include <stdlib.h>

using namespace std;

const int N = 10000, M = 10000;
int A[N][M];
int B[N][M];
int C[N][M];
int D[N][M];

void init(){
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++){
			A[i][j] = rand() % 256;
			B[i][j] = rand() % 256;
			C[i][j] = 0;
			D[i][j] = 0;
		}
}

int main(int argc, char **argv) {
	init();
	additionCPU((int *)A, (int * )B, (int * )C, N, M);
	// for (int i = 0; i < N; i++){
	// 	for (int j = 0; j < M; j++)
	// 		cout << A[i][j] << " ";
	// 	cout << endl;
	// }
	// cout << endl;
	// for (int i = 0; i < N; i++){
	// 	for (int j = 0; j < M; j++)
	// 		cout << B[i][j] << " ";
	// 	cout << endl;
	// }
	// cout << endl;
	// for (int i = 0; i < N; i++){
	// 	for (int j = 0; j < M; j++)
	// 		cout << C[i][j] << " ";
	// 	cout << endl;
	// }
	// cout << endl;
	additionGPU((int * )A, (int * )B, (int *)D, N, M);
	// for (int i = 0; i < N; i++){
	// 	for (int j = 0; j < M; j++)
	// 		cout << D[i][j] << " ";
	// 	cout << endl;
	// }

	return 0;
}
