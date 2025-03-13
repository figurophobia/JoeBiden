#include "counter.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void initializeSystem(float*** a, float** b, float** x, int N);
void Jacobi(float** a, float* b, float* x, int N, float tol, int max_iter);
int isDiagonalDominant(float** a, int N);

int main(int argc, char** argv){
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <tamaño de matriz> [número de hilos - ignorado]\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        fprintf(stderr, "El tamaño de la matriz debe ser un entero positivo.\n");
        return 1;
    }

    float tol = 1e-8;
    int max_iter = 20000;

    float** a = (float**)malloc(N * sizeof(float*));
    for (int i = 0; i < N; i++) {
        a[i] = (float*)malloc(N * sizeof(float));
    }
    
    float* b = (float*)malloc(N * sizeof(float));
    float* x = (float*)malloc(N * sizeof(float));
    
    if (!a || !b || !x) {
        fprintf(stderr, "Error en la asignación de memoria\n");
        return 1;
    }

    initializeSystem(&a, &b, &x, N);

    // Verificar si la matriz es diagonal dominante
    if (!isDiagonalDominant(a, N)) {
        fprintf(stderr, "La matriz no es diagonal dominante.\n");
        return 1;
    }

    Jacobi(a, b, x, N, tol, max_iter);

    for (int i = 0; i < N; i++) {
        free(a[i]);
    }
    free(a);
    free(b);
    free(x);

    return 0;
}

void initializeSystem(float*** a, float** b, float** x, int N) {
    srand(N);
    for (int i = 0; i < N; i++) {
        float row_sum = 0.0;
        for (int j = 0; j < N; j++) {
            (*a)[i][j] = (float)rand() / RAND_MAX;
            row_sum += (*a)[i][j];
        }
        (*a)[i][i] += row_sum;
    }
    
    for (int i = 0; i < N; i++) {
        (*b)[i] = (float)rand() / RAND_MAX;
        (*x)[i] = 0.0;
    }
}

void Jacobi(float** a, float* b, float* x, int N, float tol, int max_iter){
    float* x_new = (float*)malloc(N * sizeof(float));
    if (!x_new) {
        fprintf(stderr, "Error en la asignación de memoria\n");
        return;
    }
    float norm2 = 0.0;

    for (int iter = 0; iter < max_iter; iter++){
        norm2 = 0.0;

        for(int i = 0; i < N; i++){
            float sigma = 0.0;
            for(int j = 0; j < N; j++){
                if(i != j){
                    sigma += a[i][j] * x[j];
                }
            }
            x_new[i] = (b[i] - sigma) / a[i][i];
            norm2 += pow(x_new[i] - x[i], 2);
        }
        
        for(int i = 0; i < N; i++){
            x[i] = x_new[i];
        }
        
        if (sqrt(norm2) < tol){
            printf("Iteraciones: %d\n", iter);
            printf("Norma: %.15f\n", sqrt(norm2));
            break;
        }
    }
    
    free(x_new);
}

int isDiagonalDominant(float** a, int N) {
    for (int i = 0; i < N; i++) {
        float row_sum = 0.0;
        for (int j = 0; j < N; j++) {
            if (i != j) {
                row_sum += fabs(a[i][j]);
            }
        }

        if (fabs(a[i][i]) < row_sum) {
            return 0;  // No es diagonal dominante
        }
    }
    return 1;  // Es diagonal dominante
}
