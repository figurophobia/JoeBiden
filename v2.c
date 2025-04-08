#include "counter.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void initializeSystem(float*** a, float** b, float** x, int N);
void Jacobi(float** a, float* b, float* x, int N, float tol, int max_iter);
int min(int a, int b){
    return a < b ? a : b;
}
#define BLOQUE_SIZE 16

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

    //Abrimos un archivo para guardar resultados:
    FILE *f = fopen("resultados.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    start_counter();

    for (int iter = 0; iter < max_iter; iter++) {
        norm2 = 0.0;

        // Recorrer la matriz por bloques
        for (int bi = 0; bi < N; bi += BLOQUE_SIZE) {
            int bimin = (bi + BLOQUE_SIZE < N) ? bi + BLOQUE_SIZE : N;

            for (int bj = 0; bj < N; bj += BLOQUE_SIZE) {
                int bjmin = (bj + BLOQUE_SIZE < N) ? bj + BLOQUE_SIZE : N;

                // Calcular x_new para los elementos del bloque
                for (int i = bi; i < bimin; i++) {
                    float sigma = 0.0;

                    for (int j = bj; j < bjmin; j++) {
                        if (i != j) {
                            sigma += a[i][j] * x[j];
                        }
                    }

                    x_new[i] = (b[i] - sigma) / a[i][i]; //Revisar esta linea, sigma quiza está mal calculado
                    
                }
            }
        }

        // Actualizar x y calcular la norma fuera de los bucles de bloques
        for (int i = 0; i < N; i++) {
            norm2 += pow(x_new[i] - x[i], 2);
            x[i] = x_new[i];
            
        }

        // Comprobar convergencia
        if (sqrt(norm2) < tol) {
            double cycles = get_counter();
            printf("Iteraciones: %d\n", iter);
            printf("Norma: %.15e\n", norm2);
            printf("Ciclos: %.0f\n", cycles);
            for (int i = 0; i < N; i++) {
                fprintf(f, "x[%d] = %.15e\n", i, x[i]);
            }
            break;
        }
    }

    free(x_new);
}
