#include "counter.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>
#include <stdint.h>
#if __STDC_VERSION__ >= 201112L
    #include <stdalign.h>
#endif

#define BYTES (8 * sizeof(float))  // 32 bytes

static inline __m256 load256(const float *p) { // funcion mas segura que comprueba si la mem esta alineada
    if (((uintptr_t)p % BYTES) == 0)
        return _mm256_load_ps(p);
    else
        return _mm256_loadu_ps(p);
}

void initializeSystem(float*** a, float** b, float** x, int N);
void Jacobi(float** a, float* b, float* x, int N, float tol, int max_iter);
void save_csv(char *name, int N, int nhilos, int nciclos) {
    FILE *f = fopen(name, "a");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    fprintf(f, "%d,%d,%d\n", N, nhilos, nciclos);
    fclose(f);
}

int main(int argc, char** argv) {
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

    float** a = NULL;
    if (posix_memalign((void**)&a, BYTES, N * sizeof(float*)) != 0) {
        fprintf(stderr, "Error en la asignación de memoria para a\n");
        return 1;
    }
    for (int i = 0; i < N; i++) {
        if (posix_memalign((void**)&a[i], BYTES, N * sizeof(float)) != 0) {
            fprintf(stderr, "Error en la asignación de memoria para a[%d]\n", i);
            return 1;
        }
    }

    float* b = (float*)malloc(N * sizeof(float));
    if (b == NULL) {
        fprintf(stderr, "Error en la asignación de memoria para b\n");
        return 1;
    }

    float* x = NULL;
    if (posix_memalign((void**)&x, BYTES, N * sizeof(float)) != 0) {
        fprintf(stderr, "Error en la asignación de memoria para x\n");
        return 1;
    }

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

void Jacobi(float** a, float* b, float* x, int N, float tol, int max_iter) {
    double cycles = 0.0;

    float* x_new = (float*)malloc(N * sizeof(float));
    if (x_new == NULL) {
        fprintf(stderr, "Error en la asignación de memoria para x_new\n");
        return;
    }
    
    double norm2 = 0.0;
    int iter = 0;

    start_counter();

    for (; iter < max_iter; iter++) {
        norm2 = 0.0;
        for (int i = 0; i < N; i++) {
            float sigma = 0.0f;
            int j = 0;

            for (; j <= i - 8; j += 8) {

                __m256 va = _mm256_load_ps(&a[i][j]);
                __m256 vx = _mm256_load_ps(&x[j]);
                __m256 vmul = _mm256_mul_ps(va, vx);
                // problema de alineación
                __attribute__((aligned(32))) float temp[8];
                _mm256_store_ps(temp, vmul);
                sigma += temp[0] + temp[1] + temp[2] + temp[3]
                       + temp[4] + temp[5] + temp[6] + temp[7];
            }

            for (; j < i; j++) {
                sigma += a[i][j] * x[j];
            }

            // bucle sale cuando j==i
            j = i + 1;
            // procesa escalarmente hasta que la dirección de a[i][j] esté alineada
            while (j < N && (((uintptr_t)&a[i][j]) % BYTES) != 0) {
                sigma += a[i][j] * x[j];
                j++;
            }

            for (; j <= N - 8; j += 8) {
                __m256 va = load256(&a[i][j]);
                __m256 vx = load256(&x[j]);
                __m256 vmul = _mm256_mul_ps(va, vx);
                __attribute__((aligned(32))) float temp[8];
                _mm256_store_ps(temp, vmul);
                sigma += temp[0] + temp[1] + temp[2] + temp[3]
                       + temp[4] + temp[5] + temp[6] + temp[7];
            }

            for (; j < N; j++) {
                sigma += a[i][j] * x[j];
            }

            x_new[i] = (b[i] - sigma) / a[i][i];
            float diff = x_new[i] - x[i];
            norm2 += diff * diff;
        }
        
        printf("Norma: %.15e\n", norm2);
        memcpy(x, x_new, N * sizeof(float));
        
        if (norm2 < tol * tol) {
            break;
        }
    }

    cycles = get_counter();
    printf("Iteraciones: %d\n", iter);
    printf("Norma: %.15e\n", norm2);
    printf("Ciclos: %.0f\n", cycles);
    printf("Tiempo: %f segundos\n", cycles / 2.5e9);

    save_csv("v3.csv", N, 0, cycles);
    free(x_new);
}
