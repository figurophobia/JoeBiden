#include "counter.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>
#include <stdint.h>

#define BYTES (8 * sizeof(float))

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

    //float** a = (float**)malloc(N * sizeof(float*));
    float** a = NULL; 
    if(posix_memalign((void**)&a, BYTES, N * sizeof(float*)) != 0) {
        fprintf(stderr, "Error en la asignación de memoria\n");
        return 1;
    }


    for (int i = 0; i < N; i++) {
        //a[i] = (float*)malloc(N * sizeof(float));
        if(posix_memalign((void**)&a[i], BYTES, N * sizeof(float)) != 0) {
            fprintf(stderr, "Error en la asignación de memoria\n");
            return 1;
        }
    }
    
    float* b = (float*)malloc(N * sizeof(float));
    //float* x = (float*)malloc(N * sizeof(float));
    float* x = NULL; 
    if(posix_memalign((void**)&x, BYTES, N * sizeof(float)) != 0) {
        fprintf(stderr, "Error en la asignación de memoria\n");
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

void Jacobi(float** a, float* b, float* x, int N, float tol, int max_iter){
    double cycles = 0.0;
    //float* x_new = (float*)malloc(N * sizeof(float));
    float* x_new = (float*)malloc(N * sizeof(float));
    
    double norm2 = 0.0;

    start_counter();

    int iter = 0;

    for (; iter < max_iter; iter++){
        norm2 = 0.0;
        for(int i = 0; i < N; i++){
            float sigma = 0.0;
            int j = 0;
            // si i esta en el bloque de 8 sale del bucle
            for (; j <= i - 8; j += 8) {
                // 8 floats de a[i][j] y x[j] (se requiere que estén alineados a 32 bytes)
                __m256 va = _mm256_load_ps(&a[i][j]);
                __m256 vx = _mm256_load_ps(&x[j]);
                __m256 vmul = _mm256_mul_ps(va, vx); // multiplicacion

                float temp[8];
                _mm256_store_ps(temp, vmul);
                sigma += temp[0] + temp[1] + temp[2] + temp[3] +
                         temp[4] + temp[5] + temp[6] + temp[7];
            }

            for (; j < i; j++){
                sigma += a[i][j] * x[j];
            }

            j = i + 1; // el bucle anterior sale cuando j == i

            // procesar escalarmente hasta un indice alineado
            while (j < N && (((uintptr_t)&a[i][j]) % BYTES) != 0) {
                sigma += a[i][j] * x[j];
                j++;
            }

            for (; j <= N - 8; j += 8) {
                __m256 va = _mm256_load_ps(&a[i][j]);
                __m256 vx = _mm256_load_ps(&x[j]);
                __m256 vmul = _mm256_mul_ps(va, vx);
                float temp[8];
                _mm256_store_ps(temp, vmul);
                sigma += temp[0] + temp[1] + temp[2] + temp[3] +
                         temp[4] + temp[5] + temp[6] + temp[7];
            }

            for (; j < N; j++){ // valores restantes
                sigma += a[i][j] * x[j];
            }

            x_new[i] = (b[i] - sigma) / a[i][i];
            
            double diff = x_new[i] - x[i];
            norm2 += pow(diff, 2);
        }
        
        printf("Norma: %.15e\n", norm2);

        /*for(int i = 0; i < N; i++) {
            x[i] = x_new[i];
        }*/
       memcpy(x, x_new, N * sizeof(float)); // Copia de x_new a x
        
        if (norm2 < tol * tol){
            break;
        }
    }

    cycles = get_counter();
    printf("Iteraciones: %d\n", iter);
    printf("Norma: %.15e\n", norm2);
    printf("Ciclos: %.0f\n", cycles);
    printf("Tiempo: %f\n", cycles / 2.5e9); // Asumiendo un reloj de 2.5 GHz

    save_csv("v1.csv", N, 0, cycles); // Guardar resultados en CSV
    
    free(x_new);
}
