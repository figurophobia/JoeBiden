#include "counter.h"
#include <math.h>
#include <stdbool.h>
#include <immintrin.h>

double cicles;

#define BLOCK_SIZE 8



void initializeSystem(float*** a, float** b, float** x, int N);
void jacobi(float** a, float* b, float* x, int N, float tol, int max_iter);
int isDiagonalDominant(float** a, int N);
void guardar_solucion(float* x, int N, const char* nombre_archivo);

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

    const float tol = 1e-8;
    const int max_iter = 20000;

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

    

    jacobi(a, b, x, N, tol, max_iter);

    printf("Ciclos: %.0f\n", cicles);

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
            (*a)[i][j] = (float) rand() / RAND_MAX;
            row_sum += (*a)[i][j];
        }
        (*a)[i][i] += row_sum;
    }
    
    for (int i = 0; i < N; i++) {
        (*b)[i] = (float)rand() / RAND_MAX;
        (*x)[i] = 0.0;
    }
}
int min(int a, int b){
    return a < b ? a : b;
}

void jacobi(float** a, float* b, float* x, int N, float tol, int max_iter){

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
        for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
            int bimin = (bi + BLOCK_SIZE < N) ? bi + BLOCK_SIZE : N;

            for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
                int bjmin = (bj + BLOCK_SIZE < N) ? bj + BLOCK_SIZE : N;

                // Calcular x_new para los elementos del bloque
                for (int i = bi; i < bimin; i++) {
                    float sigma = 0.0;
                    int j = bj;
                    // Si el índice i NO está en el rango actual, se puede vectorizar directamente (i!=j,j+1...)
                    if (i < bj || i >= bjmin) {
                        for (; j <= bjmin - 4; j += 4) {
                            __m128 a_vec = _mm_loadu_ps(&a[i][j]);
                            __m128 x_vec = _mm_loadu_ps(x + j);
                            __m128 prod = _mm_mul_ps(a_vec, x_vec);
                            
                            __m128 sum1 = _mm_hadd_ps(prod, prod); // Suma horizontal (1, 2, 3, 4) -> (1+2, 3+4, 1+2, 3+4)
                            __m128 sum2 = _mm_hadd_ps(sum1, sum1); // (3, 7, 3, 7) -> (3+7, 3+7, 3+7, 3+7)
                            sigma += _mm_cvtss_f32(sum2); // Coge el primer elemento que es la suma total
                        }
                    }
                    // Caso en que i se encuentra en el bloque: se vectoriza lo máximo posible
                    else {
                        // Procesar elementos hasta llegar al índice i
                        for (; j < i; j++) {
                            sigma += a[i][j] * x[j]; // Normal, sin vectorización
                        }
                        j++; // Saltarse el elemento i
                        // Vectorizar desde j hasta el final del bloque, pero luego se resta el valor correspondiente a i
                        if (j <= bjmin - 4) {
                            __m128 a_vec = _mm_loadu_ps(&a[i][j]);
                            __m128 x_vec = _mm_loadu_ps(x + j);
                            __m128 prod = _mm_mul_ps(a_vec, x_vec);
                            __m128 sum1 = _mm_hadd_ps(prod, prod); 
                            __m128 sum2 = _mm_hadd_ps(sum1, sum1); 
                            sigma += _mm_cvtss_f32(sum2); 
                        }
                    }
                    // Procesar el resto de elementos que no se pudieron vectorizar (pueden quedar algunos menor a 4)
                    for (; j < bjmin; j++) {
                        if (i != j) {
                            sigma += a[i][j] * x[j];
                        }
                    }
                    x_new[i] = (b[i] - sigma) / a[i][i];
                }
            }
        }


        // Actualizar x y calcular la norma fuera de los bucles de bloques
        for (int i = 0; i < N; i+=4) {
            __m128 x_n = _mm_loadu_ps(x_new + i);
            __m128 x_v = _mm_loadu_ps(x + i);
            __m128 diff = _mm_sub_ps(x_n, x_v);
            __m128 norm = _mm_mul_ps(diff, diff);
            float norm_arr[4];
            _mm_storeu_ps(norm_arr, norm);
            norm2 += norm_arr[0] + norm_arr[1] + norm_arr[2] + norm_arr[3];
            _mm_storeu_ps(x + i, x_n);

            //norm2 += pow(x_new[i] - x[i], 2);
            //x[i] = x_new[i];
            
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
void guardar_solucion(float* x, int N, const char* nombre_archivo) {
    // Abrir el archivo en modo escritura
    FILE* archivo = fopen(nombre_archivo, "w");
    
    // Verificar si se pudo abrir el archivo correctamente
    if (archivo == NULL) {
        fprintf(stderr, "Error al abrir el archivo %s para escritura\n", nombre_archivo);
        return;
    }
    
    // Escribir un encabezado para el archivo
    fprintf(archivo, "# Solución del sistema de ecuaciones lineales\n");
    fprintf(archivo, "# Índice, Valor\n");
    
    // Escribir cada elemento de la solución con su índice
    for (int i = 0; i < N; i++) {
        fprintf(archivo, "%d, %.15e\n", i, x[i]);
    }
    
    // Calcular y escribir algunas estadísticas básicas
    float suma = 0.0;
    float min_val = x[0];
    float max_val = x[0];
    
    for (int i = 0; i < N; i++) {
        suma += x[i];
        if (x[i] < min_val) min_val = x[i];
        if (x[i] > max_val) max_val = x[i];
    }
    
    fprintf(archivo, "\n# Estadísticas\n");
    fprintf(archivo, "# Suma: %.15e\n", suma);
    fprintf(archivo, "# Promedio: %.15e\n", suma / N);
    fprintf(archivo, "# Mínimo: %.15e\n", min_val);
    fprintf(archivo, "# Máximo: %.15e\n", max_val);
    
    // Cerrar el archivo
    fclose(archivo);
    
    printf("Solución guardada en el archivo: %s\n", nombre_archivo);
}