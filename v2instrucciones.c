#include "counter.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

void initializeSystem(float*** a, float** b, float** x, int N);
void JacobiMenosInstrucciones(float** a, float* b, float* x, int N, float tol, int max_iter);
void JacobiDividido(float** a, float* b, float* x, int N, float tol, int max_iter);
void JacobiDesenrollado(float** a, float* b, float* x, int N, float tol, int max_iter);
void JacobiBloques(float** a, float* b, float* x, int N, float tol, int max_iter);
void save_csv(char *name, int N, int nhilos, int nciclos);

#define BLOCK_SIZE 32

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

    //JacobiMenosInstrucciones(a, b, x, N, tol, max_iter);
    //JacobiDividido(a, b, x, N, tol, max_iter);
    //JacobiDesenrollado(a, b, x, N, tol, max_iter);
    JacobiBloques(a, b, x, N, tol, max_iter);
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

void JacobiMenosInstrucciones(float** a, float* b, float* x, int N, float tol, int max_iter){
    float* x_new = (float*)malloc(N * sizeof(float));
    if (!x_new) {
        printf("Error en la asignación de memoria\n");
        return;
    }
    float* x_old=x; //Usaré x_old para aclarar que es el vector de la iteración anterior
    float* tmp; //La usaré para hacer un intercambio de punteros entre x y x_new
    double* inv_a = (double*)malloc(N * sizeof(double)); //Lo uso para guardar el inverso de a[i][i]

    //Pues es accedido en cada iteración y no quiero calcularlo cada vez
    if (!inv_a) {
        printf("Error en la asignación de memoria\n");
        free(x_new);
        return;
    }
    float norm2;
    


    start_counter();
    for (int i = 0; i < N; i++){
        inv_a[i] = 1.0 / a[i][i]; //Guardo el inverso de a[i][i]
    }

    for (int iter = 0; iter < max_iter; iter++){
        norm2 = 0.0;
        for(int i = 0; i < N; i++){
            float sigma = 0.0;
            for(int j = 0; j < N; j++){
                if(i != j){
                    sigma += a[i][j] * x_old[j];
                }
            }
            x_new[i] = (b[i] - sigma)*inv_a[i]; //Multiplico por el inverso de a[i][i]
            float diff = x_new[i] - x_old[i];
            norm2 += diff * diff; //Calculo la norma al cuadrado
            
        }
        
        //Intercambio de punteros
        tmp = x_new;
        x_new = x_old;
        x_old = tmp;
        
        if (sqrt(norm2) < tol){
            double cycles = get_counter();
            printf("Iteraciones: %d\n", iter);
            printf("Norma: %.15e\n", norm2);
            printf("Ciclos: %.0f\n", cycles);
            printf("Segundos: %.2f\n", (float)cycles / 2.5e9);
            break;
        }
    }
    
    free(x_new);
    free(inv_a);

}

/*
Además de usar las mejoras de versión anterior, realizamos la division
de un bucle de forma que nos evitamos la condición i != j
*/

void JacobiDividido(float** a, float* b, float* x, int N, float tol, int max_iter){
    float* x_new = (float*)malloc(N * sizeof(float));
    if (!x_new) {
        printf("Error en la asignación de memoria\n");
        return;
    }
    float* x_old=x; //Usaré x_old para aclarar que es el vector de la iteración anterior
    float* tmp; //La usaré para hacer un intercambio de punteros entre x y x_new
    double* inv_a = (double*)malloc(N * sizeof(double)); //Lo uso para guardar el inverso de a[i][i]

    //Pues es accedido en cada iteración y no quiero calcularlo cada vez
    if (!inv_a) {
        printf("Error en la asignación de memoria\n");
        free(x_new);
        return;
    }
    float norm2;
    


    start_counter();
    for (int i = 0; i < N; i++){
        inv_a[i] = 1.0 / a[i][i]; //Guardo el inverso de a[i][i]
    }

    for (int iter = 0; iter < max_iter; iter++){
        norm2 = 0.0;
        for(int i = 0; i < N; i++){
            float sigma = 0.0;
            
            //Dividimos el calculo de sigma en dos partes
            //Evitamos la condición i != j
            // Primera parte: j < i
            for(int j=0; j<i; j++) sigma += a[i][j] * x_old[j];
            // Segunda parte: j > i
            for(int j=i+1; j<N; j++) sigma += a[i][j] * x_old[j];
            
            
            x_new[i] = (b[i] - sigma)*inv_a[i]; //Multiplico por el inverso de a[i][i]
            float diff = x_new[i] - x_old[i];
            norm2 += diff * diff; //Calculo la norma al cuadrado
            
        }
        
        //Intercambio de punteros
        tmp = x_new;
        x_new = x_old;
        x_old = tmp;
        
        if (sqrt(norm2) < tol){
            double cycles = get_counter();
            printf("Iteraciones: %d\n", iter);
            printf("Norma: %.15e\n", norm2);
            printf("Ciclos: %.0f\n", cycles);
            printf("Segundos: %.2f\n", (float)cycles / 2.5e9);
            break;
        }
    }
    
    free(x_new);
    free(inv_a);

}

/*
Lo mismo que JacobiDividido, pero desenrollando el bucle
de forma que se hacen 4 operaciones por iteración
*/
void JacobiDesenrollado(float** a, float* b, float* x, int N, float tol, int max_iter){
    float* x_new = (float*)malloc(N * sizeof(float));
    if (!x_new) {
        printf("Error en la asignación de memoria\n");
        return;
    }
    float* x_old=x; //Usaré x_old para aclarar que es el vector de la iteración anterior
    float* tmp; //La usaré para hacer un intercambio de punteros entre x y x_new
    double* inv_a = (double*)malloc(N * sizeof(double)); //Lo uso para guardar el inverso de a[i][i]

    //Pues es accedido en cada iteración y no quiero calcularlo cada vez
    if (!inv_a) {
        printf("Error en la asignación de memoria\n");
        free(x_new);
        return;
    }
    float norm2;
    


    start_counter();
    for (int i = 0; i < N; i++){
        inv_a[i] = 1.0 / a[i][i]; //Guardo el inverso de a[i][i]
    }

    for (int iter = 0; iter < max_iter; iter++){
        norm2 = 0.0;
        for(int i = 0; i < N; i++){
            float sigma = 0.0;
            // Primera parte: j < i
            int j;
            for (j = 0; j <= i - 4; j += 4) {
                sigma += a[i][j] * x_old[j];
                sigma += a[i][j + 1] * x_old[j + 1];
                sigma += a[i][j + 2] * x_old[j + 2];
                sigma += a[i][j + 3] * x_old[j + 3];
            }
            for (; j < i; j++) {
                sigma += a[i][j] * x_old[j];
            }

            // Segunda parte: j > i
            for (j = i + 1; j <= N - 4; j += 4) {
                sigma += a[i][j] * x_old[j];
                sigma += a[i][j + 1] * x_old[j + 1];
                sigma += a[i][j + 2] * x_old[j + 2];
                sigma += a[i][j + 3] * x_old[j + 3];
            }
            for (; j < N; j++) {
                sigma += a[i][j] * x_old[j];
            }
                        
            
            x_new[i] = (b[i] - sigma)*inv_a[i]; //Multiplico por el inverso de a[i][i]
            float diff = x_new[i] - x_old[i];
            norm2 += diff * diff; //Calculo la norma al cuadrado
            
        }
        
        //Intercambio de punteros
        tmp = x_new;
        x_new = x_old;
        x_old = tmp;
        
        if (sqrt(norm2) < tol){
            double cycles = get_counter();
            printf("Iteraciones: %d\n", iter);
            printf("Norma: %.15e\n", norm2);
            printf("Ciclos: %.0f\n", cycles);
            printf("Segundos: %.2f\n", (float)cycles / 2.5e9);
            break;
        }
    }
    
    free(x_new);
    free(inv_a);

}

/*
Lo mismo que aplicamos la division de bucles y bloques
*/
void JacobiBloques(float** a, float* b, float* x, int N, float tol, int max_iter){
    double cycles;
    float* x_new = (float*)malloc(N * sizeof(float));
    if (!x_new) {
        printf("Error en la asignación de memoria\n");
        return;
    }
    float* x_old=x; //Usaré x_old para aclarar que es el vector de la iteración anterior
    float* tmp; //La usaré para hacer un intercambio de punteros entre x y x_new
    double* inv_a = (double*)malloc(N * sizeof(double)); //Lo uso para guardar el inverso de a[i][i]

    //Pues es accedido en cada iteración y no quiero calcularlo cada vez
    if (!inv_a) {
        printf("Error en la asignación de memoria\n");
        free(x_new);
        return;
    }
    float norm2;

    start_counter();
    for (int i = 0; i < N; i++){
        inv_a[i] = 1.0 / a[i][i]; //Guardo el inverso de a[i][i]
    }

    for (int iter = 0; iter < max_iter; iter++){
        norm2 = 0.0;
        for(int i = 0; i < N; i++){
            float sigma = 0.0;
            
            //Dividimos el calculo de sigma en dos partes
            //Evitamos la condición i != j
            // Procesamiento por bloques para j < i
            for(int b=0; b<i; b+=BLOCK_SIZE) {
                int end = (b + BLOCK_SIZE) < i ? (b + BLOCK_SIZE) : i;
                for(int j=b; j<end; j++) {
                    sigma += a[i][j] * x_old[j];
                }
            }
            
            // Procesamiento por bloques para j > i
            for(int b=i+1; b<N; b+=BLOCK_SIZE) {
                int end = (b + BLOCK_SIZE) < N ? (b + BLOCK_SIZE) : N;
                for(int j=b; j<end; j++) {
                    sigma += a[i][j] * x_old[j];
                }
            }
            
            x_new[i] = (b[i] - sigma)*inv_a[i]; //Multiplico por el inverso de a[i][i]
            float diff = x_new[i] - x_old[i];
            norm2 += diff * diff; //Calculo la norma al cuadrado
            
        }
        
        //Intercambio de punteros
        tmp = x_new;
        x_new = x_old;
        x_old = tmp;
        
        if (sqrt(norm2) < tol){
            cycles = get_counter();
            printf("Iteraciones: %d\n", iter);
            printf("Norma: %.15e\n", norm2);
            printf("Ciclos: %.0f\n", cycles);
            printf("Segundos: %.2f\n", (float)cycles / 2.5e9);
            break;
        }
    }

    save_csv("v2-blocks.csv", N, 0, cycles); // Guardar resultados en CSV
    
    free(x_new);
    free(inv_a);

}

void save_csv(char *name, int N, int nhilos, int nciclos) {
    FILE *f = fopen(name, "a");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    printf("printing to file\n");
    fprintf(f, "%d,%d,%d\n", N, nhilos, nciclos);
    fclose(f);
}
