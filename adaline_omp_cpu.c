/**
 *
 * Esta versão paraleliza o BGD usando OpenMP para CPUs multicore.
 *
 * REQUISITO (iii): MUDANÇAS DA PARALELIZAÇÃO:
 * 1. adaline_fit_bgd:
 * - O loop principal sobre as amostras (i=0 to N) foi paralelizado.
 * - Foi usada a diretiva `#pragma omp parallel for`
 * - A variável `sum_squared_errors` usa `reduction(+:...)` para somar
 * corretamente os erros de todas as threads.
 * - O array `total_gradient` também usa `reduction(+:...)` para somar
 * os vetores de gradiente de cada thread. A sintaxe
 * `total_gradient[0:ada->num_weights]` é usada para tratar o array
 *
 * 2. load_and_preprocess_data:
 * - Os loops de "Passagem 1", "Passagem 2" e "Padronização" também
 * foram paralelizados com `parallel for` e `reduction` para
 * acelerar o pré-processamento.
 *
 * 3. main:
 * - O código agora executa o treinamento em um loop para testar
 * automaticamente 1, 2, 4, 8, 16 e 32 threads.
 * - `omp_set_num_threads()` é usada para controlar o número de threads.
 * - O modelo é resetado (`delete_adaline` / `new_adaline`) a cada
 * iteração para garantir um treinamento justo.
 *
 * REQUISITO (iv): TEMPOS DE EXECUÇÃO
 *
 * Hardware de Teste: [AMD Ryzen 7 5700X3D 8-Core Processor]
 *
 * Tempo Paralelo (1 thread): 0.048000 s
 * Tempo Paralelo (2 threads): 0.040000 s (Speedup: 1.20x)
 * Tempo Paralelo (4 threads): 0.031000 s (Speedup: 1.55x)
 * Tempo Paralelo (8 threads): 0.032000 s (Speedup: 1.50x)
 * Tempo Paralelo (16 threads): 0.047000 s (Speedup: 1.02x)
 * Tempo Paralelo (32 threads): 0.072000 s (Speedup: 0.67x)
 */

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h> 

#define NUM_SAMPLES 569
#define NUM_FEATURES 30
#define DATA_FILE "data.csv"
#define MAX_ADALINE_ITER 500
#define ADALINE_ACCURACY 1e-5

struct adaline
{
    double eta;
    double *weights;
    int num_weights;
};

struct adaline new_adaline(const int num_features, const double eta);
void delete_adaline(struct adaline *ada);
int adaline_activation(double x);
int adaline_predict(struct adaline *ada, const double *x, double *out);
double adaline_fit_bgd(struct adaline *ada, double **X, const int *y, const int N, int num_threads);
void load_and_preprocess_data(double ***X, int **Y, double **feature_mean, double **feature_stddev, int num_threads);
void free_data(double **X, int *Y, double *feature_mean, double *feature_stddev);

struct adaline new_adaline(const int num_features, const double eta)
{
    if (eta <= 0.f || eta >= 1.f)
    {
        fprintf(stderr, "Taxa de aprendizado deve ser > 0 e < 1\n");
        exit(EXIT_FAILURE);
    }
    int num_weights = num_features + 1;
    struct adaline ada;
    ada.eta = eta;
    ada.num_weights = num_weights;
    ada.weights = (double *)malloc(num_weights * sizeof(double));
    if (!ada.weights) {
        perror("Falha ao alocar pesos");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_weights; i++)
        ada.weights[i] = ((double)rand() / RAND_MAX) * 0.02 - 0.01;
    return ada;
}

void delete_adaline(struct adaline *ada)
{
    if (ada == NULL) return;
    free(ada->weights);
    ada->weights = NULL;
};

int adaline_activation(double x) { return x >= 0 ? 1 : -1; }

int adaline_predict(struct adaline *ada, const double *x, double *net_input_out)
{
    double y = ada->weights[ada->num_weights - 1];
    for (int i = 0; i < ada->num_weights - 1; i++)
        y += x[i] * ada->weights[i];
    
    if (net_input_out)
        *net_input_out = y;
    return adaline_activation(y);
}

double adaline_fit_bgd(struct adaline *ada, double **X, const int *y, const int N, int num_threads)
{
    double mse = 1.0;
    int iter;
    
    omp_set_num_threads(num_threads);

    double *total_gradient = (double *)calloc(ada->num_weights, sizeof(double));
    if (!total_gradient)
    {
        perror("Falha ao alocar gradientes");
        exit(EXIT_FAILURE);
    }

    printf("Iniciando treinamento BGD com %d thread(s)...\n", num_threads);
    
    double start_time = omp_get_wtime();

    for (iter = 0;
         (iter < MAX_ADALINE_ITER) && (mse > ADALINE_ACCURACY);
         iter++)
    {
        double sum_squared_errors = 0.f;
        for(int j=0; j < ada->num_weights; j++) total_gradient[j] = 0.0;
        //REQUISITO (iii): MUDANÇA DA PARALELIZAÇÃO (Início)
        #pragma omp parallel for reduction(+:sum_squared_errors) reduction(+:total_gradient[0:ada->num_weights])
        for (int i = 0; i < N; i++)
        {
            double net_input;
            // ada->weights é lido por todas as threads (shared)
            adaline_predict(ada, X[i], &net_input);

            double prediction_error = (double)y[i] - net_input;
            
            sum_squared_errors += prediction_error * prediction_error;
            for (int j = 0; j < ada->num_weights - 1; j++)
            {
                total_gradient[j] += prediction_error * X[i][j];
            }
            total_gradient[ada->num_weights - 1] += prediction_error;
        }
        mse = sum_squared_errors / N;

        // Atualização dos pesos (executado pela thread master)
        for (int j = 0; j < ada->num_weights; j++)
        {
            ada->weights[j] += ada->eta * total_gradient[j] / (double)N;
        }
    }
    
    double end_time = omp_get_wtime();
    double exec_time = end_time - start_time;

    printf("\tIter %3d: MSE: %.8f\n", iter, mse);
    if (iter < MAX_ADALINE_ITER)
        printf("Convergiu apos %d iteracoes.\n", iter);
    else
        printf("Nao convergiu apos %d iteracoes.\n", iter);
    
    free(total_gradient);
    return exec_time;
}

int main(int argc, char **argv)
{
    srand(time(NULL));

    double **X;
    int *Y;
    double *feature_mean;
    double *feature_stddev;

    // Define as contagens de threads para testar
    int thread_counts[] = {1, 2, 4, 8, 16, 32};
    int num_tests = sizeof(thread_counts) / sizeof(int);
    double exec_times[num_tests];

    int max_threads = thread_counts[num_tests - 1];
    omp_set_num_threads(max_threads);

    load_and_preprocess_data(&X, &Y, &feature_mean, &feature_stddev, max_threads);

    double eta = 0.001;
    struct adaline ada; 

    printf("\n--- Iniciando Testes de Performance OpenMP CPU ---\n");

    for (int t = 0; t < num_tests; t++)
    {
        int n_threads = thread_counts[t];
        if (t > 0) delete_adaline(&ada);
        ada = new_adaline(NUM_FEATURES, eta); // new_adaline retorna uma struct
        
        exec_times[t] = adaline_fit_bgd(&ada, X, Y, NUM_SAMPLES, n_threads);
        
        printf("Tempo de Treinamento (%d threads): %.6f segundos\n", n_threads, exec_times[t]);
        printf("-------------------------------------------------\n");
    }

    printf("\n--- Resultados Finais (Copie para o REQUISITO (iv) no cabecalho) ---\n");
    printf("Hardware: %d Cores Logicos Disponiveis\n", omp_get_max_threads());
    printf("Tempo Sequencial (1 thread): %.6f s\n", exec_times[0]);
    for (int t = 1; t < num_tests; t++)
    {
        printf("Tempo Paralelo (%d threads): %.6f s (Speedup: %.2fx)\n",
               thread_counts[t], exec_times[t], exec_times[0] / exec_times[t]);
    }
    printf("-----------------------------------------------------------------\n");

    int correct_predictions = 0;
    #pragma omp parallel for reduction(+:correct_predictions)
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        int prediction = adaline_predict(&ada, X[i], NULL);
        if (prediction == Y[i])
        {
            correct_predictions++;
        }
    }
    printf("Acuracia final (modelo de %d threads): %.2f%% (%d / %d)\n",
           thread_counts[num_tests-1],
           (double)correct_predictions / NUM_SAMPLES * 100.0,
           correct_predictions, NUM_SAMPLES);
    printf("-----------------------------------------------------------------\n");

    free_data(X, Y, feature_mean, feature_stddev);
    delete_adaline(&ada); 

    return 0;
}

void load_and_preprocess_data(double ***X_ptr, int **Y_ptr, double **mean_ptr, double **stddev_ptr, int num_threads)
{
    double **X = (double **)malloc(NUM_SAMPLES * sizeof(double *));
    int *Y = (int *)malloc(NUM_SAMPLES * sizeof(int));
    double *feature_mean = (double *)calloc(NUM_FEATURES, sizeof(double));
    double *feature_stddev = (double *)calloc(NUM_FEATURES, sizeof(double));

    if (!X || !Y || !feature_mean || !feature_stddev) exit(EXIT_FAILURE);
    for (int i = 0; i < NUM_SAMPLES; i++) {
        X[i] = (double *)malloc(NUM_FEATURES * sizeof(double));
        if (!X[i]) exit(EXIT_FAILURE);
    }
    
    FILE *fp = fopen(DATA_FILE, "r");
    if (!fp) {
        perror("Nao foi possivel abrir data.csv");
        exit(EXIT_FAILURE);
    }
    char line[4096];
    fgets(line, sizeof(line), fp); // Pula cabeçalho

    int sample_count = 0;
    while (fgets(line, sizeof(line), fp) && sample_count < NUM_SAMPLES)
    {
        char *token;
        token = strtok(line, ","); // ID
        token = strtok(NULL, ","); // Diagnóstico
        Y[sample_count] = (strcmp(token, "\"M\"") == 0 || strcmp(token, "M") == 0) ? 1 : -1;
        
        for (int j = 0; j < NUM_FEATURES; j++) {
            token = strtok(NULL, ",");
            X[sample_count][j] = atof(token);
        }
        sample_count++;
    }
    fclose(fp);

    printf("Dados brutos carregados. Iniciando pre-processamento paralelo...\n");


    //Paralelização do cálculo da média
    #pragma omp parallel for reduction(+:feature_mean[0:NUM_FEATURES])
    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            feature_mean[j] += X[i][j];
        }
    }

    for (int j = 0; j < NUM_FEATURES; j++) {
        feature_mean[j] /= NUM_SAMPLES;
    }

    // Paralelização do cálculo do desvio padrão
    #pragma omp parallel for reduction(+:feature_stddev[0:NUM_FEATURES])
    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            feature_stddev[j] += (X[i][j] - feature_mean[j]) * (X[i][j] - feature_mean[j]);
        }
    }

    for (int j = 0; j < NUM_FEATURES; j++) {
        feature_stddev[j] = sqrt(feature_stddev[j] / NUM_SAMPLES);
        if (feature_stddev[j] < 1e-10) feature_stddev[j] = 1.0; 
    }

    // Paralelização da padronização Z-score
    #pragma omp parallel for
    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            X[i][j] = (X[i][j] - feature_mean[j]) / feature_stddev[j];
        }
    }
    
    printf("Carregamento e padronizacao paralelos completos.\n");

    *X_ptr = X;
    *Y_ptr = Y;
    *mean_ptr = feature_mean;
    *stddev_ptr = feature_stddev;
}

void free_data(double **X, int *Y, double *feature_mean, double *feature_stddev)
{
    printf("Limpando memoria...\n");
    for (int i = 0; i < NUM_SAMPLES; i++) {
        free(X[i]);
    }
    free(X);
    free(Y);
    free(feature_mean);
    free(feature_stddev);
}