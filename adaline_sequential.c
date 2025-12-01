/**
 * Esta é a versão base para medição de speedup.
 * O treinamento foi modificado de Stochastic (SGD) para Batch (BGD)
 * para permitir a paralelização nas versões futuras.
 *
 * O BGD calcula o gradiente para todas as amostras antes de
 * aplicar uma única atualização de pesos por época.
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
void load_and_preprocess_data(double ***X, int **Y, double **feature_mean, double **feature_stddev);
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
    if (!ada.weights)
    {
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
    double *total_gradient = (double *)calloc(ada->num_weights, sizeof(double));
    if (!total_gradient)
    {
        perror("Falha ao alocar gradientes");
        exit(EXIT_FAILURE);
    }

    printf("Iniciando treinamento com Batch Gradient Descent...\n");
    
    double start_time = omp_get_wtime();

    for (iter = 0;
         (iter < MAX_ADALINE_ITER) && (mse > ADALINE_ACCURACY);
         iter++)
    {
        double sum_squared_errors = 0.f;
        for(int j=0; j < ada->num_weights; j++) total_gradient[j] = 0.0;
        for (int i = 0; i < N; i++) 
        {
            double net_input;
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

        for (int j = 0; j < ada->num_weights; j++)
        {
            ada->weights[j] += ada->eta * total_gradient[j] / (double)N;
        }

        if (iter % 50 == 0)
            printf("\tIter %3d: MSE: %.8f\n", iter, mse);
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

    load_and_preprocess_data(&X, &Y, &feature_mean, &feature_stddev);

    double eta = 0.001;
    struct adaline ada = new_adaline(NUM_FEATURES, eta);
    
    printf("--- Versao Sequencial (1 Thread) ---\n");
    double sequential_time = adaline_fit_bgd(&ada, X, Y, NUM_SAMPLES, 1);
    printf("Tempo de Treinamento (Sequencial): %.6f segundos\n", sequential_time);

    int correct_predictions = 0;
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        int prediction = adaline_predict(&ada, X[i], NULL);
        if (prediction == Y[i])
        {
            correct_predictions++;
        }
    }
    printf("----------------------------------\n");
    printf("Acuracia final: %.2f%% (%d / %d)\n",
           (double)correct_predictions / NUM_SAMPLES * 100.0,
           correct_predictions, NUM_SAMPLES);
    printf("----------------------------------\n");

    free_data(X, Y, feature_mean, feature_stddev);
    delete_adaline(&ada);

    return 0;
}

void load_and_preprocess_data(double ***X_ptr, int **Y_ptr, double **mean_ptr, double **stddev_ptr)
{
    double **X = (double **)malloc(NUM_SAMPLES * sizeof(double *));
    int *Y = (int *)malloc(NUM_SAMPLES * sizeof(int));
    double *feature_mean = (double *)calloc(NUM_FEATURES, sizeof(double));
    double *feature_stddev = (double *)calloc(NUM_FEATURES, sizeof(double));

    if (!X || !Y || !feature_mean || !feature_stddev) {
        perror("Falha ao alocar memoria para o dataset");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < NUM_SAMPLES; i++) {
        X[i] = (double *)malloc(NUM_FEATURES * sizeof(double));
        if (!X[i]) {
            perror("Falha ao alocar memoria para o vetor de features");
            exit(EXIT_FAILURE);
        }
    }

    FILE *fp = fopen(DATA_FILE, "r");
    if (!fp) {
        perror("Nao foi possivel abrir data.csv");
        exit(EXIT_FAILURE);
    }

    char line[4096];
    fgets(line, sizeof(line), fp); // Pula cabeçalho

    printf("Carregando dados... (Passo 1: Calculando Media)\n");
    int sample_count = 0;
    while (fgets(line, sizeof(line), fp) && sample_count < NUM_SAMPLES)
    {
        char *token;
        token = strtok(line, ","); // ID
        token = strtok(NULL, ","); // Diagnóstico
        
        for (int j = 0; j < NUM_FEATURES; j++) {
            token = strtok(NULL, ",");
            feature_mean[j] += atof(token);
        }
        sample_count++;
    }

    for (int j = 0; j < NUM_FEATURES; j++) {
        feature_mean[j] /= sample_count;
    }
    
    rewind(fp); 
    fgets(line, sizeof(line), fp); 

    printf("Carregando dados... (Passo 2: Calculando StdDev e Padronizando)\n");
    
    sample_count = 0;
    while (fgets(line, sizeof(line), fp) && sample_count < NUM_SAMPLES)
    {
        char *token;
        token = strtok(line, ","); // ID
        token = strtok(NULL, ","); // Diagnóstico
        
        // Mapeia Y (M = 1, B = -1)
        Y[sample_count] = (strcmp(token, "\"M\"") == 0 || strcmp(token, "M") == 0) ? 1 : -1;
        
        for (int j = 0; j < NUM_FEATURES; j++) {
            token = strtok(NULL, ",");
            double val = atof(token);
            X[sample_count][j] = val; // Armazena valor bruto
            feature_stddev[j] += (val - feature_mean[j]) * (val - feature_mean[j]);
        }
        sample_count++;
    }

    for (int j = 0; j < NUM_FEATURES; j++) {
        feature_stddev[j] = sqrt(feature_stddev[j] / sample_count);
        if (feature_stddev[j] < 1e-10) feature_stddev[j] = 1.0; 
    }

    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            X[i][j] = (X[i][j] - feature_mean[j]) / feature_stddev[j];
        }
    }
    
    fclose(fp);
    printf("Carregamento e padronizacao completos.\n");

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