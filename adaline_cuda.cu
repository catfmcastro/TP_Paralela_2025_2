/**
 * REQUISITO (iii): MUDANÇAS DA PARALELIZAÇÃO:
 * 1. Estrutura de Dados:
 * - Usa os mesmos dados "achatados" (X_flat) da versão OpenMP GPU.
 *
 * 2. Gerenciamento de Memória:
 * - A função `adaline_fit_bgd_cuda` agora aloca memória na GPU
 * usando `cudaMalloc` (para d_X, d_Y, d_weights, d_errors, d_sq_errors).
 * - Os dados são copiados do Host (CPU) para o Device (GPU) com `cudaMemcpy`.
 *
 * 3. Kernels CUDA:
 * - `calculate_errors_kernel` (Kernel 1):
 * - Lançado com N (569) threads.
 * - Cada thread (i) processa uma amostra.
 * - Calcula o net_input, o erro (`d_Y[i] - net_input`) e o erro
 * quadrático.
 * - Salva os resultados em `d_errors[i]` e `d_sq_errors[i]`.
 *
 * - `update_weights_kernel` (Kernel 2):
 * - Lançado com num_weights (31) threads.
 * - Cada thread (j) é responsável por *um* peso.
 * - A thread (j) faz um loop sobre *todas* as N amostras (uma redução)
 * para calcular o `total_gradient` para aquele peso.
 * - A thread (j) atualiza seu peso `d_weights[j]`.
 *
 * 4. Fluxo de Controle:
 * - O loop de épocas (iter) é controlado pela CPU (host).
 * - A cada época, os dois kernels são lançados.
 * - O array `d_sq_errors` é copiado da GPU para a CPU (`cudaMemcpy` D2H)
 * para calcular o MSE e verificar a convergência.
 * - No final, `d_weights` é copiado de volta para a CPU.
 */

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h> // Usado APENAS para o timer omp_get_wtime()

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define NUM_SAMPLES 569
#define NUM_FEATURES 30
#define DATA_FILE "data.csv"
#define MAX_ADALINE_ITER 500
#define ADALINE_ACCURACY 1e-5

// Ver se tem erro no cuda
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Erro CUDA em %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

struct adaline
{
    double eta;
    double *weights;
    int num_weights;
};

struct adaline new_adaline(const int num_features, const double eta);
void delete_adaline(struct adaline *ada);
void load_and_preprocess_flat(double **X_flat_ptr, int **Y_ptr);
void free_flat_data(double *X_flat, int *Y);
double adaline_fit_bgd_cuda(struct adaline *ada, double *X_flat, const int *y, const int N);
int adaline_predict_cpu(struct adaline *ada, const double *x_sample, double *net_input_out);

__global__
void calculate_errors_kernel(const double *d_weights, int num_weights,
                             const double *d_X, const int *d_Y,
                             double *d_errors, double *d_sq_errors, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= N) return;

    double net_input = d_weights[num_weights - 1]; 

    const double *x_sample = &d_X[i * NUM_FEATURES]; 
    
    for (int j = 0; j < num_weights - 1; j++)
    {
        net_input += x_sample[j] * d_weights[j];
    }

    // Calcula e armazena erro e erro quadrático
    double error = (double)d_Y[i] - net_input;
    d_errors[i] = error;
    d_sq_errors[i] = error * error;
}

__global__
void update_weights_kernel(double *d_weights, int num_weights,
                           const double *d_X, const double *d_errors,
                           double eta, int N)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    if (j >= num_weights) return;
    double total_gradient = 0.0;
    
    for (int i = 0; i < N; i++)
    {
        double feature_val = (j == num_weights - 1) ? 
                             1.0 : 
                             d_X[i * NUM_FEATURES + j];
        
        total_gradient += d_errors[i] * feature_val;
    }
    d_weights[j] += eta * total_gradient / (double)N;
}
double adaline_fit_bgd_cuda(struct adaline *ada, double *h_X_flat, const int *h_Y, const int N)
{
    int num_weights = ada->num_weights;
    double eta = ada->eta;
    double *d_X, *d_weights, *d_errors, *d_sq_errors;
    int *d_Y;

    CUDA_CHECK(cudaMalloc(&d_X, N * NUM_FEATURES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Y, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_weights, num_weights * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_errors, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sq_errors, N * sizeof(double)));

    double *h_sq_errors = (double *)malloc(N * sizeof(double));
    if(!h_sq_errors) { perror("Falha ao alocar h_sq_errors"); exit(1); }

    printf("Copiando dados para a VRAM da GPU...\n");
    CUDA_CHECK(cudaMemcpy(d_X, h_X_flat, N * NUM_FEATURES * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, h_Y, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, ada->weights, num_weights * sizeof(double), cudaMemcpyHostToDevice));

    printf("Iniciando treinamento BGD (CUDA)...\n");
    double start_time = omp_get_wtime();

    // Configuração dos blocos/grids
    int threads_per_block = 256;
    int blocks_for_samples = (N + threads_per_block - 1) / threads_per_block;
    int blocks_for_weights = (num_weights + threads_per_block - 1) / threads_per_block;

    double mse = 1.0;
    int iter;

    for (iter = 0;
         (iter < MAX_ADALINE_ITER) && (mse > ADALINE_ACCURACY);
         iter++)
    {
        calculate_errors_kernel<<<blocks_for_samples, threads_per_block>>>(
            d_weights, num_weights, d_X, d_Y, d_errors, d_sq_errors, N
        );
        CUDA_CHECK(cudaGetLastError()); 

        update_weights_kernel<<<blocks_for_weights, threads_per_block>>>(
            d_weights, num_weights, d_X, d_errors, eta, N
        );
        CUDA_CHECK(cudaGetLastError()); 
        CUDA_CHECK(cudaMemcpy(h_sq_errors, d_sq_errors, N * sizeof(double), cudaMemcpyDeviceToHost));

        double sum_squared_errors = 0.0;
        for (int i = 0; i < N; i++) {
            sum_squared_errors += h_sq_errors[i];
        }
        mse = sum_squared_errors / N;

        if (iter % 50 == 0)
            printf("\tIter %3d: MSE: %.8f\n", iter, mse);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    double end_time = omp_get_wtime();
    double exec_time = end_time - start_time;

    printf("\tIter %3d: MSE: %.8f\n", iter, mse);
    if (iter < MAX_ADALINE_ITER)
        printf("Convergiu apos %d iteracoes.\n", iter);
    else
        printf("Nao convergiu apos %d iteracoes.\n", iter);
        
    CUDA_CHECK(cudaMemcpy(ada->weights, d_weights, num_weights * sizeof(double), cudaMemcpyDeviceToHost));

    free(h_sq_errors);
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_errors));
    CUDA_CHECK(cudaFree(d_sq_errors));

    return exec_time;
}

int main(int argc, char **argv)
{
    srand(time(NULL));

    double *X_flat; 
    int *Y;

    load_and_preprocess_flat(&X_flat, &Y);

    double eta = 0.001;
    struct adaline ada = new_adaline(NUM_FEATURES, eta);
    
    printf("--- Versao CUDA ---\n");
    double gpu_time = adaline_fit_bgd_cuda(&ada, X_flat, Y, NUM_SAMPLES);
    printf("Tempo de Treinamento (CUDA): %.6f segundos\n", gpu_time);

    int correct_predictions = 0;
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        double net_input;
        int prediction = adaline_predict_cpu(&ada, &X_flat[i * NUM_FEATURES], &net_input);
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

    free_flat_data(X_flat, Y);
    delete_adaline(&ada);

    return 0;
}

struct adaline new_adaline(const int num_features, const double eta)
{
    int num_weights = num_features + 1;
    struct adaline ada;
    ada.eta = eta;
    ada.num_weights = num_weights;
    ada.weights = (double *)malloc(num_weights * sizeof(double));
    if (!ada.weights) { perror("Falha ao alocar pesos"); exit(EXIT_FAILURE); }
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
int adaline_predict_cpu(struct adaline *ada, const double *x_sample, double *net_input_out)
{
    double y = ada->weights[ada->num_weights - 1];
    for (int i = 0; i < ada->num_weights - 1; i++)
        y += x_sample[i] * ada->weights[i];
    if (net_input_out)
        *net_input_out = y;
    return adaline_activation(y);
}

void load_and_preprocess_flat(double **X_flat_ptr, int **Y_ptr)
{
    double *X_flat = (double *)malloc(NUM_SAMPLES * NUM_FEATURES * sizeof(double));
    int *Y = (int *)malloc(NUM_SAMPLES * sizeof(int));
    double *feature_mean = (double *)calloc(NUM_FEATURES, sizeof(double));
    double *feature_stddev = (double *)calloc(NUM_FEATURES, sizeof(double));

    if (!X_flat || !Y || !feature_mean || !feature_stddev) exit(EXIT_FAILURE);
    
    FILE *fp = fopen(DATA_FILE, "r");
    if (!fp) exit(EXIT_FAILURE);
    char line[4096];
    fgets(line, sizeof(line), fp); 

    int sample_count = 0;
    while (fgets(line, sizeof(line), fp) && sample_count < NUM_SAMPLES)
    {
        char *token;
        token = strtok(line, ","); 
        token = strtok(NULL, ","); 
        Y[sample_count] = (strcmp(token, "\"M\"") == 0 || strcmp(token, "M") == 0) ? 1 : -1;
        for (int j = 0; j < NUM_FEATURES; j++) {
            token = strtok(NULL, ",");
            X_flat[sample_count * NUM_FEATURES + j] = atof(token);
        }
        sample_count++;
    }
    fclose(fp);
    printf("Dados brutos carregados. Iniciando pre-processamento...\n");

    for (int i = 0; i < NUM_SAMPLES; i++)
        for (int j = 0; j < NUM_FEATURES; j++)
            feature_mean[j] += X_flat[i * NUM_FEATURES + j];
    for (int j = 0; j < NUM_FEATURES; j++) feature_mean[j] /= NUM_SAMPLES;

    for (int i = 0; i < NUM_SAMPLES; i++)
        for (int j = 0; j < NUM_FEATURES; j++)
            feature_stddev[j] += (X_flat[i * NUM_FEATURES + j] - feature_mean[j]) * (X_flat[i * NUM_FEATURES + j] - feature_mean[j]);
    for (int j = 0; j < NUM_FEATURES; j++) {
        feature_stddev[j] = sqrt(feature_stddev[j] / NUM_SAMPLES);
        if (feature_stddev[j] < 1e-10) feature_stddev[j] = 1.0; 
    }

    for (int i = 0; i < NUM_SAMPLES; i++)
        for (int j = 0; j < NUM_FEATURES; j++)
            X_flat[i * NUM_FEATURES + j] = (X_flat[i * NUM_FEATURES + j] - feature_mean[j]) / feature_stddev[j];
    
    printf("Carregamento e padronizacao completos.\n");
    *X_flat_ptr = X_flat;
    *Y_ptr = Y;
    free(feature_mean);
    free(feature_stddev);
}

void free_flat_data(double *X_flat, int *Y)
{
    printf("Limpando memoria...\n");
    free(X_flat);
    free(Y);
}