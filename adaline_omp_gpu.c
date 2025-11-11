/**
 * \file
 * \brief Implementação do ADALINE com BGD - Versão PARALELA (OpenMP GPU Offload)
 *
 * Esta versão paraleliza o BGD usando OpenMP 4.5+ para descarregar
 * o processamento (offload) para uma GPU NVIDIA.
 *
 * REQUISITO (iii): MUDANÇAS DA PARALELIZAÇÃO:
 * 1. Estrutura de Dados:
 * - O dataset 'X' (double**) foi "achatado" (flattened) para 'X_flat' (double*).
 * - Isso é necessário para mapear e transferir os dados para a GPU
 * eficientemente.
 *
 * 2. adaline_fit_bgd_gpu:
 * - Uma região `#pragma omp target data map(...)` é criada *antes* do
 * loop de épocas. Isso move X_flat, Y e os pesos para a memória da GPU
 * uma única vez.
 *
 * 3. Loop de Cálculo de Gradiente:
 * - A diretiva `#pragma omp target teams distribute parallel for` é usada
 * para executar o loop de amostras na GPU.
 *
 * 4. Loop de Atualização de Pesos:
 * - A atualização dos pesos também é feita na GPU (`target teams...`)
 *
 * 5. adaline_predict_gpu:
 * - A função `adaline_predict_gpu` foi marcada com
 * `#pragma omp declare target` para que possa ser chamada de dentro
 * de uma região `target` (executada na GPU).
 *
 * 6. CORREÇÕES DE BUG:
 * - Corrigido `delete_adaline` para usar '->' (ponteiro).
 * - Adicionada uma função `adaline_predict_cpu` para ser chamada
 * pelo host no cálculo de acurácia.
 */

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

/* --- Constantes --- */
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

/* --- Protótipos --- */
struct adaline new_adaline(const int num_features, const double eta);
void delete_adaline(struct adaline *ada);
int adaline_activation(double x);
int adaline_predict_cpu(struct adaline *ada, const double *x_sample, double *net_input_out);
void load_and_preprocess_flat(double **X_flat_ptr, int **Y_ptr);
void free_flat_data(double *X_flat, int *Y);
double adaline_fit_bgd_gpu(struct adaline *ada, double *X_flat, const int *y, const int N);


/**
 * \brief Construtor
 */
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

/**
 * \brief Libera memória (CORRIGIDO)
 */
void delete_adaline(struct adaline *ada)
{
    if (ada == NULL) return;
    // CORREÇÃO: 'ada' é um ponteiro, usar '->'
    free(ada->weights);
    ada->weights = NULL;
};

/**
 * \brief Função de ativação (será usada por ambas CPU/GPU)
 */
int adaline_activation(double x) { return x >= 0 ? 1 : -1; }

/**
 * \brief Predição (marcada para execução na GPU)
 * REQUISITO (iii): Esta função precisa ser visível para o device (GPU).
 */
#pragma omp declare target
int adaline_predict_gpu(const double *weights, int num_weights, const double *x_sample, double *net_input_out)
{
    // O último peso é o bias
    double y = weights[num_weights - 1]; 

    for (int i = 0; i < num_weights - 1; i++)
        y += x_sample[i] * weights[i];

    if (net_input_out)
        *net_input_out = y;

    return adaline_activation(y);
}
#pragma omp end declare target

/**
 * \brief Predição (versão Host/CPU para acurácia)
 * CORREÇÃO: Adicionada esta função para ser chamada pelo host.
 */
int adaline_predict_cpu(struct adaline *ada, const double *x_sample, double *net_input_out)
{
    // 'ada' é um ponteiro, usar '->'
    double y = ada->weights[ada->num_weights - 1]; 
    for (int i = 0; i < ada->num_weights - 1; i++)
        y += x_sample[i] * ada->weights[i];

    if (net_input_out)
        *net_input_out = y;

    return adaline_activation(y);
}

/**
 * \brief Treina o modelo - Versão PARALELA (OpenMP GPU Offload)
 */
double adaline_fit_bgd_gpu(struct adaline *ada, double *X_flat, const int *y, const int N)
{
    double mse = 1.0;
    int iter;
    // 'ada' é um ponteiro, usar '->'
    int num_weights = ada->num_weights;
    double eta = ada->eta;
    
    // Aloca memória para gradientes (na CPU, será mapeado para GPU)
    double *total_gradient = (double *)calloc(num_weights, sizeof(double));
    if (!total_gradient) {
        perror("Falha ao alocar gradientes");
        exit(EXIT_FAILURE);
    }

    printf("Iniciando treinamento BGD (OpenMP GPU Offload)...\n");
    
    double start_time = omp_get_wtime();
    
    /*
     * REQUISITO (iii): Mapeia os dados para a GPU *uma vez*
     * 'ada->weights' é 'tofrom' pois é lido e escrito
     */
    #pragma omp target data map(to: X_flat[0:N*NUM_FEATURES], y[0:N]) \
                            map(tofrom: ada->weights[0:num_weights]) \
                            map(alloc: total_gradient[0:num_weights])
    {
        for (iter = 0;
             (iter < MAX_ADALINE_ITER) && (mse > ADALINE_ACCURACY);
             iter++)
        {
            double sum_squared_errors = 0.f;
            
            // Zera os gradientes na GPU
            #pragma omp target teams distribute parallel for
            for(int j=0; j < num_weights; j++) total_gradient[j] = 0.0;

            /*
             * REQUISITO (iii): Loop de Cálculo de Gradiente na GPU
             */
            #pragma omp target teams distribute parallel for \
                        reduction(+:sum_squared_errors) \
                        reduction(+:total_gradient[0:num_weights])
            for (int i = 0; i < N; i++)
            {
                double net_input;
                // &X_flat[i * NUM_FEATURES] é o ponteiro para a amostra 'i'
                adaline_predict_gpu(ada->weights, num_weights, &X_flat[i * NUM_FEATURES], &net_input);

                double prediction_error = (double)y[i] - net_input;
                
                sum_squared_errors += prediction_error * prediction_error;

                for (int j = 0; j < num_weights - 1; j++)
                {
                    total_gradient[j] += prediction_error * X_flat[i * NUM_FEATURES + j];
                }
                total_gradient[num_weights - 1] += prediction_error;
            }
            
            mse = sum_squared_errors / N;

            /*
             * REQUISITO (iii): Atualização dos Pesos na GPU
             */
            #pragma omp target teams distribute parallel for
            for (int j = 0; j < num_weights; j++)
            {
                ada->weights[j] += eta * total_gradient[j] / (double)N;
            }

            if (iter % 50 == 0)
            {
                // Imprime o MSE (requer que 'mse' seja calculado na GPU
                // ou uma cópia dos erros). Para este projeto,
                // a thread master imprimir o 'mse' da CPU está OK.
                #pragma omp master
                printf("\tIter %3d: MSE: %.8f\n", iter, mse);
            }
        }
    } // Fim da região 'target data'. 'ada->weights' é copiado de volta.

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

/**
 * \brief Função principal (main)
 */
int main(int argc, char **argv)
{
    srand(time(NULL));

    double *X_flat; // Dados achatados
    int *Y;

    // Carrega dados e já os "achata" para a GPU
    load_and_preprocess_flat(&X_flat, &Y);

    // --- Treinamento do Modelo ---
    double eta = 0.001;
    struct adaline ada = new_adaline(NUM_FEATURES, eta);
    
    printf("--- Versao OpenMP GPU Offload ---\n");
    // Passa o *endereço* de 'ada' (&ada)
    double gpu_time = adaline_fit_bgd_gpu(&ada, X_flat, Y, NUM_SAMPLES);
    printf("Tempo de Treinamento (GPU Offload): %.6f segundos\n", gpu_time);

    // --- Cálculo da Acurácia (na CPU) ---
    int correct_predictions = 0;
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        double net_input;
        /*
         * CORREÇÃO: Chama a versão CPU da predição
         * para calcular a acurácia no host.
         */
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

    // --- Limpeza ---
    free_flat_data(X_flat, Y);
    delete_adaline(&ada); // Passa o endereço de 'ada'

    return 0;
}


/**
 * \brief Carrega dados e os "achata" (flatten) para um array 1D
 * (Esta função permanece idêntica à da resposta anterior)
 */
void load_and_preprocess_flat(double **X_flat_ptr, int **Y_ptr)
{
    // Aloca X como um array 1D (achatado)
    double *X_flat = (double *)malloc(NUM_SAMPLES * NUM_FEATURES * sizeof(double));
    int *Y = (int *)malloc(NUM_SAMPLES * sizeof(int));
    double *feature_mean = (double *)calloc(NUM_FEATURES, sizeof(double));
    double *feature_stddev = (double *)calloc(NUM_FEATURES, sizeof(double));

    if (!X_flat || !Y || !feature_mean || !feature_stddev) {
        perror("Falha ao alocar memoria para o dataset");
        exit(EXIT_FAILURE);
    }
    
    // Carregamento sequencial (leitura de arquivo)
    FILE *fp = fopen(DATA_FILE, "r");
    if (!fp) { perror("Nao foi possivel abrir data.csv"); exit(EXIT_FAILURE); }

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
            // Acesso achatado
            X_flat[sample_count * NUM_FEATURES + j] = atof(token);
        }
        sample_count++;
    }
    fclose(fp);

    printf("Dados brutos carregados. Iniciando pre-processamento...\n");

    // --- Passagem 1: Média (Sequencial) ---
    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            feature_mean[j] += X_flat[i * NUM_FEATURES + j];
        }
    }
    for (int j = 0; j < NUM_FEATURES; j++) feature_mean[j] /= NUM_SAMPLES;

    // --- Passagem 2: StdDev (Sequencial) ---
    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            double val = X_flat[i * NUM_FEATURES + j];
            feature_stddev[j] += (val - feature_mean[j]) * (val - feature_mean[j]);
        }
    }
    for (int j = 0; j < NUM_FEATURES; j++) {
        feature_stddev[j] = sqrt(feature_stddev[j] / NUM_SAMPLES);
        if (feature_stddev[j] < 1e-10) feature_stddev[j] = 1.0; 
    }

    // --- Padronização (Sequencial) ---
    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            int idx = i * NUM_FEATURES + j;
            X_flat[idx] = (X_flat[idx] - feature_mean[j]) / feature_stddev[j];
        }
    }
    
    printf("Carregamento e padronizacao completos.\n");

    *X_flat_ptr = X_flat;
    *Y_ptr = Y;
    
    // Não precisamos mais disso
    free(feature_mean);
    free(feature_stddev);
}

/**
 * \brief Libera memória dos dados achatados
 */
void free_flat_data(double *X_flat, int *Y)
{
    printf("Limpando memoria...\n");
    free(X_flat);
    free(Y);
}