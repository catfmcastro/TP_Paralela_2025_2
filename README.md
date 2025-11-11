# Projeto de Paralelização de IA: ADALINE (Adaptive Linear Neuron)

Este projeto implementa e paralisa um modelo de Rede Neural ADALINE
para o dataset de Câncer de Mama de Wisconsin (data.csv).

## 1. Sobre a Aplicação

O ADALINE (Adaptive Linear Neuron) é um dos tipos mais antigos de redes
neurais de camada única. Ele é um classificador linear, similar ao
Perceptron, mas seu treinamento é diferente.

O objetivo é encontrar um hiperplano (uma linha, em 2D) que separe
duas classes de dados. Neste projeto, usamos 30 características
(features) de tumores de mama para classificar um tumor como
**Maligno (1)** ou **Benigno (-1)**.

### Modificação: Batch Gradient Descent (BGD)

A forma de treinamento padrão (SGD) atualiza os pesos após cada amostra,
criando uma dependência sequencial que impede a paralelização.

Para este projeto, o algoritmo foi modificado para **Batch Gradient
Descent (BGD)**:
1.  **Cálculo (Paralelo):** Os "erros" de todas as 569 amostras são
    calculados em paralelo.
2.  **Redução (Paralelo):** Os "gradientes" (direção de correção) de
    todas as amostras são somados.
3.  **Aplicação (Sequencial):** Uma única atualização de pesos é
    aplicada usando a média dos gradientes.

O passo 1 é o mais custoso e é o principal alvo da paralelização
em OpenMP (CPU e GPU) e CUDA.

## 2. Instruções de Compilação e Execução

Você precisará de:
* Um compilador C (gcc ou clang) com suporte a OpenMP.
* O CUDA Toolkit (nvcc) para a versão CUDA.
* O arquivo `data.csv` no mesmo diretório.

---
### Versão 1: Sequencial (BGD)

Compilação:
gcc adaline_sequential.c -o adaline_seq -lm -fopenmp

(Nota: -fopenmp é usado apenas para a biblioteca de tempo omp_get_wtime())

Execução:
./adaline_seq

---
### Versão 2: OpenMP para Multicore (CPU)

Compilação:
gcc adaline_omp_cpu.c -o adaline_omp_cpu -lm -fopenmp

Execução:
./adaline_omp_cpu

(Este executável irá testar e relatar automaticamente os tempos
para 1, 2, 4, 8, 16 e 32 threads)

---
### Versão 3: OpenMP para GPU (Offloading)

Compilação:
gcc -fopenmp  adaline_omp_gpu.c -o adaline_omp_gpu -lm  

Execução:
./adaline_omp_gpu

---
### Versão 4: CUDA para GPU

Compilação (usando o compilador CUDA 'nvcc'):
nvcc adaline_cuda.cu -o adaline_cuda -lm

(Nota: nvcc precisa linkar a biblioteca matemática '-lm' se
 omp_get_wtime() não for encontrado; neste caso, usamos -Xcompiler
 "-fopenmp" para linkar o timer, ou podemos trocar o timer por
 time.h)

Compilação mais robusta (incluindo o timer OpenMP):
nvcc adaline_cuda.cu -o adaline_cuda -Xcompiler "-fopenmp" -lm

Execução:
./adaline_cuda