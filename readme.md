# Classificação de Imagens de Lesões de Pele com Inteligência Artificial

## Objetivo

Este projeto tem como objetivo realizar a classificação automática de imagens de lesões de pele utilizando técnicas de Inteligência Artificial. São extraídas características das imagens (Hu Moments, LBP, GLCM) e aplicados diferentes algoritmos de classificação (KNN, SVM, Árvore de Decisão, Random Forest, MLP) para avaliar o desempenho na distinção entre diferentes diagnósticos clínicos.

## Estrutura

- `build_dataset.py`: Script para construir os datasets a partir das imagens e máscaras.
- `data_extraction.py`: Funções para extração de características das imagens.
- `classify_data.py`: Treinamento e avaliação dos classificadores.
- `main.py`: Executa todo o pipeline (construção dos datasets e classificação).
- `datasets/`: Datasets gerados.
- `visualizations/`: Métricas e matrizes de confusão dos modelos.

## Setup do Ambiente

1. **Clone o repositório**
   ```bash
   git clone <url-do-repositorio>
   cd classificationAICabalo
   ```

2. **Crie e ative um ambiente virtual**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instale as dependências**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Execução

1. **Executar todo o pipeline (construção dos datasets e classificação)**
   ```bash
   python main.py
   ```

Os resultados das classificações e métricas serão salvos na pasta `visualizations/`.
