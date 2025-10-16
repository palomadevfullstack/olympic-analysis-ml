# Análise de Dados Olímpicos e Previsão de Medalhas com Machine Learning

## 📖 Visão Geral do Projeto

Este projeto realiza uma análise de dados completa do dataset "120 Years of Olympic History" do Kaggle. O objetivo é extrair insights sobre a performance de atletas e países ao longo da história dos Jogos Olímpicos e, ao final, construir um modelo de Machine Learning capaz de prever a probabilidade de um atleta ganhar uma medalha.

A análise foi desenvolvida de ponta a ponta, desde a exploração e visualização dos dados até o treinamento, avaliação e interpretação de modelos preditivos.

## 📂 Estrutura de Arquivos

O projeto está organizado de forma modular para separar as etapas de análise, preparação e modelagem.

```
/
├── data/
|   └── athlete_events.zip
├── scripts-python/
|    ├── analysis_olympic_athlete.py
|    ├── ml_data_preparation.py
|    └── ml_training.py
├── notebooks/
|   └── analise_olimpica.ipynb
├── images/
|   ├── evolucao_altura_natacao_vs_ginastica.png
|   ├── evolucao_peso_natacao_vs_ginastica.png
|   ├── top_15_paises_medalhas_olimpicas.png
|   ├── heatmap_paises.png
|   ├── analise_brasil_esportes.png
|   └── feature_importance.png
├── requirements.txt
└── README.md
```

## 📊 Dataset

O projeto utiliza o dataset [120 years of Olympic History (athletes and results)](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results) do Kaggle. Ele contém mais de 270.000 registros de participações de atletas em eventos olímpicos desde 1896.

## 🚀 Metodologia

O projeto foi dividido em três fases principais:

### 1. Análise Exploratória de Dados (EDA)

Nesta fase, buscamos entender os padrões e as histórias contidas nos dados. As principais análises foram:
* **Evolução do Perfil Físico:** Análise da altura e peso médio de atletas de Natação e Ginástica, mostrando a especialização física ao longo do tempo.
* **Potências vs. Especialistas:** Criação de um heatmap para identificar países com performance diversificada (EUA, URS) e países especialistas em modalidades específicas (Itália na Esgrima, Grã-Bretanha no Remo).
* **Deep Dive no Brasil:** Uma análise focada na performance olímpica do Brasil, identificando seus esportes mais vitoriosos e a evolução do número de medalhas por edição.

### 2. Modelagem Preditiva (Machine Learning)

O objetivo desta fase foi construir um modelo para prever se um atleta ganharia uma medalha (`Medal_Won` = 1 ou 0).
* **Pré-processamento:** Os dados foram limpos, valores faltantes foram tratados com a imputação pela mediana, e variáveis categóricas (`Sex`, `NOC`, `Sport`) foram transformadas em numéricas via One-Hot Encoding.
* **Modelo Baseline (Regressão Logística):** Um primeiro modelo foi treinado para estabelecer uma performance base. A análise do `classification_report` revelou um **recall de apenas 8%** para a classe de medalhistas, evidenciando o problema dos dados desbalanceados.
* **Modelo Avançado (Random Forest):** Um segundo modelo, `RandomForestClassifier`, foi treinado. O resultado foi um salto de performance massivo, **aumentando o recall para 38%** (uma melhoria de mais de 4x), com uma precisão de 55%.

### 3. Interpretação do Modelo

Com um modelo de alta performance, a etapa final foi "abrir a caixa-preta" para entender suas decisões.
* A análise de `feature_importances_` do Random Forest revelou que os preditores mais fortes para o sucesso olímpico são os **atributos físicos do atleta (Idade, Peso, Altura)**, seguidos pela **nacionalidade de potências históricas (EUA, URS)**.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3
* **Bibliotecas:** Pandas, Matplotlib, Seaborn, Scikit-learn
* **Ambiente:** Jupyter Notebook

## ⚙️ Como Executar o Projeto


1.  Clone o repositório:
    ```bash
    git clone https://github.com/palomadevfullstack/olympic-analysis-ml
    ```
2.  Navegue até a pasta do projeto:
    ```bash
    cd olympic-analysis-ml
    ```
3.  Descompacte o dataset dentro da pasta `data/`.

4.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

5.  Execute a análise:

    **Opção A (Recomendado): Execução via Jupyter Notebook**
    Abra o notebook `analise_olimpica.ipynb` na pasta `notebooks/`. Ele serve como o relatório principal, guiando por toda a análise e exibindo os resultados de forma organizada.

    **Opção B: Execução Sequencial dos Scripts**
    Se preferir, você pode executar os scripts Python em sequência a partir do terminal (estando na pasta raiz do projeto):
    ```bash
    # 1. Executar a análise exploratória
    python scripts-python/analysis_olympic_athlete.py

    # 2. Preparar os dados para o Machine Learning
    python scripts-python/ml_data_preparation.py

    # 3. Treinar e avaliar os modelos
    python scripts-python/ml_training.py
    ```

## 🔮 Possíveis Melhorias Futuras

* **Otimização de Hiperparâmetros:** Utilizar `GridSearchCV` ou `RandomizedSearchCV` para encontrar a melhor configuração para o Random Forest.
* **Experimentar Outros Modelos:** Testar algoritmos de Gradient Boosting como XGBoost ou LightGBM.
* **Engenharia de Features Avançada:** Criar novas variáveis para enriquecer o modelo.
