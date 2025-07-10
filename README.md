# 📊 Previsão de Inadimplência com XGBoost

Este projeto implementa uma solução preditiva para prever a **probabilidade de inadimplência** com base em dados cadastrais, históricos e comportamentais de clientes de uma base de dados da empresa **Datarisk**.

O notebook foi desenvolvido como parte de um processo seletivo e segue as instruções do PDF do case técnico **Case DS Júnior 2025.pdf**.  
O foco está em **previsões probabilísticas**, ou seja, o modelo prevê um valor entre 0 e 1 indicando a chance de inadimplência.

---

## 🚀 Estratégia Utilizada

- **Modelo escolhido:**  
  `XGBoostClassifier` — escolhido por sua robustez, desempenho e capacidade de lidar com variáveis tabulares heterogêneas.

- **Métricas de avaliação:**  
  - `ROC AUC`  
  - `Log Loss`  
  - `Brier Score`  
  - `KS Statistic`  
  Essas métricas são adequadas para avaliar a performance de previsões probabilísticas.

- **Feature engineering:**  
  Realizei o `merge` entre diferentes bases de dados utilizando as colunas `ID_CLIENTE` e `SAFRA_REF`.

- **Criação da variável target:**  
  A variável target (inadimplência) foi construída com base nos dados fornecidos, conforme orientações do case.

- **Pipeline de pré-processamento:**  
  Desenvolvi um pipeline com:
  - Tratamento de valores ausentes  
  - Codificação de variáveis categóricas  
  - Padronização de variáveis numéricas  
  - Separação treino/teste com estratificação da variável target  

---

## 📚 Bibliotecas Utilizadas

- [`pandas`](https://pandas.pydata.org/) — manipulação de dados  
- [`numpy`](https://numpy.org/) — operações numéricas  
- [`scikit-learn`](https://scikit-learn.org/) — pipeline, métricas e pré-processamento  
- [`scipy.stats.ks_2samp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html) — cálculo da estatística KS  
- [`xgboost`](https://xgboost.readthedocs.io/) — modelo de aprendizado de máquina utilizado

---

## 💡 Considerações Finais

O projeto demonstra um pipeline completo de modelagem preditiva com foco em probabilidade de inadimplência. A abordagem utilizada pode ser aplicada ou expandida para problemas semelhantes que envolvam **score de crédito**, **risco financeiro** ou **classificação com desbalanceamento**.
