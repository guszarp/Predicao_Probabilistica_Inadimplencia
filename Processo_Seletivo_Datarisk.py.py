"""
Original file is located at
    https://colab.research.google.com/drive/1Rc2MLtAPR4Kzl0fU1Q242DZzNBWz1_9r

# Case Processo Seletivo para Cientista de Dados Junior/Trainee

Este notebook implementa uma solução preditiva para prever a probabilidade de inadimplência com base em dados 
cadastrais, históricos e comportamentais de clientes de uma base de dados da empresa Datarisk

O projeto faz parte de um processo seletivo com case técnico e segue as instruções contidas no PDF de 
instruções Case DS Júnior 2025.pdf. Dito isto, o foco é em previsões probabilísticas, ou seja, 
um valor entre 0 e 1

## Estratégia utilizada

- Modelo escolhido: XGBoostClassifier, por sua robustez, desempenho e capacidade de 
lidar com variáveis tabulares heterogêneas.

- Métrica de avaliação: ROC AUC, Log Loss, Brier Score e KS Statistic, que são adequadas para 
prever probabilidades calibradas.

- Feature engineering: Merge das bases usando 'ID_CLIENTE' e 'SAFRA_REF'.

### Importação de bibliotecas
"""

### Importação de Bibliotecas

# Manipulação de Dados

import pandas as pd
import numpy as np

# Visualização

import matplotlib.pyplot as plt

# Pré Processamento

from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Modelagem

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.calibration import calibration_curve

# Métricas de performance

from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, RocCurveDisplay

# Gerenciamento de pipeline

from sklearn.pipeline import Pipeline

### Carregamento de Bases

df_cad = pd.read_csv("base_cadastral.csv", sep=';')
df_info = pd.read_csv("base_info.csv", sep=';')
df_pag_dev = pd.read_csv("base_pagamentos_desenvolvimento.csv", sep=';')
df_pag_test = pd.read_csv("base_pagamentos_teste.csv", sep=';')

### Criação da coluna 'Inadimplente' para target

df_pag_dev['DATA_VENCIMENTO'] = pd.to_datetime(df_pag_dev['DATA_VENCIMENTO'])
df_pag_dev['DATA_PAGAMENTO'] = pd.to_datetime(df_pag_dev['DATA_PAGAMENTO'])
df_pag_dev['DIAS_ATRASO'] = (df_pag_dev['DATA_PAGAMENTO'] - df_pag_dev['DATA_VENCIMENTO']).dt.days
df_pag_dev['inadimplente'] = (df_pag_dev['DIAS_ATRASO'] >= 5).astype(int)

### Criação do df_train

df_train = df_pag_dev.merge(df_cad, on="ID_CLIENTE", how="left")
df_train = df_train.merge(df_info, on=["ID_CLIENTE", "SAFRA_REF"], how="left")

### Seleção das features e do target

drop_cols = ['ID_CLIENTE', 'SAFRA_REF', 'DATA_PAGAMENTO', 'DATA_VENCIMENTO', 'DATA_EMISSAO_DOCUMENTO', 'inadimplente', 'DIAS_ATRASO']
X = df_train.drop(columns=drop_cols)
y = df_train['inadimplente']

### Separação das númericas e das categóricas

num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

### Split treino/teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

### Pré-processamento (pipeline)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

pipeline.fit(X_train, y_train)

### Previsão de probabilidade

y_probs = pipeline.predict_proba(X_test)[:, 1]

### Aplicação de métricas de performance

# ROC AUC

roc_auc = roc_auc_score(y_test, y_probs)
print("ROC AUC:", roc_auc)


RocCurveDisplay.from_predictions(y_test, y_probs)
plt.title("Curva ROC")
plt.grid()
plt.show()

# KS

ks_stat, _ = ks_2samp(y_probs[y_test==1], y_probs[y_test==0])
print("KS Statistic:", ks_stat)

#Log Loss

print("Log Loss:", log_loss(y_test, y_probs))

# Brier Score

print("Brier Score:", brier_score_loss(y_test, y_probs))

### Curva de calibração

prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("Curva de Calibração")
plt.xlabel("Probabilidade prevista")
plt.ylabel("Proporção real")
plt.grid()
plt.show()

### Preparação da base teste

df_test = df_pag_test.merge(df_cad, on="ID_CLIENTE", how="left")
df_test = df_test.merge(df_info, on=["ID_CLIENTE", "SAFRA_REF"], how="left")

### Previsão

X_test = df_test[X.columns]
test_probs = pipeline.predict_proba(X_test)[:, 1]

### Geração do arquivo csv exigido para submissão

submissao = df_test[['ID_CLIENTE', 'SAFRA_REF']].copy()
submissao['PROBABILIDADE_INADIMPLENCIA'] = test_probs
submissao.to_csv("submissao_case.csv", index=False, sep=';')