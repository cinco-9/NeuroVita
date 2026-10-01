# -*- coding: utf-8 -*-
"""
Treina o modelo comportamental (BRFSS 2015) usado pela página "Modelo Comportamental" do app.

Divisão 60/20/20: hiperparâmetros (AUC) e limiar (F1) escolhidos na validação; métricas finais no teste.
Gera modelo_comportamental.json (XGBoost) e modelo_comportamental_meta.json (features, threshold, métricas).
"""

import itertools
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'PhysActivity', 'GenHlth', 'Age', 'Sex']

df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
X = df[FEATURES]
y = df['Diabetes_binary'].astype(int)
print(f"Registros: {len(df):,} | prevalência: {y.mean()*100:.1f}%")

X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=42)

# Referência: regressão logística
logistica = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_train, y_train)
auc_logistica = roc_auc_score(y_val, logistica.predict_proba(X_val)[:, 1])
print(f"Regressão logística: AUC validação {auc_logistica:.4f}")

# Grade de XGBoost
grade = {'max_depth': [3, 4, 5, 6], 'n_estimators': [300, 600], 'min_child_weight': [1, 10]}
melhor_auc, melhor_params, modelo = -1, None, None
for valores in itertools.product(*grade.values()):
    params = dict(zip(grade, valores))
    candidato = xgb.XGBClassifier(learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric='logloss',
                                  random_state=42, n_jobs=-1, **params).fit(X_train, y_train)
    auc = roc_auc_score(y_val, candidato.predict_proba(X_val)[:, 1])
    print(f"  XGBoost {params}: AUC validação {auc:.4f}")
    if auc > melhor_auc:
        melhor_auc, melhor_params, modelo = auc, params, candidato
print(f"Melhor XGBoost: {melhor_params} (AUC validação {melhor_auc:.4f})")

proba_val = modelo.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.10, 0.61, 0.01)
threshold = float(round(thresholds[int(np.argmax([f1_score(y_val, proba_val >= t) for t in thresholds]))], 2))

proba_test = modelo.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= threshold).astype(int)
metricas = {
    'f1_score': round(float(f1_score(y_test, pred_test)), 3),
    'precision': round(float(precision_score(y_test, pred_test)), 3),
    'recall': round(float(recall_score(y_test, pred_test)), 3),
    'roc_auc': round(float(roc_auc_score(y_test, proba_test)), 3),
    'brier': round(float(brier_score_loss(y_test, proba_test)), 4),
    'threshold': threshold,
    'hiperparametros': melhor_params,
    'auc_validacao_logistica': round(float(auc_logistica), 4),
    'n_treino': int(len(X_train)),
    'n_teste': int(len(X_test)),
}
print(json.dumps(metricas, indent=2))

# Formato JSON (portável entre versões do xgboost/numpy, ao contrário de pickle)
modelo.save_model('modelo_comportamental.json')
with open('modelo_comportamental_meta.json', 'w', encoding='utf-8') as f:
    json.dump({'tipo': 'xgboost', 'features': FEATURES, 'threshold': threshold, 'metricas': metricas}, f, indent=2)
print("Salvo: modelo_comportamental.json + modelo_comportamental_meta.json")
