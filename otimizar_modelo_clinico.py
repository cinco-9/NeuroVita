# -*- coding: utf-8 -*-
"""
OTIMIZAÇÃO DO MODELO CLÍNICO (PIMA) COM VALIDAÇÃO CRUZADA ANINHADA

Tudo que é escolhido (hiperparâmetros, imputação, limiar de decisão) é escolhido só com os dados de treino
de cada partição externa; a partição de teste só mede o desempenho.

Regra de escolha (definida antes de rodar): maior F1 médio; se um modelo mais simples ficar a menos de
1 erro-padrão do melhor, escolhe-se o mais simples. Ordem de simplicidade: Logística < XGBoost < RF < SVM.

Saída: resultados_otimizacao_clinico.json
Autor: David Reis | 2026
"""

import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
SEED = 42

COLUNAS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
           'DiabetesPedigreeFunction', 'Age']
ZERO_E_AUSENTE = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

df = pd.read_csv('pima_diabetes.csv', header=None, names=COLUNAS + ['Outcome'])
df[ZERO_E_AUSENTE] = df[ZERO_E_AUSENTE].replace(0, np.nan)
X = df[COLUNAS].to_numpy(dtype=float)
y = df['Outcome'].to_numpy()
print(f"Pacientes: {len(y)} | diabetes: {y.mean()*100:.1f}%")
print("Ausentes (zeros): " + ", ".join(f"{c} {df[c].isna().mean()*100:.0f}%" for c in ZERO_E_AUSENTE))

# Índices das colunas brutas
G, BP, INS, BMI, AGE = (COLUNAS.index(c) for c in ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age'])


def combinadas(Z):
    """Acrescenta as 5 variáveis combinadas usadas no modelo original (sobre as 8 primeiras colunas)."""
    Z = np.asarray(Z, dtype=float)
    extra = np.column_stack([Z[:, BMI] * Z[:, AGE], Z[:, G] * Z[:, BMI], Z[:, G] * Z[:, AGE],
                             Z[:, INS] * Z[:, G], Z[:, BP] * Z[:, BMI]])
    return np.hstack([Z, extra])


def sem_mudanca(Z):
    return Z


def pipe_linear(modelo):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('fe', FunctionTransformer(sem_mudanca)),
        ('scaler', StandardScaler()),
        ('modelo', modelo),
    ])


FE_OPCOES = [FunctionTransformer(sem_mudanca), FunctionTransformer(combinadas)]

CANDIDATOS = {
    'Regressão Logística': (pipe_linear(LogisticRegression(max_iter=2000)), {
        'imputer__add_indicator': [False, True],
        'fe': FE_OPCOES,
        'modelo__C': [0.01, 0.1, 1, 10],
    }),
    'XGBoost': (Pipeline([('fe', FunctionTransformer(sem_mudanca)), ('modelo', xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, subsample=0.9, eval_metric='logloss', random_state=SEED, n_jobs=1))]), {
        'fe': FE_OPCOES,
        'modelo__max_depth': [2, 3, 4],
        'modelo__min_child_weight': [1, 5],
    }),
    'Random Forest': (pipe_linear(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1)), {
        'imputer__add_indicator': [False, True],
        'modelo__max_depth': [4, 6, None],
        'modelo__min_samples_leaf': [1, 5],
    }),
    'SVM': (pipe_linear(SVC(probability=True, random_state=SEED)), {
        'imputer__add_indicator': [False, True],
        'modelo__C': [0.3, 1, 3],
    }),
}
SIMPLICIDADE = ['Regressão Logística', 'XGBoost', 'Random Forest', 'SVM']
GRADE_LIMIAR = np.arange(0.10, 0.71, 0.01)


def melhor_limiar(y_true, proba):
    f1s = [f1_score(y_true, proba >= t) for t in GRADE_LIMIAR]
    return float(GRADE_LIMIAR[int(np.argmax(f1s))])


def ajustar(pipe, grade, X_tr, y_tr):
    """Escolhe hiperparâmetros (AUC) e limiar (F1) usando só os dados de treino."""
    interno = StratifiedKFold(5, shuffle=True, random_state=SEED)
    busca = GridSearchCV(pipe, grade, scoring='roc_auc', cv=interno, n_jobs=-1)
    busca.fit(X_tr, y_tr)
    proba_oof = cross_val_predict(clone(busca.best_estimator_), X_tr, y_tr, cv=interno,
                                  method='predict_proba', n_jobs=-1)[:, 1]
    return busca.best_estimator_, busca.best_params_, melhor_limiar(y_tr, proba_oof)


externo = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
resultados = {}
for nome, (pipe, grade) in CANDIDATOS.items():
    metricas = []
    for tr, te in externo.split(X, y):
        modelo, _, limiar = ajustar(pipe, grade, X[tr], y[tr])
        p = modelo.predict_proba(X[te])[:, 1]
        pred = (p >= limiar).astype(int)
        metricas.append({
            'f1': f1_score(y[te], pred), 'precisao': precision_score(y[te], pred, zero_division=0),
            'recall': recall_score(y[te], pred), 'auc': roc_auc_score(y[te], p),
            'brier': brier_score_loss(y[te], p), 'limiar': limiar,
        })
    m = pd.DataFrame(metricas)
    resultados[nome] = {
        **{f'{k}_media': round(float(m[k].mean()), 4) for k in m.columns},
        'f1_desvio': round(float(m['f1'].std()), 4),
        'f1_erro_padrao': round(float(m['f1'].std() / np.sqrt(len(m))), 4),
    }
    r = resultados[nome]
    print(f"{nome:20s} F1 {r['f1_media']:.3f} ± {r['f1_desvio']:.3f} | AUC {r['auc_media']:.3f} | "
          f"prec {r['precisao_media']:.3f} | rec {r['recall_media']:.3f} | Brier {r['brier_media']:.3f}")

melhor = max(resultados, key=lambda n: resultados[n]['f1_media'])
corte = resultados[melhor]['f1_media'] - resultados[melhor]['f1_erro_padrao']
escolhido = next(n for n in SIMPLICIDADE if resultados[n]['f1_media'] >= corte)
print(f"\nMaior F1: {melhor} | dentro de 1 erro-padrão e mais simples: {escolhido}")

# Modelo final: ajustado com todos os dados pelo mesmo procedimento
modelo_final, params_final, limiar_final = ajustar(*CANDIDATOS[escolhido], X, y)
params_legiveis = {k: (v.func.__name__ if isinstance(v, FunctionTransformer) else v) for k, v in params_final.items()}
print(f"Hiperparâmetros finais: {params_legiveis} | limiar: {limiar_final:.2f}")

with open('resultados_otimizacao_clinico.json', 'w', encoding='utf-8') as f:
    json.dump({
        'metodo': 'validação cruzada aninhada 5x3 (externa) / 5 (interna); limiar por F1 no treino',
        'resultados': resultados, 'maior_f1': melhor, 'escolhido': escolhido,
        'params_finais': params_legiveis, 'limiar_final': limiar_final,
    }, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Exportação para o app: JSON lido com numpy puro (sem pickle e sem scikit-learn no deploy)
# ---------------------------------------------------------------------------
if escolhido != 'Random Forest' or params_final.get('imputer__add_indicator'):
    raise SystemExit(f"Exportação implementada para Random Forest sem indicador; escolhido: {escolhido} {params_legiveis}")

imputer, scaler, floresta = (modelo_final.named_steps[k] for k in ('imputer', 'scaler', 'modelo'))
arvores = []
for est in floresta.estimators_:
    t = est.tree_
    valores = t.value[:, 0, :]
    arvores.append({
        'esquerda': t.children_left.tolist(), 'direita': t.children_right.tolist(),
        'variavel': t.feature.tolist(), 'limite': t.threshold.tolist(),
        'prob': (valores[:, 1] / valores.sum(axis=1)).tolist(),
    })

r = resultados[escolhido]
meta = {
    'tipo': 'random_forest',
    'features': COLUNAS,
    'zero_e_ausente': ZERO_E_AUSENTE,
    'imputer_medianas': imputer.statistics_.tolist(),
    'scaler_media': scaler.mean_.tolist(),
    'scaler_escala': scaler.scale_.tolist(),
    'threshold': round(limiar_final, 2),
    'metricas_cv_aninhada': {k: r[k] for k in ('f1_media', 'f1_desvio', 'precisao_media', 'recall_media',
                                               'auc_media', 'brier_media')},
    'hiperparametros': params_legiveis,
}
with open('modelo_clinico.json', 'w', encoding='utf-8') as f:
    json.dump({'arvores': arvores}, f)
with open('modelo_clinico_meta.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)


def prever_exportado(Xn):
    """Mesma conta que o app faz: imputação -> padronização -> média das árvores."""
    Z = np.where(np.isnan(Xn), np.array(meta['imputer_medianas']), Xn)
    # arredonda em float32 como o scikit-learn, mas compara em float64 (no numpy 2, float32 <= float do Python compara em float32)
    Z = ((Z - np.array(meta['scaler_media'])) / np.array(meta['scaler_escala'])).astype(np.float32).astype(np.float64)
    total = np.zeros(len(Z))
    for a in arvores:
        for i, z in enumerate(Z):
            no = 0
            while a['esquerda'][no] != -1:
                no = a['esquerda'][no] if z[a['variavel'][no]] <= a['limite'][no] else a['direita'][no]
            total[i] += a['prob'][no]
    return total / len(arvores)


diferenca = np.abs(prever_exportado(X) - modelo_final.predict_proba(X)[:, 1]).max()
print(f"Exportado: modelo_clinico.json + modelo_clinico_meta.json | diferença máx. vs scikit-learn: {diferenca:.2e}")
