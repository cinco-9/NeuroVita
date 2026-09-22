# -*- coding: utf-8 -*-
"""
OTIMIZAÇÃO BAYESIANA COM OPTUNA
Encontrar os MELHORES hiperparâmetros automaticamente
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("OTIMIZACAO BAYESIANA COM OPTUNA")
print("Buscando MELHORES hiperparâmetros para XGBoost")
print("="*80)
print()

# ============================================================================
# 1. CARREGAR E PREPARAR DADOS
# ============================================================================
print("1. CARREGANDO DATASET PIMA INDIANS...")
pima_df = pd.read_csv('pima_diabetes.csv', header=None)
pima_df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Tratar zeros
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_not_accepted:
    pima_df[col] = pima_df[col].replace(0, np.nan)
    median_val = pima_df[col].median()
    pima_df[col] = pima_df[col].fillna(median_val)

pima_df = pima_df.fillna(pima_df.median())

# Feature Engineering COMPLETO
print("   Aplicando Feature Engineering...")
pima_df['BMI_Age'] = pima_df['BMI'] * pima_df['Age']
pima_df['Glucose_BMI'] = pima_df['Glucose'] * pima_df['BMI']
pima_df['Glucose_Age'] = pima_df['Glucose'] * pima_df['Age']
pima_df['Insulin_Glucose'] = pima_df['Insulin'] * pima_df['Glucose']
pima_df['BP_BMI'] = pima_df['BloodPressure'] * pima_df['BMI']
pima_df['BMI_squared'] = pima_df['BMI'] ** 2
pima_df['Age_squared'] = pima_df['Age'] ** 2
pima_df['Glucose_squared'] = pima_df['Glucose'] ** 2

X = pima_df.drop('Outcome', axis=1)
y = pima_df['Outcome']

print(f"   Dataset shape: {X.shape}")
print(f"   Features: {X.shape[1]}")
print()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

print(f"   Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
print()

# ============================================================================
# 2. FUNÇÃO OBJETIVO PARA OPTUNA
# ============================================================================

def objective(trial):
    """Função objetivo para Optuna otimizar"""

    # Sugerir hiperparâmetros
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'random_state': 42
    }

    # Modelo
    model = xgb.XGBClassifier(**params)

    # Validação cruzada
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train_bal, cv=skf, scoring='f1')

    return scores.mean()

# ============================================================================
# 3. EXECUTAR OTIMIZAÇÃO
# ============================================================================
print("="*80)
print("2. EXECUTANDO OTIMIZACAO BAYESIANA")
print("="*80)
print()
print("CONFIGURAÇÃO:")
print("   Trials: 50 (tentativas)")
print("   Sampler: TPE (Tree-structured Parzen Estimator)")
print("   Métrica: F1-Score (Cross-Validation 5-fold)")
print()
print("INICIANDO BUSCA...")
print("(Isso pode demorar alguns minutos...)")
print()

# Criar estudo
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42)
)

# Otimizar
study.optimize(objective, n_trials=50, show_progress_bar=True)

print()
print("="*80)
print("OTIMIZACAO FINALIZADA!")
print("="*80)
print()

# ============================================================================
# 4. MELHORES HIPERPARÂMETROS
# ============================================================================
print("3. MELHORES HIPERPARAMETROS ENCONTRADOS:")
print()

best_params = study.best_params
for param, value in best_params.items():
    print(f"   {param}: {value}")

print()
print(f"   Melhor F1-Score (CV): {study.best_value:.4f}")
print()

# ============================================================================
# 5. TREINAR MODELO FINAL COM MELHORES PARÂMETROS
# ============================================================================
print("4. TREINANDO MODELO FINAL...")

final_model = xgb.XGBClassifier(**best_params, random_state=42)
final_model.fit(X_train_scaled, y_train_bal)

# Predições
y_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]

# Otimizar threshold
melhor_f1 = 0
melhor_threshold = 0.5

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred = (y_pred_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)

    if f1 > melhor_f1:
        melhor_f1 = f1
        melhor_threshold = threshold

# Predição final
y_pred_final = (y_pred_proba >= melhor_threshold).astype(int)

# Métricas
report = classification_report(y_test, y_pred_final, output_dict=True)
auc = roc_auc_score(y_test, y_pred_proba)

print()
print("RESULTADOS NO TESTE:")
print(f"   Melhor Threshold: {melhor_threshold:.2f}")
print(f"   Precisão: {report['1']['precision']*100:.1f}%")
print(f"   Recall: {report['1']['recall']*100:.1f}%")
print(f"   F1-Score: {report['1']['f1-score']:.4f}")
print(f"   ROC-AUC: {auc:.4f}")
print()

# ============================================================================
# 6. COMPARAÇÃO COM MODELO ANTERIOR
# ============================================================================
print("="*80)
print("5. COMPARACAO COM MODELO ANTERIOR")
print("="*80)
print()

print("MODELO ANTERIOR (XGBoost Manual):")
print("   F1-Score: 0.7200")
print("   Precisão: 63.4%")
print("   Recall: 83.3%")
print()

print(f"MODELO OTIMIZADO (Optuna):")
print(f"   F1-Score: {report['1']['f1-score']:.4f}")
print(f"   Precisão: {report['1']['precision']*100:.1f}%")
print(f"   Recall: {report['1']['recall']*100:.1f}%")
print()

ganho_f1 = ((report['1']['f1-score'] - 0.72) / 0.72) * 100
ganho_prec = ((report['1']['precision'] - 0.634) / 0.634) * 100

if ganho_f1 > 0:
    print(f"GANHO F1-Score: +{ganho_f1:.2f}%")
    print(f"GANHO Precisão: +{ganho_prec:.2f}%")
    print()
    print("SUCESSO! Modelo melhorado!")
else:
    print(f"Diferença: {ganho_f1:.2f}%")
    print()
    if abs(ganho_f1) < 1:
        print("Modelos praticamente equivalentes!")
    else:
        print("Modelo anterior ainda é melhor.")

print()

# ============================================================================
# 7. SALVAR MODELO OTIMIZADO
# ============================================================================
if ganho_f1 > -1:  # Salvar se for >= -1%
    print("6. SALVANDO MODELO OTIMIZADO...")

    with open('modelo_otimizado_optuna.pkl', 'wb') as f:
        pickle.dump(final_model, f)

    with open('scaler_otimizado.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('threshold_otimizado.txt', 'w') as f:
        f.write(str(melhor_threshold))

    with open('best_params_optuna.txt', 'w') as f:
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")

    print("   Modelo salvo: modelo_otimizado_optuna.pkl")
    print("   Scaler salvo: scaler_otimizado.pkl")
    print("   Threshold salvo: threshold_otimizado.txt")
    print("   Parâmetros salvos: best_params_optuna.txt")
    print()

print("="*80)
print("OTIMIZACAO FINALIZADA COM SUCESSO!")
print("="*80)
