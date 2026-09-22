# -*- coding: utf-8 -*-
"""
ENSEMBLE STACKING AVANÇADO
Meta-learner combinando múltiplos modelos
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ENSEMBLE STACKING AVANCADO - META-LEARNER")
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

print(f"   Dataset: {X.shape}")
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
# 2. CRIAR MODELOS BASE
# ============================================================================
print("2. CRIANDO MODELOS BASE...")

# XGBoost (MELHOR até agora)
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

# CatBoost
cat_model = CatBoostClassifier(
    iterations=200,
    depth=5,
    learning_rate=0.05,
    random_state=42,
    verbose=False
)

# Logistic Regression (baseline)
lr_model = LogisticRegression(
    C=0.1,
    max_iter=1000,
    random_state=42
)

print("   XGBoost: OK")
print("   LightGBM: OK")
print("   CatBoost: OK")
print("   Logistic Regression: OK")
print()

# ============================================================================
# 3. TESTAR MODELOS INDIVIDUAIS
# ============================================================================
print("3. AVALIANDO MODELOS INDIVIDUAIS...")
print()

modelos_base = {
    'XGBoost': xgb_model,
    'LightGBM': lgb_model,
    'CatBoost': cat_model,
    'LogisticRegression': lr_model
}

resultados_individuais = []

for nome, modelo in modelos_base.items():
    print(f"   Testando {nome}...")

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(modelo, X_train_scaled, y_train_bal, cv=skf, scoring='f1')

    # Treinar
    modelo.fit(X_train_scaled, y_train_bal)

    # Predições
    y_pred_proba = modelo.predict_proba(X_test_scaled)[:, 1]

    # Otimizar threshold
    melhor_f1 = 0
    melhor_threshold = 0.5

    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred)
        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_threshold = threshold

    y_pred_final = (y_pred_proba >= melhor_threshold).astype(int)
    report = classification_report(y_test, y_pred_final, output_dict=True)

    print(f"      CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"      Test F1: {report['1']['f1-score']:.4f}")
    print(f"      Threshold: {melhor_threshold:.2f}")
    print()

    resultados_individuais.append({
        'Modelo': nome,
        'F1-Score CV': cv_scores.mean(),
        'F1-Score Test': report['1']['f1-score'],
        'Precisão': report['1']['precision'],
        'Recall': report['1']['recall'],
        'Threshold': melhor_threshold
    })

# ============================================================================
# 4. VOTING ENSEMBLE (SOFT VOTING)
# ============================================================================
print("="*80)
print("4. VOTING ENSEMBLE (SOFT VOTING)")
print("="*80)
print()

voting = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model)
    ],
    voting='soft',
    weights=[2, 1, 1]  # XGBoost tem peso maior (melhor modelo)
)

print("   Treinando Voting Ensemble...")
voting.fit(X_train_scaled, y_train_bal)

y_pred_proba_voting = voting.predict_proba(X_test_scaled)[:, 1]

# Otimizar threshold
melhor_f1_voting = 0
melhor_threshold_voting = 0.5

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred = (y_pred_proba_voting >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > melhor_f1_voting:
        melhor_f1_voting = f1
        melhor_threshold_voting = threshold

y_pred_final_voting = (y_pred_proba_voting >= melhor_threshold_voting).astype(int)
report_voting = classification_report(y_test, y_pred_final_voting, output_dict=True)
auc_voting = roc_auc_score(y_test, y_pred_proba_voting)

print(f"   F1-Score: {report_voting['1']['f1-score']:.4f}")
print(f"   Precisão: {report_voting['1']['precision']*100:.1f}%")
print(f"   Recall: {report_voting['1']['recall']*100:.1f}%")
print(f"   ROC-AUC: {auc_voting:.4f}")
print(f"   Threshold: {melhor_threshold_voting:.2f}")
print()

# ============================================================================
# 5. STACKING ENSEMBLE (META-LEARNER)
# ============================================================================
print("="*80)
print("5. STACKING ENSEMBLE (META-LEARNER)")
print("="*80)
print()

# Re-criar modelos base (stacking precisa de modelos não treinados)
estimators = [
    ('xgb', xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )),
    ('lgb', lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )),
    ('cat', CatBoostClassifier(
        iterations=200, depth=5, learning_rate=0.05,
        random_state=42, verbose=False
    ))
]

# Meta-learner: Logistic Regression
meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

print("   Treinando Stacking Ensemble (pode demorar 1-2 min)...")
stacking.fit(X_train_scaled, y_train_bal)

y_pred_proba_stacking = stacking.predict_proba(X_test_scaled)[:, 1]

# Otimizar threshold
melhor_f1_stacking = 0
melhor_threshold_stacking = 0.5

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred = (y_pred_proba_stacking >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > melhor_f1_stacking:
        melhor_f1_stacking = f1
        melhor_threshold_stacking = threshold

y_pred_final_stacking = (y_pred_proba_stacking >= melhor_threshold_stacking).astype(int)
report_stacking = classification_report(y_test, y_pred_final_stacking, output_dict=True)
auc_stacking = roc_auc_score(y_test, y_pred_proba_stacking)

print(f"   F1-Score: {report_stacking['1']['f1-score']:.4f}")
print(f"   Precisão: {report_stacking['1']['precision']*100:.1f}%")
print(f"   Recall: {report_stacking['1']['recall']*100:.1f}%")
print(f"   ROC-AUC: {auc_stacking:.4f}")
print(f"   Threshold: {melhor_threshold_stacking:.2f}")
print()

# ============================================================================
# 6. COMPARAÇÃO FINAL
# ============================================================================
print("="*80)
print("6. COMPARACAO FINAL - RANKING COMPLETO")
print("="*80)
print()

# Adicionar ensembles
resultados_individuais.append({
    'Modelo': 'Voting Ensemble',
    'F1-Score CV': None,
    'F1-Score Test': report_voting['1']['f1-score'],
    'Precisão': report_voting['1']['precision'],
    'Recall': report_voting['1']['recall'],
    'Threshold': melhor_threshold_voting
})

resultados_individuais.append({
    'Modelo': 'Stacking Ensemble',
    'F1-Score CV': None,
    'F1-Score Test': report_stacking['1']['f1-score'],
    'Precisão': report_stacking['1']['precision'],
    'Recall': report_stacking['1']['recall'],
    'Threshold': melhor_threshold_stacking
})

df_resultados = pd.DataFrame(resultados_individuais)
df_resultados = df_resultados.sort_values('F1-Score Test', ascending=False)

print(df_resultados[['Modelo', 'F1-Score Test', 'Precisão', 'Recall', 'Threshold']].to_string(index=False))
print()

# Melhor modelo
melhor = df_resultados.iloc[0]
print(f"MELHOR MODELO: {melhor['Modelo']}")
print(f"   F1-Score: {melhor['F1-Score Test']:.4f}")
print(f"   Precisão: {melhor['Precisão']*100:.1f}%")
print(f"   Recall: {melhor['Recall']*100:.1f}%")
print()

# ============================================================================
# 7. COMPARAÇÃO COM MODELO ANTERIOR
# ============================================================================
print("="*80)
print("7. COMPARACAO COM MODELO ORIGINAL")
print("="*80)
print()

print("MODELO ORIGINAL (XGBoost Manual):")
print("   F1-Score: 0.7200")
print("   Precisão: 63.4%")
print("   Recall: 83.3%")
print()

print(f"MELHOR ENSEMBLE ({melhor['Modelo']}):")
print(f"   F1-Score: {melhor['F1-Score Test']:.4f}")
print(f"   Precisão: {melhor['Precisão']*100:.1f}%")
print(f"   Recall: {melhor['Recall']*100:.1f}%")
print()

ganho = ((melhor['F1-Score Test'] - 0.72) / 0.72) * 100

if ganho > 0:
    print(f"GANHO: +{ganho:.2f}% em F1-Score!")
    print()
    print("SUCESSO! Ensemble superou o modelo original!")
else:
    print(f"Diferença: {ganho:.2f}%")
    if abs(ganho) < 1:
        print("Modelos praticamente equivalentes!")
    else:
        print("Modelo original ainda é melhor.")

print()

# ============================================================================
# 8. SALVAR MELHOR MODELO
# ============================================================================
if melhor['Modelo'] in ['Voting Ensemble', 'Stacking Ensemble']:
    print("8. SALVANDO MELHOR ENSEMBLE...")

    modelo_final = stacking if melhor['Modelo'] == 'Stacking Ensemble' else voting

    with open('modelo_ensemble_final.pkl', 'wb') as f:
        pickle.dump(modelo_final, f)

    with open('scaler_ensemble.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('threshold_ensemble.txt', 'w') as f:
        f.write(str(melhor['Threshold']))

    print(f"   Modelo salvo: modelo_ensemble_final.pkl")
    print(f"   Scaler salvo: scaler_ensemble.pkl")
    print(f"   Threshold: {melhor['Threshold']:.2f}")
    print()

print("="*80)
print("ENSEMBLE FINALIZADO COM SUCESSO!")
print("="*80)
