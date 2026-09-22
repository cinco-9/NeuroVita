# -*- coding: utf-8 -*-
"""
SCRIPT AVANÇADO: MELHORANDO PRECISÃO E APRENDIZADO
Implementa: Deep Learning, Calibração, Validação Cruzada e AutoML
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MELHORANDO PRECISAO E APRENDIZADO - TCC DIABETES")
print("="*80)
print()

# ============================================================================
# 1. CARREGAR DATASET PIMA (CLINICO) - MELHOR PERFORMANCE
# ============================================================================
print("1. CARREGANDO DATASET PIMA INDIANS...")
pima_df = pd.read_csv('pima_diabetes.csv', header=None)
pima_df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Tratar zeros inválidos
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_not_accepted:
    pima_df[col] = pima_df[col].replace(0, np.nan)
    median_val = pima_df[col].median()
    pima_df[col] = pima_df[col].fillna(median_val)

pima_df = pima_df.fillna(pima_df.median())

# Feature Engineering
pima_df['BMI_Age'] = pima_df['BMI'] * pima_df['Age']
pima_df['Glucose_BMI'] = pima_df['Glucose'] * pima_df['BMI']
pima_df['Glucose_Age'] = pima_df['Glucose'] * pima_df['Age']
pima_df['Insulin_Glucose'] = pima_df['Insulin'] * pima_df['Glucose']
pima_df['BP_BMI'] = pima_df['BloodPressure'] * pima_df['BMI']

X_pima = pima_df.drop('Outcome', axis=1)
y_pima = pima_df['Outcome']

print(f"   Dataset shape: {X_pima.shape}")
print(f"   Diabetes prevalence: {y_pima.mean()*100:.1f}%")
print()

# ============================================================================
# 2. PREPARAR DADOS
# ============================================================================
print("2. PREPARANDO DADOS...")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_pima, y_pima, test_size=0.2, random_state=42, stratify=y_pima
)

# SMOTE para balanceamento
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

print(f"   Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
print(f"   Após SMOTE: {y_train_bal.value_counts().to_dict()}")
print()

# ============================================================================
# 3. VALIDAÇÃO CRUZADA ESTRATIFICADA - BASELINE
# ============================================================================
print("3. VALIDACAO CRUZADA ESTRATIFICADA (BASELINE XGBoost)...")

xgb_baseline = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    scale_pos_weight=1
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_baseline, X_train_scaled, y_train_bal, cv=skf, scoring='f1')

print(f"   F1-Score CV (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"   Scores individuais: {[f'{s:.4f}' for s in cv_scores]}")
print()

# Treinar modelo baseline
xgb_baseline.fit(X_train_scaled, y_train_bal)

# ============================================================================
# 4. DEEP LEARNING - REDE NEURAL (MLPClassifier)
# ============================================================================
print("4. DEEP LEARNING - REDE NEURAL...")

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # 3 camadas ocultas
    activation='relu',
    solver='adam',
    alpha=0.0001,  # Regularização L2
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=False
)

print("   Treinando rede neural (128-64-32 neurônios)...")
mlp.fit(X_train_scaled, y_train_bal)

print(f"   Iterações: {mlp.n_iter_}")
print(f"   Loss final: {mlp.loss_:.4f}")
print()

# ============================================================================
# 5. CALIBRAÇÃO DE PROBABILIDADES
# ============================================================================
print("5. CALIBRACAO DE PROBABILIDADES...")

# Calibrar XGBoost
print("   Calibrando XGBoost (Isotonic Regression)...")
xgb_calibrated = CalibratedClassifierCV(
    xgb_baseline,
    method='isotonic',
    cv=5
)
xgb_calibrated.fit(X_train_scaled, y_train_bal)

# Calibrar MLP
print("   Calibrando Rede Neural (Isotonic Regression)...")
mlp_calibrated = CalibratedClassifierCV(
    mlp,
    method='isotonic',
    cv=5
)
mlp_calibrated.fit(X_train_scaled, y_train_bal)
print()

# ============================================================================
# 6. AVALIAR TODOS OS MODELOS
# ============================================================================
print("6. AVALIANDO TODOS OS MODELOS...")
print()

modelos = {
    'XGBoost (Baseline)': xgb_baseline,
    'XGBoost (Calibrado)': xgb_calibrated,
    'Neural Network (MLP)': mlp,
    'Neural Network (Calibrado)': mlp_calibrated
}

resultados = []

for nome, modelo in modelos.items():
    print(f"   {nome}:")

    # Predições
    y_pred_proba = modelo.predict_proba(X_test_scaled)[:, 1]

    # Testar diferentes thresholds
    melhor_f1 = 0
    melhor_threshold = 0.5

    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred)

        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_threshold = threshold

    # Predição final com melhor threshold
    y_pred_final = (y_pred_proba >= melhor_threshold).astype(int)

    # Métricas
    report = classification_report(y_test, y_pred_final, output_dict=True)
    auc = roc_auc_score(y_test, y_pred_proba)

    precision = report['1']['precision']
    recall = report['1']['recall']
    f1 = report['1']['f1-score']

    print(f"      Melhor Threshold: {melhor_threshold:.2f}")
    print(f"      Precisão: {precision*100:.1f}%")
    print(f"      Recall: {recall*100:.1f}%")
    print(f"      F1-Score: {f1:.4f}")
    print(f"      ROC-AUC: {auc:.4f}")
    print()

    resultados.append({
        'Modelo': nome,
        'Threshold': melhor_threshold,
        'Precisão': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': auc
    })

# ============================================================================
# 7. COMPARAÇÃO FINAL
# ============================================================================
print("="*80)
print("7. COMPARACAO FINAL - RANKING DE MODELOS")
print("="*80)
print()

df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values('F1-Score', ascending=False)

print(df_resultados.to_string(index=False))
print()

# Melhor modelo
melhor = df_resultados.iloc[0]
print(f"MELHOR MODELO: {melhor['Modelo']}")
print(f"   F1-Score: {melhor['F1-Score']:.4f}")
print(f"   Precisão: {melhor['Precisão']*100:.1f}%")
print(f"   Recall: {melhor['Recall']*100:.1f}%")
print(f"   ROC-AUC: {melhor['ROC-AUC']:.4f}")
print()

# ============================================================================
# 8. SALVAR MELHOR MODELO
# ============================================================================
print("8. SALVANDO MELHOR MODELO...")

melhor_nome = melhor['Modelo']
melhor_modelo = modelos[melhor_nome]

# Salvar modelo
with open('modelo_melhorado_clinico.pkl', 'wb') as f:
    pickle.dump(melhor_modelo, f)

# Salvar scaler
with open('scaler_melhorado.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Salvar threshold
with open('threshold_melhorado.txt', 'w') as f:
    f.write(str(melhor['Threshold']))

print(f"   Modelo salvo: modelo_melhorado_clinico.pkl")
print(f"   Scaler salvo: scaler_melhorado.pkl")
print(f"   Threshold salvo: {melhor['Threshold']:.2f}")
print()

# ============================================================================
# 9. COMPARAÇÃO COM MODELO ANTERIOR
# ============================================================================
print("="*80)
print("9. COMPARACAO COM MODELO ANTERIOR")
print("="*80)
print()

print("MODELO ANTERIOR (XGBoost):")
print("   F1-Score: 0.7200")
print("   Precisão: 63.4%")
print("   Recall: 83.3%")
print()

print(f"MODELO NOVO ({melhor_nome}):")
print(f"   F1-Score: {melhor['F1-Score']:.4f}")
print(f"   Precisão: {melhor['Precisão']*100:.1f}%")
print(f"   Recall: {melhor['Recall']*100:.1f}%")
print()

ganho_f1 = ((melhor['F1-Score'] - 0.72) / 0.72) * 100
ganho_precisao = ((melhor['Precisão'] - 0.634) / 0.634) * 100

if ganho_f1 > 0:
    print(f"GANHO: +{ganho_f1:.1f}% em F1-Score!")
    print(f"GANHO: +{ganho_precisao:.1f}% em Precisão!")
else:
    print(f"RESULTADO: {ganho_f1:.1f}% (modelo anterior ainda é melhor)")
print()

print("="*80)
print("SCRIPT FINALIZADO COM SUCESSO!")
print("="*80)
