# -*- coding: utf-8 -*-
"""
ANALISE SHAP - EXPLICABILIDADE DO MODELO
Entendendo POR QUE o modelo toma decisoes
Comparacao: Features Clinicas vs Comportamentais
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ANALISE SHAP - EXPLICABILIDADE DO MODELO")
print("Entendendo POR QUE o modelo toma decisoes")
print("=" * 80)

# ============================================================================
# 1. PREPARAR PIMA INDIANS
# ============================================================================

print("\n[1/6] Carregando e preparando Pima Indians...")

pima_columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

pima_df = pd.read_csv('pima_diabetes.csv', names=pima_columns)

# Tratar zeros
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_not_accepted:
    pima_df[col] = pima_df[col].replace(0, np.nan)
    pima_df[col] = pima_df[col].fillna(pima_df[col].median())

# Feature Engineering (simplificado)
pima_df['Glucose_BMI'] = pima_df['Glucose'] * pima_df['BMI']
pima_df['Age_Glucose'] = pima_df['Age'] * pima_df['Glucose']

pima_df = pima_df.fillna(pima_df.median())

X_pima = pima_df.drop('Outcome', axis=1)
y_pima = pima_df['Outcome']

X_pima_train, X_pima_test, y_pima_train, y_pima_test = train_test_split(
    X_pima, y_pima, test_size=0.2, random_state=42, stratify=y_pima
)

scaler_pima = StandardScaler()
X_pima_train_scaled = scaler_pima.fit_transform(X_pima_train)
X_pima_test_scaled = scaler_pima.transform(X_pima_test)

print(f"Pima: {len(X_pima)} registros, {X_pima.shape[1]} features")

# ============================================================================
# 2. TREINAR MODELO PIMA (XGBoost simples para SHAP)
# ============================================================================

print("\n[2/6] Treinando modelo XGBoost no Pima...")

smote_pima = SMOTE(random_state=42)
X_pima_train_smote, y_pima_train_smote = smote_pima.fit_resample(X_pima_train_scaled, y_pima_train)

modelo_pima = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

modelo_pima.fit(X_pima_train_smote, y_pima_train_smote)
print("Modelo Pima treinado!")

# ============================================================================
# 3. PREPARAR BRFSS (AMOSTRA)
# ============================================================================

print("\n[3/6] Carregando e preparando BRFSS (amostra para performance)...")

brfss_df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

# Usar amostra simples para SHAP (mais rapido)
brfss_sample = brfss_df.sample(n=10000, random_state=42)

# Feature Engineering (simplificado)
brfss_sample['BMI_x_HighBP'] = brfss_sample['BMI'] * brfss_sample['HighBP']
brfss_sample['Age_x_GenHlth'] = brfss_sample['Age'] * brfss_sample['GenHlth']

X_brfss = brfss_sample.drop('Diabetes_binary', axis=1)
y_brfss = brfss_sample['Diabetes_binary']

X_brfss_train, X_brfss_test, y_brfss_train, y_brfss_test = train_test_split(
    X_brfss, y_brfss, test_size=0.2, random_state=42, stratify=y_brfss
)

scaler_brfss = StandardScaler()
X_brfss_train_scaled = scaler_brfss.fit_transform(X_brfss_train)
X_brfss_test_scaled = scaler_brfss.transform(X_brfss_test)

print(f"BRFSS (amostra): {len(X_brfss)} registros, {X_brfss.shape[1]} features")

# ============================================================================
# 4. TREINAR MODELO BRFSS
# ============================================================================

print("\n[4/6] Treinando modelo XGBoost no BRFSS...")

smote_brfss = SMOTE(random_state=42)
X_brfss_train_smote, y_brfss_train_smote = smote_brfss.fit_resample(X_brfss_train_scaled, y_brfss_train)

modelo_brfss = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

modelo_brfss.fit(X_brfss_train_smote, y_brfss_train_smote)
print("Modelo BRFSS treinado!")

# ============================================================================
# 5. ANALISE SHAP - PIMA INDIANS
# ============================================================================

print("\n" + "=" * 80)
print("[5/6] Criando explicacoes SHAP para Pima Indians...")
print("=" * 80)

# Criar explainer
explainer_pima = shap.TreeExplainer(modelo_pima)

# Calcular SHAP values (usar amostra do teste)
X_pima_test_sample = X_pima_test_scaled[:100]  # 100 amostras
shap_values_pima = explainer_pima.shap_values(X_pima_test_sample)

print(f"SHAP values calculados para {len(X_pima_test_sample)} amostras do Pima")

# ============================================================================
# 6. ANALISE SHAP - BRFSS
# ============================================================================

print("\n" + "=" * 80)
print("[6/6] Criando explicacoes SHAP para BRFSS...")
print("=" * 80)

# Criar explainer
explainer_brfss = shap.TreeExplainer(modelo_brfss)

# Calcular SHAP values
X_brfss_test_sample = X_brfss_test_scaled[:100]
shap_values_brfss = explainer_brfss.shap_values(X_brfss_test_sample)

print(f"SHAP values calculados para {len(X_brfss_test_sample)} amostras do BRFSS")

# ============================================================================
# 7. VISUALIZACOES SHAP
# ============================================================================

print("\n" + "=" * 80)
print("Gerando visualizacoes SHAP...")
print("=" * 80)

# Criar figura grande com múltiplos plots
fig = plt.figure(figsize=(20, 14))

# Plot 1: Summary Plot - Pima Indians
print("\n1. Summary Plot - Pima Indians (importancia global)")
plt.subplot(3, 2, 1)
shap.summary_plot(
    shap_values_pima,
    X_pima_test_sample,
    feature_names=X_pima.columns.tolist(),
    show=False,
    max_display=10
)
plt.title('Features Mais Importantes - Pima Indians (Clinico)', fontsize=12, fontweight='bold')

# Plot 2: Summary Plot - BRFSS
print("2. Summary Plot - BRFSS (importancia global)")
plt.subplot(3, 2, 2)
shap.summary_plot(
    shap_values_brfss,
    X_brfss_test_sample,
    feature_names=X_brfss.columns.tolist(),
    show=False,
    max_display=10
)
plt.title('Features Mais Importantes - BRFSS (Comportamental)', fontsize=12, fontweight='bold')

# Plot 3: Bar Plot - Importancia Media Pima
print("3. Bar Plot - Importancia absoluta Pima")
plt.subplot(3, 2, 3)
shap_importance_pima = np.abs(shap_values_pima).mean(axis=0)
feature_importance_pima = pd.DataFrame({
    'feature': X_pima.columns,
    'importance': shap_importance_pima
}).sort_values('importance', ascending=True).tail(10)

plt.barh(feature_importance_pima['feature'], feature_importance_pima['importance'], color='steelblue')
plt.xlabel('Mean |SHAP value|', fontweight='bold')
plt.title('Top 10 Features - Pima Indians', fontsize=11, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# Plot 4: Bar Plot - Importancia Media BRFSS
print("4. Bar Plot - Importancia absoluta BRFSS")
plt.subplot(3, 2, 4)
shap_importance_brfss = np.abs(shap_values_brfss).mean(axis=0)
feature_importance_brfss = pd.DataFrame({
    'feature': X_brfss.columns,
    'importance': shap_importance_brfss
}).sort_values('importance', ascending=True).tail(10)

plt.barh(feature_importance_brfss['feature'], feature_importance_brfss['importance'], color='coral')
plt.xlabel('Mean |SHAP value|', fontweight='bold')
plt.title('Top 10 Features - BRFSS', fontsize=11, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# Plot 5: Dependence Plot - Feature mais importante Pima
print("5. Dependence Plot - Feature principal Pima")
plt.subplot(3, 2, 5)
top_feature_pima_idx = np.abs(shap_values_pima).mean(axis=0).argmax()
top_feature_pima = X_pima.columns[top_feature_pima_idx]

shap.dependence_plot(
    top_feature_pima_idx,
    shap_values_pima,
    X_pima_test_sample,
    feature_names=X_pima.columns.tolist(),
    show=False
)
plt.title(f'Relacao: {top_feature_pima} vs Predicao (Pima)', fontsize=11, fontweight='bold')

# Plot 6: Dependence Plot - Feature mais importante BRFSS
print("6. Dependence Plot - Feature principal BRFSS")
plt.subplot(3, 2, 6)
top_feature_brfss_idx = np.abs(shap_values_brfss).mean(axis=0).argmax()
top_feature_brfss = X_brfss.columns[top_feature_brfss_idx]

shap.dependence_plot(
    top_feature_brfss_idx,
    shap_values_brfss,
    X_brfss_test_sample,
    feature_names=X_brfss.columns.tolist(),
    show=False
)
plt.title(f'Relacao: {top_feature_brfss} vs Predicao (BRFSS)', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('shap_analise_comparativa.png', dpi=300, bbox_inches='tight')
print("\nOK - Graficos SHAP salvos em: shap_analise_comparativa.png")

# ============================================================================
# 8. EXEMPLOS DE PREDICOES INDIVIDUAIS
# ============================================================================

print("\n" + "=" * 80)
print("EXEMPLOS DE PREDICOES INDIVIDUAIS (Waterfall Plots)")
print("=" * 80)

# Criar figura para exemplos individuais
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

# Pima: Exemplo positivo (diabetes)
idx_pos_pima = np.where(y_pima_test.iloc[:100] == 1)[0][0]
print(f"\n1. Pima - Caso POSITIVO (diabetes), indice {idx_pos_pima}")

plt.sca(axes[0, 0])
shap.plots._waterfall.waterfall_legacy(
    explainer_pima.expected_value,
    shap_values_pima[idx_pos_pima],
    X_pima_test_sample[idx_pos_pima],
    feature_names=X_pima.columns.tolist(),
    max_display=10,
    show=False
)
axes[0, 0].set_title('Pima - Caso COM Diabetes', fontsize=12, fontweight='bold')

# Pima: Exemplo negativo (sem diabetes)
idx_neg_pima = np.where(y_pima_test.iloc[:100] == 0)[0][0]
print(f"2. Pima - Caso NEGATIVO (sem diabetes), indice {idx_neg_pima}")

plt.sca(axes[0, 1])
shap.plots._waterfall.waterfall_legacy(
    explainer_pima.expected_value,
    shap_values_pima[idx_neg_pima],
    X_pima_test_sample[idx_neg_pima],
    feature_names=X_pima.columns.tolist(),
    max_display=10,
    show=False
)
axes[0, 1].set_title('Pima - Caso SEM Diabetes', fontsize=12, fontweight='bold')

# BRFSS: Exemplo positivo
idx_pos_brfss = np.where(y_brfss_test.iloc[:100] == 1)[0][0]
print(f"3. BRFSS - Caso POSITIVO (diabetes), indice {idx_pos_brfss}")

plt.sca(axes[1, 0])
shap.plots._waterfall.waterfall_legacy(
    explainer_brfss.expected_value,
    shap_values_brfss[idx_pos_brfss],
    X_brfss_test_sample[idx_pos_brfss],
    feature_names=X_brfss.columns.tolist(),
    max_display=10,
    show=False
)
axes[1, 0].set_title('BRFSS - Caso COM Diabetes', fontsize=12, fontweight='bold')

# BRFSS: Exemplo negativo
idx_neg_brfss = np.where(y_brfss_test.iloc[:100] == 0)[0][0]
print(f"4. BRFSS - Caso NEGATIVO (sem diabetes), indice {idx_neg_brfss}")

plt.sca(axes[1, 1])
shap.plots._waterfall.waterfall_legacy(
    explainer_brfss.expected_value,
    shap_values_brfss[idx_neg_brfss],
    X_brfss_test_sample[idx_neg_brfss],
    feature_names=X_brfss.columns.tolist(),
    max_display=10,
    show=False
)
axes[1, 1].set_title('BRFSS - Caso SEM Diabetes', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('shap_exemplos_individuais.png', dpi=300, bbox_inches='tight')
print("\nOK - Exemplos individuais salvos em: shap_exemplos_individuais.png")

# ============================================================================
# 9. RELATORIO INTERPRETATIVO
# ============================================================================

print("\n" + "=" * 80)
print("RELATORIO DE EXPLICABILIDADE")
print("=" * 80)

print(f"""
TOP 3 FEATURES MAIS IMPORTANTES:

PIMA INDIANS (Clinico):
  1. {feature_importance_pima.iloc[-1]['feature']}: {feature_importance_pima.iloc[-1]['importance']:.4f}
  2. {feature_importance_pima.iloc[-2]['feature']}: {feature_importance_pima.iloc[-2]['importance']:.4f}
  3. {feature_importance_pima.iloc[-3]['feature']}: {feature_importance_pima.iloc[-3]['importance']:.4f}

BRFSS (Comportamental):
  1. {feature_importance_brfss.iloc[-1]['feature']}: {feature_importance_brfss.iloc[-1]['importance']:.4f}
  2. {feature_importance_brfss.iloc[-2]['feature']}: {feature_importance_brfss.iloc[-2]['importance']:.4f}
  3. {feature_importance_brfss.iloc[-3]['feature']}: {feature_importance_brfss.iloc[-3]['importance']:.4f}

INTERPRETACAO:

1. PIMA INDIANS (Features Clinicas):
   - Features laboratoriais (glucose, insulin) tem ALTO poder preditivo
   - Valores individuais impactam MUITO a predicao
   - Explicacoes sao DIRETAS: "glucose alto = risco alto"

2. BRFSS (Features Comportamentais):
   - Multiplas features contribuem MODERADAMENTE
   - Nao ha uma feature "matadora" individual
   - Explicacoes sao COMPOSTAS: combinacao de fatores

CONCLUSAO:
- Features CLINICAS sao mais interpretaveis e preditivas
- Features COMPORTAMENTAIS requerem combinacao de multiplos fatores
- Modelo clinico e mais "explicavel" para pacientes e medicos
""")

print("=" * 80)
print("ANALISE SHAP COMPLETA!")
print("=" * 80)
print("\nArquivos gerados:")
print("  1. shap_analise_comparativa.png - Importancia global")
print("  2. shap_exemplos_individuais.png - Casos reais explicados")
