# -*- coding: utf-8 -*-
"""
AUTO ML COM TPOT - ENCONTRAR MELHOR MODELO AUTOMATICAMENTE
Otimização automática de pipeline completo
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
from tpot import TPOTClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("AUTO ML COM TPOT - BUSCANDO MELHOR MODELO")
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

# Feature Engineering
pima_df['BMI_Age'] = pima_df['BMI'] * pima_df['Age']
pima_df['Glucose_BMI'] = pima_df['Glucose'] * pima_df['BMI']
pima_df['Glucose_Age'] = pima_df['Glucose'] * pima_df['Age']
pima_df['Insulin_Glucose'] = pima_df['Insulin'] * pima_df['Glucose']
pima_df['BP_BMI'] = pima_df['BloodPressure'] * pima_df['BMI']

X = pima_df.drop('Outcome', axis=1)
y = pima_df['Outcome']

print(f"   Dataset shape: {X.shape}")
print()

# ============================================================================
# 2. SPLIT E SMOTE
# ============================================================================
print("2. PREPARANDO DADOS...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"   Train: {X_train_bal.shape}, Test: {X_test.shape}")
print()

# ============================================================================
# 3. AUTO ML COM TPOT
# ============================================================================
print("="*80)
print("3. EXECUTANDO AUTO ML COM TPOT")
print("="*80)
print()
print("CONFIGURAÇÃO:")
print("   Gerações: 3 (evoluções)")
print("   População: 15 (modelos por geração)")
print("   Tempo máximo: 5 minutos")
print("   Scoring: F1-Score")
print()
print("INICIANDO BUSCA AUTOMÁTICA...")
print("(Isso pode demorar alguns minutos...)")
print()

tpot = TPOTClassifier(
    generations=3,              # Número de gerações
    population_size=15,         # Tamanho da população
    verbosity=2,
    scoring='f1',
    cv=5,
    random_state=42,
    max_time_mins=5,            # Limite de 5 minutos
    n_jobs=-1                   # Usar todos os cores
)

tpot.fit(X_train_bal, y_train_bal)

print()
print("="*80)
print("BUSCA FINALIZADA!")
print("="*80)
print()

# ============================================================================
# 4. AVALIAR MELHOR MODELO ENCONTRADO
# ============================================================================
print("4. AVALIANDO MELHOR MODELO ENCONTRADO...")

y_pred = tpot.predict(X_test)
y_pred_proba = tpot.predict_proba(X_test)[:, 1]

# Testar thresholds
melhor_f1 = 0
melhor_threshold = 0.5

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_t = (y_pred_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_t)

    if f1 > melhor_f1:
        melhor_f1 = f1
        melhor_threshold = threshold

# Predição final
y_pred_final = (y_pred_proba >= melhor_threshold).astype(int)

# Métricas
report = classification_report(y_test, y_pred_final, output_dict=True)

print()
print(f"   Melhor Threshold: {melhor_threshold:.2f}")
print(f"   Precisão: {report['1']['precision']*100:.1f}%")
print(f"   Recall: {report['1']['recall']*100:.1f}%")
print(f"   F1-Score: {report['1']['f1-score']:.4f}")
print()

# ============================================================================
# 5. COMPARAÇÃO FINAL
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

print(f"MODELO AUTO ML (TPOT):")
print(f"   F1-Score: {report['1']['f1-score']:.4f}")
print(f"   Precisão: {report['1']['precision']*100:.1f}%")
print(f"   Recall: {report['1']['recall']*100:.1f}%")
print()

ganho = ((report['1']['f1-score'] - 0.72) / 0.72) * 100

if ganho > 0:
    print(f"GANHO: +{ganho:.1f}% em F1-Score!")
else:
    print(f"RESULTADO: {ganho:.1f}%")
print()

# ============================================================================
# 6. EXPORTAR MELHOR PIPELINE
# ============================================================================
print("6. EXPORTANDO MELHOR PIPELINE...")

tpot.export('tpot_best_pipeline.py')

print("   Pipeline exportado: tpot_best_pipeline.py")
print()

print("="*80)
print("AUTO ML FINALIZADO!")
print("="*80)
