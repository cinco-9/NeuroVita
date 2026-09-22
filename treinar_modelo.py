# -*- coding: utf-8 -*-
"""
Passo 2: Treinar modelo de Regressão Logística para prever risco de diabetes
Dataset: Diabetes Health Indicators (BRFSS 2015)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 1. CARREGAR E PREPARAR OS DADOS
# ============================================================================

print("=" * 70)
print("ETAPA 1: CARREGANDO E PREPARANDO DADOS")
print("=" * 70)

# Carregar dataset
df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
print(f"Dataset carregado: {df.shape[0]} registros, {df.shape[1]} colunas\n")

# Separar features (X) e target (y)
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

print("Distribuição das classes:")
print(y.value_counts())
print(f"\nProporção de casos positivos (diabetes): {y.mean():.2%}\n")

# Dividir em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Mantém a proporção das classes
)

print(f"Conjunto de treino: {X_train.shape[0]} amostras")
print(f"Conjunto de teste: {X_test.shape[0]} amostras\n")

# ============================================================================
# 2. TREINAR MODELO BÁSICO (SEM BALANCEAMENTO)
# ============================================================================

print("=" * 70)
print("ETAPA 2: TREINANDO MODELO BÁSICO DE REGRESSÃO LOGÍSTICA")
print("=" * 70)

# Modelo básico
modelo_basico = LogisticRegression(max_iter=1000, random_state=42)
modelo_basico.fit(X_train, y_train)

# Fazer predições
y_pred_basico = modelo_basico.predict(X_test)
y_proba_basico = modelo_basico.predict_proba(X_test)[:, 1]

print("Modelo básico treinado com sucesso!\n")

# ============================================================================
# 3. AVALIAR MODELO BÁSICO
# ============================================================================

print("=" * 70)
print("ETAPA 3: AVALIAÇÃO DO MODELO BÁSICO")
print("=" * 70)

# Métricas principais
acc_basico = accuracy_score(y_test, y_pred_basico)
precision_basico = precision_score(y_test, y_pred_basico)
recall_basico = recall_score(y_test, y_pred_basico)
f1_basico = f1_score(y_test, y_pred_basico)
roc_auc_basico = roc_auc_score(y_test, y_proba_basico)

print("\nMETRICAS DO MODELO BASICO:")
print(f"  - Acuracia: {acc_basico:.4f} ({acc_basico*100:.2f}%)")
print(f"  - Precisao: {precision_basico:.4f}")
print(f"  - Recall (Sensibilidade): {recall_basico:.4f}")
print(f"  - F1-Score: {f1_basico:.4f}")
print(f"  - ROC-AUC: {roc_auc_basico:.4f}\n")

# Matriz de confusão
print("MATRIZ DE CONFUSAO:")
cm_basico = confusion_matrix(y_test, y_pred_basico)
print(cm_basico)
print(f"\nVerdadeiros Negativos: {cm_basico[0,0]}")
print(f"Falsos Positivos: {cm_basico[0,1]}")
print(f"Falsos Negativos: {cm_basico[1,0]}")
print(f"Verdadeiros Positivos: {cm_basico[1,1]}\n")

# Relatório completo
print("RELATORIO DE CLASSIFICACAO:")
print(classification_report(y_test, y_pred_basico,
                          target_names=['Sem Diabetes', 'Com Diabetes']))

# ============================================================================
# 4. TREINAR MODELO COM BALANCEAMENTO DE CLASSES
# ============================================================================

print("\n" + "=" * 70)
print("ETAPA 4: MODELO COM BALANCEAMENTO DE CLASSES")
print("=" * 70)

# Modelo com class_weight='balanced'
modelo_balanceado = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'  # Ajusta pesos automaticamente
)
modelo_balanceado.fit(X_train, y_train)

# Predições
y_pred_balanceado = modelo_balanceado.predict(X_test)
y_proba_balanceado = modelo_balanceado.predict_proba(X_test)[:, 1]

# Métricas
acc_bal = accuracy_score(y_test, y_pred_balanceado)
precision_bal = precision_score(y_test, y_pred_balanceado)
recall_bal = recall_score(y_test, y_pred_balanceado)
f1_bal = f1_score(y_test, y_pred_balanceado)
roc_auc_bal = roc_auc_score(y_test, y_proba_balanceado)

print("\nMETRICAS DO MODELO BALANCEADO:")
print(f"  - Acuracia: {acc_bal:.4f} ({acc_bal*100:.2f}%)")
print(f"  - Precisao: {precision_bal:.4f}")
print(f"  - Recall (Sensibilidade): {recall_bal:.4f}")
print(f"  - F1-Score: {f1_bal:.4f}")
print(f"  - ROC-AUC: {roc_auc_bal:.4f}\n")

# Matriz de confusão
print("MATRIZ DE CONFUSAO:")
cm_bal = confusion_matrix(y_test, y_pred_balanceado)
print(cm_bal)
print(f"\nVerdadeiros Negativos: {cm_bal[0,0]}")
print(f"Falsos Positivos: {cm_bal[0,1]}")
print(f"Falsos Negativos: {cm_bal[1,0]}")
print(f"Verdadeiros Positivos: {cm_bal[1,1]}\n")

# ============================================================================
# 5. COMPARAÇÃO E VISUALIZAÇÕES
# ============================================================================

print("=" * 70)
print("COMPARAÇÃO DOS MODELOS")
print("=" * 70)

comparacao = pd.DataFrame({
    'Métrica': ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Modelo Básico': [acc_basico, precision_basico, recall_basico, f1_basico, roc_auc_basico],
    'Modelo Balanceado': [acc_bal, precision_bal, recall_bal, f1_bal, roc_auc_bal]
})
comparacao['Diferença (%)'] = ((comparacao['Modelo Balanceado'] - comparacao['Modelo Básico']) /
                                comparacao['Modelo Básico'] * 100).round(2)

print("\n" + comparacao.to_string(index=False))

# Visualizar importância das features
print("\n" + "=" * 70)
print("FEATURES MAIS IMPORTANTES (TOP 10)")
print("=" * 70)

# Coeficientes do modelo balanceado (valores absolutos indicam importância)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coeficiente': modelo_balanceado.coef_[0]
})
feature_importance['Importância_Abs'] = np.abs(feature_importance['Coeficiente'])
feature_importance = feature_importance.sort_values('Importância_Abs', ascending=False)

print("\n" + feature_importance.head(10).to_string(index=False))

# ============================================================================
# 6. SALVAR VISUALIZAÇÕES
# ============================================================================

print("\n" + "=" * 70)
print("GERANDO VISUALIZAÇÕES")
print("=" * 70)

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Figura com 3 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Comparação de métricas
ax1 = axes[0, 0]
metricas_plot = comparacao.set_index('Métrica')[['Modelo Básico', 'Modelo Balanceado']]
metricas_plot.plot(kind='bar', ax=ax1, rot=0)
ax1.set_title('Comparação de Métricas', fontsize=14, fontweight='bold')
ax1.set_ylabel('Score')
ax1.set_ylim(0, 1)
ax1.legend(loc='lower right')
ax1.grid(axis='y', alpha=0.3)

# 2. Top 10 features mais importantes
ax2 = axes[0, 1]
top_features = feature_importance.head(10).sort_values('Importância_Abs')
ax2.barh(top_features['Feature'], top_features['Importância_Abs'], color='steelblue')
ax2.set_title('Top 10 Features Mais Importantes', fontsize=14, fontweight='bold')
ax2.set_xlabel('Importância (|Coeficiente|)')
ax2.grid(axis='x', alpha=0.3)

# 3. Curva ROC
ax3 = axes[1, 0]
fpr_basico, tpr_basico, _ = roc_curve(y_test, y_proba_basico)
fpr_bal, tpr_bal, _ = roc_curve(y_test, y_proba_balanceado)

ax3.plot(fpr_basico, tpr_basico, label=f'Modelo Básico (AUC = {roc_auc_basico:.3f})', linewidth=2)
ax3.plot(fpr_bal, tpr_bal, label=f'Modelo Balanceado (AUC = {roc_auc_bal:.3f})', linewidth=2)
ax3.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.5)', linewidth=1)
ax3.set_xlabel('Taxa de Falsos Positivos')
ax3.set_ylabel('Taxa de Verdadeiros Positivos')
ax3.set_title('Curva ROC', fontsize=14, fontweight='bold')
ax3.legend(loc='lower right')
ax3.grid(alpha=0.3)

# 4. Matriz de confusão do modelo balanceado
ax4 = axes[1, 1]
sns.heatmap(cm_bal, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['Sem Diabetes', 'Com Diabetes'],
            yticklabels=['Sem Diabetes', 'Com Diabetes'])
ax4.set_title('Matriz de Confusão - Modelo Balanceado', fontsize=14, fontweight='bold')
ax4.set_ylabel('Valor Real')
ax4.set_xlabel('Valor Predito')

plt.tight_layout()
plt.savefig('resultados_modelo_diabetes.png', dpi=300, bbox_inches='tight')
print("OK - Graficos salvos em: resultados_modelo_diabetes.png")

# ============================================================================
# 7. SUGESTÕES DE MELHORIAS
# ============================================================================

print("\n" + "=" * 70)
print("SUGESTOES DE MELHORIAS PARA O TCC")
print("=" * 70)

print("""
1. **BALANCEAMENTO DE CLASSES** (ja implementado)
   - Usar class_weight='balanced' melhorou significativamente o Recall
   - Alternativas: SMOTE, Random Undersampling, Ensemble methods

2. **FEATURE ENGINEERING**
   - Criar interações entre features (ex: BMI × HighBP)
   - Normalizar/padronizar features numéricas (StandardScaler)
   - Criar bins para variáveis contínuas (ex: categorizar BMI)
   - Analisar correlações e remover features redundantes

3. **OTIMIZAÇÃO DE HIPERPARÂMETROS**
   - Usar GridSearchCV ou RandomizedSearchCV
   - Testar diferentes valores de C (regularização)
   - Experimentar penalidades L1 (Lasso) vs L2 (Ridge)

4. **MODELOS ALTERNATIVOS**
   - Random Forest
   - Gradient Boosting (XGBoost, LightGBM)
   - Support Vector Machines (SVM)
   - Redes Neurais (MLPClassifier)

5. **VALIDAÇÃO CRUZADA**
   - Usar cross-validation (k-fold) para avaliar estabilidade
   - Avaliar múltiplas métricas simultaneamente

6. **ANÁLISE DE ERROS**
   - Investigar falsos positivos e falsos negativos
   - Ajustar threshold de classificação conforme necessidade clínica
   - Analisar casos limítrofes (probabilidades próximas de 0.5)

7. **INTERPRETABILIDADE**
   - Usar SHAP values para explicar predições individuais
   - Análise de importância de features mais robusta
   - Criar visualizações para comunicar resultados a leigos

8. **FEATURES CLÍNICAS ADICIONAIS**
   - Se possível, adicionar dados de histórico familiar
   - Informações sobre medicamentos
   - Resultados de exames laboratoriais
""")

print("\n" + "=" * 70)
print("ANALISE COMPLETA! Proximos passos definidos acima.")
print("=" * 70)
