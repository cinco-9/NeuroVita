# -*- coding: utf-8 -*-
"""
Modelo Avançado: XGBoost + SMOTE para Predição de Diabetes
Dataset: Diabetes Health Indicators (BRFSS 2015)
Técnicas: Balanceamento com SMOTE + Gradient Boosting + Otimização de Hiperparâmetros
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
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
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MODELO AVANCADO: XGBoost + SMOTE")
print("=" * 80)

# ============================================================================
# 1. CARREGAR E PREPARAR OS DADOS
# ============================================================================

print("\n[1/6] Carregando dados...")
df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
print(f"Dataset: {df.shape[0]:,} registros, {df.shape[1]} colunas")

# Separar features e target
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

print(f"Classes originais: Sem diabetes = {(y==0).sum():,} | Com diabetes = {(y==1).sum():,}")
print(f"Desbalanceamento: {(y==0).sum() / (y==1).sum():.1f}:1")

# Split treino/teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nConjunto de treino: {X_train.shape[0]:,} amostras")
print(f"Conjunto de teste: {X_test.shape[0]:,} amostras")

# ============================================================================
# 2. APLICAR SMOTE PARA BALANCEAMENTO
# ============================================================================

print("\n" + "=" * 80)
print("[2/6] Aplicando SMOTE (Synthetic Minority Over-sampling Technique)...")
print("=" * 80)

# SMOTE cria exemplos sintéticos da classe minoritária
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nAntes do SMOTE:")
print(f"  - Sem diabetes: {(y_train==0).sum():,}")
print(f"  - Com diabetes: {(y_train==1).sum():,}")
print(f"\nDepois do SMOTE:")
print(f"  - Sem diabetes: {(y_train_smote==0).sum():,}")
print(f"  - Com diabetes: {(y_train_smote==1).sum():,}")
print(f"\nAumento no conjunto de treino: {len(y_train_smote) - len(y_train):,} amostras sintéticas")

# ============================================================================
# 3. TREINAR MODELO XGBOOST BÁSICO
# ============================================================================

print("\n" + "=" * 80)
print("[3/6] Treinando XGBoost (configuração inicial)...")
print("=" * 80)

# Modelo XGBoost com parâmetros padrão
xgb_basico = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

xgb_basico.fit(X_train_smote, y_train_smote)
print("Modelo XGBoost básico treinado!")

# Predições
y_pred_xgb_basico = xgb_basico.predict(X_test)
y_proba_xgb_basico = xgb_basico.predict_proba(X_test)[:, 1]

# Métricas
acc_xgb_basico = accuracy_score(y_test, y_pred_xgb_basico)
prec_xgb_basico = precision_score(y_test, y_pred_xgb_basico)
rec_xgb_basico = recall_score(y_test, y_pred_xgb_basico)
f1_xgb_basico = f1_score(y_test, y_pred_xgb_basico)
auc_xgb_basico = roc_auc_score(y_test, y_proba_xgb_basico)

print(f"\nMETRICAS - XGBoost Basico:")
print(f"  - Acuracia: {acc_xgb_basico:.4f} ({acc_xgb_basico*100:.2f}%)")
print(f"  - Precisao: {prec_xgb_basico:.4f}")
print(f"  - Recall: {rec_xgb_basico:.4f}")
print(f"  - F1-Score: {f1_xgb_basico:.4f}")
print(f"  - ROC-AUC: {auc_xgb_basico:.4f}")

# ============================================================================
# 4. OTIMIZAÇÃO DE HIPERPARÂMETROS COM GRID SEARCH
# ============================================================================

print("\n" + "=" * 80)
print("[4/6] Otimizando hiperparametros com GridSearchCV...")
print("=" * 80)
print("(Isso pode levar alguns minutos...)\n")

# Grade de hiperparâmetros para testar
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# GridSearchCV com validação cruzada 3-fold
xgb_model = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_smote, y_train_smote)

print(f"\nMelhores hiperparametros encontrados:")
for param, value in grid_search.best_params_.items():
    print(f"  - {param}: {value}")

# Modelo otimizado
xgb_otimizado = grid_search.best_estimator_

# Predições com modelo otimizado
y_pred_xgb_otim = xgb_otimizado.predict(X_test)
y_proba_xgb_otim = xgb_otimizado.predict_proba(X_test)[:, 1]

# Métricas
acc_xgb_otim = accuracy_score(y_test, y_pred_xgb_otim)
prec_xgb_otim = precision_score(y_test, y_pred_xgb_otim)
rec_xgb_otim = recall_score(y_test, y_pred_xgb_otim)
f1_xgb_otim = f1_score(y_test, y_pred_xgb_otim)
auc_xgb_otim = roc_auc_score(y_test, y_proba_xgb_otim)

print(f"\nMETRICAS - XGBoost Otimizado:")
print(f"  - Acuracia: {acc_xgb_otim:.4f} ({acc_xgb_otim*100:.2f}%)")
print(f"  - Precisao: {prec_xgb_otim:.4f}")
print(f"  - Recall: {rec_xgb_otim:.4f}")
print(f"  - F1-Score: {f1_xgb_otim:.4f}")
print(f"  - ROC-AUC: {auc_xgb_otim:.4f}")

# Matriz de confusão
cm_xgb = confusion_matrix(y_test, y_pred_xgb_otim)
print(f"\nMATRIZ DE CONFUSAO:")
print(cm_xgb)
print(f"  - Verdadeiros Negativos: {cm_xgb[0,0]:,}")
print(f"  - Falsos Positivos: {cm_xgb[0,1]:,}")
print(f"  - Falsos Negativos: {cm_xgb[1,0]:,}")
print(f"  - Verdadeiros Positivos: {cm_xgb[1,1]:,}")

# ============================================================================
# 5. COMPARAÇÃO COM MODELOS ANTERIORES (REGRESSÃO LOGÍSTICA)
# ============================================================================

print("\n" + "=" * 80)
print("[5/6] Comparando com Regressao Logistica...")
print("=" * 80)

# Treinar Regressão Logística com SMOTE para comparação justa
lr_smote = LogisticRegression(max_iter=1000, random_state=42)
lr_smote.fit(X_train_smote, y_train_smote)

y_pred_lr = lr_smote.predict(X_test)
y_proba_lr = lr_smote.predict_proba(X_test)[:, 1]

acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
rec_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_proba_lr)

# Tabela comparativa
comparacao = pd.DataFrame({
    'Metrica': ['Acuracia', 'Precisao', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Reg. Logistica + SMOTE': [acc_lr, prec_lr, rec_lr, f1_lr, auc_lr],
    'XGBoost Basico + SMOTE': [acc_xgb_basico, prec_xgb_basico, rec_xgb_basico, f1_xgb_basico, auc_xgb_basico],
    'XGBoost Otimizado + SMOTE': [acc_xgb_otim, prec_xgb_otim, rec_xgb_otim, f1_xgb_otim, auc_xgb_otim]
})

print("\n" + "=" * 80)
print("COMPARACAO FINAL DE TODOS OS MODELOS")
print("=" * 80)
print("\n" + comparacao.to_string(index=False))

# Calcular melhorias
melhoria_acc = ((acc_xgb_otim - acc_lr) / acc_lr * 100)
melhoria_f1 = ((f1_xgb_otim - f1_lr) / f1_lr * 100)
melhoria_auc = ((auc_xgb_otim - auc_lr) / auc_lr * 100)

print(f"\nMELHORIA DO XGBoost OTIMIZADO vs REGRESSAO LOGISTICA:")
print(f"  - Acuracia: {melhoria_acc:+.2f}%")
print(f"  - F1-Score: {melhoria_f1:+.2f}%")
print(f"  - ROC-AUC: {melhoria_auc:+.2f}%")

# ============================================================================
# 6. FEATURE IMPORTANCE E VISUALIZAÇÕES
# ============================================================================

print("\n" + "=" * 80)
print("[6/6] Gerando visualizacoes e analise de features...")
print("=" * 80)

# Feature Importance do XGBoost
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_otimizado.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTOP 10 FEATURES MAIS IMPORTANTES (XGBoost):")
print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# VISUALIZAÇÕES
# ============================================================================

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Comparação de métricas
ax1 = fig.add_subplot(gs[0, :2])
metricas_plot = comparacao.set_index('Metrica').T
x = np.arange(len(metricas_plot.columns))
width = 0.25

for i, (idx, row) in enumerate(metricas_plot.iterrows()):
    ax1.bar(x + i*width, row.values, width, label=idx, alpha=0.8)

ax1.set_xlabel('Metricas', fontweight='bold')
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_title('Comparacao de Todos os Modelos', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width)
ax1.set_xticklabels(metricas_plot.columns, rotation=0)
ax1.legend(loc='lower right', fontsize=8)
ax1.set_ylim(0, 1)
ax1.grid(axis='y', alpha=0.3)

# 2. Feature Importance (Top 15)
ax2 = fig.add_subplot(gs[0, 2])
top_15 = feature_importance.head(15).sort_values('Importance')
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_15)))
ax2.barh(top_15['Feature'], top_15['Importance'], color=colors)
ax2.set_xlabel('Importancia', fontweight='bold')
ax2.set_title('Top 15 Features (XGBoost)', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# 3. Curvas ROC
ax3 = fig.add_subplot(gs[1, 0])
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_xgb_bas, tpr_xgb_bas, _ = roc_curve(y_test, y_proba_xgb_basico)
fpr_xgb_otim, tpr_xgb_otim, _ = roc_curve(y_test, y_proba_xgb_otim)

ax3.plot(fpr_lr, tpr_lr, label=f'Reg. Log. (AUC={auc_lr:.3f})', linewidth=2)
ax3.plot(fpr_xgb_bas, tpr_xgb_bas, label=f'XGB Basico (AUC={auc_xgb_basico:.3f})', linewidth=2)
ax3.plot(fpr_xgb_otim, tpr_xgb_otim, label=f'XGB Otimizado (AUC={auc_xgb_otim:.3f})', linewidth=2.5, linestyle='--')
ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Aleatorio')
ax3.set_xlabel('Taxa de Falsos Positivos', fontweight='bold')
ax3.set_ylabel('Taxa de Verdadeiros Positivos', fontweight='bold')
ax3.set_title('Curvas ROC', fontsize=12, fontweight='bold')
ax3.legend(loc='lower right', fontsize=8)
ax3.grid(alpha=0.3)

# 4. Matriz de Confusão - XGBoost Otimizado
ax4 = fig.add_subplot(gs[1, 1])
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=ax4, cbar_kws={'label': 'Contagem'})
ax4.set_title('Matriz de Confusao - XGBoost Otimizado', fontsize=12, fontweight='bold')
ax4.set_ylabel('Valor Real', fontweight='bold')
ax4.set_xlabel('Valor Predito', fontweight='bold')
ax4.set_xticklabels(['Sem Diabetes', 'Com Diabetes'])
ax4.set_yticklabels(['Sem Diabetes', 'Com Diabetes'], rotation=0)

# 5. Distribuição de probabilidades preditas
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(y_proba_xgb_otim[y_test == 0], bins=50, alpha=0.6, label='Sem Diabetes', color='green')
ax5.hist(y_proba_xgb_otim[y_test == 1], bins=50, alpha=0.6, label='Com Diabetes', color='red')
ax5.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
ax5.set_xlabel('Probabilidade Predita', fontweight='bold')
ax5.set_ylabel('Frequencia', fontweight='bold')
ax5.set_title('Distribuicao de Probabilidades', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3)

# 6. Comparação F1-Score
ax6 = fig.add_subplot(gs[2, 0])
modelos = ['Reg. Log.\n+ SMOTE', 'XGBoost\nBasico', 'XGBoost\nOtimizado']
f1_scores = [f1_lr, f1_xgb_basico, f1_xgb_otim]
colors_bar = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax6.bar(modelos, f1_scores, color=colors_bar, alpha=0.8, edgecolor='black')
ax6.set_ylabel('F1-Score', fontweight='bold')
ax6.set_title('Comparacao F1-Score', fontsize=12, fontweight='bold')
ax6.set_ylim(0, 1)
ax6.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

# 7. Comparação Recall
ax7 = fig.add_subplot(gs[2, 1])
recalls = [rec_lr, rec_xgb_basico, rec_xgb_otim]
bars = ax7.bar(modelos, recalls, color=colors_bar, alpha=0.8, edgecolor='black')
ax7.set_ylabel('Recall', fontweight='bold')
ax7.set_title('Comparacao Recall (Sensibilidade)', fontsize=12, fontweight='bold')
ax7.set_ylim(0, 1)
ax7.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

# 8. Comparação ROC-AUC
ax8 = fig.add_subplot(gs[2, 2])
aucs = [auc_lr, auc_xgb_basico, auc_xgb_otim]
bars = ax8.bar(modelos, aucs, color=colors_bar, alpha=0.8, edgecolor='black')
ax8.set_ylabel('ROC-AUC', fontweight='bold')
ax8.set_title('Comparacao ROC-AUC', fontsize=12, fontweight='bold')
ax8.set_ylim(0, 1)
ax8.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

plt.savefig('resultados_xgboost_smote.png', dpi=300, bbox_inches='tight')
print("\nOK - Graficos salvos em: resultados_xgboost_smote.png")

# ============================================================================
# RELATÓRIO FINAL
# ============================================================================

print("\n" + "=" * 80)
print("RELATORIO FINAL - MODELO XGBOOST + SMOTE")
print("=" * 80)

print(f"""
MELHOR MODELO: XGBoost Otimizado + SMOTE

DESEMPENHO:
  - Acuracia: {acc_xgb_otim:.4f} ({acc_xgb_otim*100:.2f}%)
  - Precisao: {prec_xgb_otim:.4f}
  - Recall: {rec_xgb_otim:.4f}
  - F1-Score: {f1_xgb_otim:.4f}
  - ROC-AUC: {auc_xgb_otim:.4f}

MATRIZ DE CONFUSAO:
  - Verdadeiros Negativos: {cm_xgb[0,0]:,} (sem diabetes corretamente identificados)
  - Falsos Positivos: {cm_xgb[0,1]:,} (alarmes falsos)
  - Falsos Negativos: {cm_xgb[1,0]:,} (casos de diabetes nao detectados)
  - Verdadeiros Positivos: {cm_xgb[1,1]:,} (casos de diabetes corretamente identificados)

INTERPRETACAO CLINICA:
  - De cada 100 pessoas COM diabetes, o modelo detecta {rec_xgb_otim*100:.0f}
  - De cada 100 alertas do modelo, {prec_xgb_otim*100:.0f} sao casos reais de diabetes

PROXIMOS PASSOS PARA O TCC:
  1. Validacao cruzada (k-fold) para estabilidade
  2. Analise SHAP para interpretabilidade
  3. Ajuste de threshold para balanco precisao/recall
  4. Teste com outros algoritmos (LightGBM, CatBoost)
  5. Analise de casos mal classificados
  6. Desenvolvimento da interface do agente de IA
""")

print("=" * 80)
print("ANALISE COMPLETA! Modelo pronto para seu TCC.")
print("=" * 80)
