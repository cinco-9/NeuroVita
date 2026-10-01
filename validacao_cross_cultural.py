# -*- coding: utf-8 -*-
"""
VALIDAÇÃO CROSS-CULTURAL: BRASIL (VIGITEL 2023) x EUA (BRFSS 2015)

- Harmoniza as variáveis com definição equivalente nas duas pesquisas.
- Treina o mesmo pipeline XGBoost em cada país (60/20/20; limiar escolhido na validação).
- Avalia cada modelo no próprio país e no outro (transferência).
- Gera dados_vigitel/comparacao_cross_cultural.json, figuras PNG e ANALISE_CROSS_CULTURAL.md.

Pré-requisitos: python mapear_vigitel_completo.py e python baixar_brfss.py
Autor: David Reis | 2026
"""

import json
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import chi2_contingency
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

PASTA = 'dados_vigitel'
SEED = 42

# Variáveis com a mesma definição nas duas pesquisas
FEATURES = ['Age', 'Sex', 'BMI', 'HighBP', 'GenHlth']
NOMES = {'Age': 'Idade', 'Sex': 'Sexo (masc.)', 'BMI': 'IMC', 'HighBP': 'Pressão alta', 'GenHlth': 'Saúde autoavaliada'}

# Paleta validada (dataviz validate_palette.js, modo claro): todas as checagens PASS
COR = {'Brasil': '#2a78d6', 'EUA': '#eb6834'}
SUPERFICIE = '#fcfcfb'
TINTA = '#1a1a19'
TINTA_2 = '#5f5e58'
GRADE = '#e4e3dd'


def idade_para_faixa_brfss(anos):
    """Mesma codificação do BRFSS: 1 = 18-24, 2 = 25-29, ..., 13 = 80+."""
    anos = np.asarray(anos)
    return np.where(anos < 25, 1, np.minimum(13, (anos - 25) // 5 + 2))


FAIXAS = [('18-34', [1, 2, 3]), ('35-44', [4, 5]), ('45-54', [6, 7]), ('55-64', [8, 9]), ('65+', [10, 11, 12, 13])]


# ---------------------------------------------------------------------------
# 1. Dados harmonizados
# ---------------------------------------------------------------------------
br = pd.read_csv(f'{PASTA}/vigitel_2023_processado.csv')
br['AgeCat'] = idade_para_faixa_brfss(br['Age'])

# Dados originais do CDC (python baixar_brfss.py). A versão do Kaggle agrupa pré-diabetes com diabetes;
# aqui o desfecho é só diabetes diagnosticado, como no VIGITEL.
bruto = pd.read_csv('dados_brfss/brfss2015_colunas.csv')
bruto = bruto[bruto.DIABETE3.isin([1, 2, 3, 4]) & bruto._AGEG5YR.between(1, 13) & bruto._RFHYPE5.isin([1, 2])
              & bruto.GENHLTH.between(1, 5) & bruto.SMOKE100.isin([1, 2]) & bruto._TOTINDA.isin([1, 2])
              & bruto._FRTLT1.isin([1, 2]) & bruto._BMI5.notna()]
us = pd.DataFrame({
    'Diabetes': (bruto.DIABETE3 == 1).astype(int),  # 2 = só na gravidez, 3 = não, 4 = pré-diabetes -> 0
    'AgeCat': bruto._AGEG5YR.astype(int),
    'Sex': (bruto.SEX == 1).astype(int),
    'BMI': bruto._BMI5 / 100,
    'HighBP': (bruto._RFHYPE5 == 2).astype(int),
    'GenHlth': bruto.GENHLTH.astype(int),
    'Smoker': (bruto.SMOKE100 == 1).astype(int),
    'PhysActivity': (bruto._TOTINDA == 1).astype(int),
    'Fruits': (bruto._FRTLT1 == 1).astype(int),
    'peso': bruto._LLCPWT,
})
us = us[us.BMI.between(10, 60)].reset_index(drop=True)

paises = {'Brasil': br, 'EUA': us}
for df in paises.values():
    df['Age'] = df['AgeCat']  # os modelos usam a faixa etária no padrão BRFSS

print(f"Brasil: {len(br):,} | EUA: {len(us):,}")

# ---------------------------------------------------------------------------
# 2. Estatística descritiva
# ---------------------------------------------------------------------------
desc = {}
for nome, df in paises.items():
    d = {
        'n': int(len(df)),
        'prevalencia_diabetes': float(df['Diabetes'].mean()),
        'imc_medio': float(df['BMI'].mean()),
        'obesidade': float((df['BMI'] >= 30).mean()),
        'pressao_alta': float(df['HighBP'].mean()),
        'saude_ruim_ou_muito_ruim': float((df['GenHlth'] >= 4).mean()),
        'fumante*': float(df['Smoker'].mean()),
        'atividade_fisica*': float(df['PhysActivity'].mean()),
        'frutas*': float(df['Fruits'].mean()),
        'prevalencia_por_faixa': {
            f: float(df.loc[df['AgeCat'].isin(cats), 'Diabetes'].mean()) for f, cats in FAIXAS
        },
        'distribuicao_faixa': {
            f: float(df['AgeCat'].isin(cats).mean()) for f, cats in FAIXAS
        },
    }
    desc[nome] = d
for nome, df in paises.items():
    desc[nome]['prevalencia_diabetes_ponderada'] = float(np.average(df['Diabetes'], weights=df['peso']))
    desc[nome]['pressao_alta_ponderada'] = float(np.average(df['HighBP'], weights=df['peso']))

tabela = [[br['Diabetes'].sum(), len(br) - br['Diabetes'].sum()], [us['Diabetes'].sum(), len(us) - us['Diabetes'].sum()]]
p_prev = float(chi2_contingency(tabela)[1])

# Prevalência padronizada pela distribuição etária dos EUA (remove o efeito de o Brasil ter amostra de outra idade)
prev_br_padronizada = float(sum(
    desc['Brasil']['prevalencia_por_faixa'][f] * desc['EUA']['distribuicao_faixa'][f] for f, _ in FAIXAS
))

# ---------------------------------------------------------------------------
# 3. Modelos: mesmo pipeline nos dois países
# ---------------------------------------------------------------------------
def treinar(df):
    X, y = df[FEATURES], df['Diabetes']
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)
    modelo = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9,
                               colsample_bytree=0.9, eval_metric='logloss', random_state=SEED, n_jobs=-1)
    modelo.fit(X_tr, y_tr)
    p_va = modelo.predict_proba(X_va)[:, 1]
    grade = np.arange(0.05, 0.61, 0.01)
    limiar = float(round(grade[int(np.argmax([f1_score(y_va, p_va >= t) for t in grade]))], 2))
    return modelo, limiar, (X_te, y_te)


def avaliar(modelo, limiar, X, y):
    p = modelo.predict_proba(X)[:, 1]
    pred = (p >= limiar).astype(int)
    return {
        'auc': round(float(roc_auc_score(y, p)), 3),
        'f1': round(float(f1_score(y, pred)), 3),
        'precisao': round(float(precision_score(y, pred, zero_division=0)), 3),
        'recall': round(float(recall_score(y, pred)), 3),
    }


modelos, testes = {}, {}
for nome, df in paises.items():
    modelo, limiar, teste = treinar(df)
    modelos[nome] = (modelo, limiar)
    testes[nome] = teste
    print(f"Modelo {nome}: limiar {limiar}")

resultados = {}
for origem, (modelo, limiar) in modelos.items():
    for destino, (X_te, y_te) in testes.items():
        resultados[f'{origem}->{destino}'] = avaliar(modelo, limiar, X_te, y_te)
        print(f"  treinado {origem:6s} -> testado {destino:6s}: {resultados[f'{origem}->{destino}']}")

importancia = {}
for nome, (modelo, _) in modelos.items():
    gain = modelo.get_booster().get_score(importance_type='gain')
    total = sum(gain.get(f, 0) for f in FEATURES)
    importancia[nome] = {f: round(gain.get(f, 0) / total, 3) for f in FEATURES}

with open(f'{PASTA}/modelo_vigitel_xgboost.pkl', 'wb') as f:
    pickle.dump({'modelo': modelos['Brasil'][0], 'features': FEATURES, 'threshold': modelos['Brasil'][1],
                 'metricas': resultados['Brasil->Brasil']}, f)

# ---------------------------------------------------------------------------
# 4. Figuras (uma por arquivo, para uso direto no texto do TCC)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'figure.facecolor': SUPERFICIE, 'axes.facecolor': SUPERFICIE, 'savefig.facecolor': SUPERFICIE,
    'axes.edgecolor': GRADE, 'axes.labelcolor': TINTA_2, 'xtick.color': TINTA_2, 'ytick.color': TINTA_2,
    'text.color': TINTA, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRADE,
    'grid.linewidth': 0.8, 'axes.axisbelow': True, 'legend.frameon': False,
})


def barras_agrupadas(ax, categorias, series, fmt, legenda='upper left'):
    n = len(series)
    largura = 0.36
    x = np.arange(len(categorias))
    for i, (nome, valores) in enumerate(series.items()):
        pos = x + (i - (n - 1) / 2) * (largura + 0.02)
        barras = ax.bar(pos, valores, largura, color=COR[nome], label=nome, edgecolor=SUPERFICIE, linewidth=1)
        for b, v in zip(barras, valores):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt(v), ha='center', va='bottom',
                    fontsize=8.5, color=TINTA_2)
    ax.set_xticks(x)
    ax.set_xticklabels(categorias)
    ax.tick_params(length=0)
    ax.legend(loc=legenda, ncol=2)


# 4.1 Prevalência por faixa etária
fig, ax = plt.subplots(figsize=(7.5, 4))
faixas = [f for f, _ in FAIXAS]
barras_agrupadas(ax, faixas, {n: [desc[n]['prevalencia_por_faixa'][f] * 100 for f in faixas] for n in paises},
                 lambda v: f'{v:.1f}%')
ax.set_title('Prevalência de diabetes por faixa etária', loc='left', fontsize=12, color=TINTA, pad=12)
ax.set_ylabel('% com diabetes')
ax.set_xlabel('Faixa etária (anos)')
fig.tight_layout()
fig.savefig(f'{PASTA}/fig_prevalencia_faixa_etaria.png', dpi=200)
plt.close(fig)

# 4.2 AUC: no próprio país e transferido
fig, ax = plt.subplots(figsize=(7.5, 4))
destinos = ['Brasil', 'EUA']
barras_agrupadas(ax, [f'Testado no {d}' if d == 'Brasil' else f'Testado nos {d}' for d in destinos],
                 {o: [resultados[f'{o}->{d}']['auc'] for d in destinos] for o in paises},
                 lambda v: f'{v:.3f}')
for t in ax.get_legend().get_texts():
    t.set_text(f"Treinado no {t.get_text()}" if t.get_text() == 'Brasil' else f"Treinado nos {t.get_text()}")
ax.set_ylim(0.5, 0.9)
ax.set_title('Desempenho (AUC) no próprio país e no outro país', loc='left', fontsize=12, color=TINTA, pad=12)
ax.set_ylabel('AUC (0,5 = acaso)')
fig.tight_layout()
fig.savefig(f'{PASTA}/fig_auc_transferencia.png', dpi=200)
plt.close(fig)

# 4.3 Importância das variáveis
fig, ax = plt.subplots(figsize=(7.5, 4))
ordem = sorted(FEATURES, key=lambda f: importancia['EUA'][f] + importancia['Brasil'][f], reverse=True)
barras_agrupadas(ax, [NOMES[f] for f in ordem], {n: [importancia[n][f] * 100 for f in ordem] for n in paises},
                 lambda v: f'{v:.0f}%', legenda='upper right')
ax.set_title('Importância de cada variável no modelo (ganho)', loc='left', fontsize=12, color=TINTA, pad=12)
ax.set_ylabel('% do ganho total')
fig.tight_layout()
fig.savefig(f'{PASTA}/fig_importancia_variaveis.png', dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. JSON e relatório
# ---------------------------------------------------------------------------
saida = {
    'gerado_em': datetime.now().isoformat(timespec='seconds'),
    'features_harmonizadas': FEATURES,
    'descritiva': desc,
    'qui_quadrado_prevalencia_p': p_prev,
    'prevalencia_brasil_padronizada_idade_eua': prev_br_padronizada,
    'limiares': {n: modelos[n][1] for n in modelos},
    'resultados': resultados,
    'importancia': importancia,
}
with open(f'{PASTA}/comparacao_cross_cultural.json', 'w', encoding='utf-8') as f:
    json.dump(saida, f, indent=2, ensure_ascii=False)


def pct(v):
    return f'{v * 100:.1f}%'


B, U = desc['Brasil'], desc['EUA']
R = resultados
linhas_faixa = '\n'.join(
    f"| {f} | {pct(B['prevalencia_por_faixa'][f])} | {pct(U['prevalencia_por_faixa'][f])} | "
    f"{pct(B['distribuicao_faixa'][f])} | {pct(U['distribuicao_faixa'][f])} |" for f in faixas
)
linhas_imp = '\n'.join(f"| {NOMES[f]} | {pct(importancia['Brasil'][f])} | {pct(importancia['EUA'][f])} |" for f in ordem)

relatorio = f"""# Validação Cross-Cultural: Brasil (VIGITEL 2023) x EUA (BRFSS 2015)

*Gerado por `validacao_cross_cultural.py` em {datetime.now().strftime('%d/%m/%Y')}.*

## 1. Dados e harmonização

| | Brasil | EUA |
|---|---|---|
| Pesquisa | VIGITEL 2023 (telefone) | BRFSS 2015 (telefone, dados originais do CDC) |
| Registros válidos | {B['n']:,} | {U['n']:,} |
| Diabetes diagnosticado | {pct(B['prevalencia_diabetes'])} na amostra, **{pct(B['prevalencia_diabetes_ponderada'])} ponderada** | {pct(U['prevalencia_diabetes'])} na amostra, **{pct(U['prevalencia_diabetes_ponderada'])} ponderada** |

Nos dois países o desfecho é **diabetes diagnosticado por médico** (autorreferido). No BRFSS foram usados os
dados originais do CDC, e não a versão do Kaggle do modelo comportamental, porque esta agrupa pré-diabetes com
diabetes; aqui, pré-diabetes e diabetes apenas na gravidez contam como "não".

As variáveis do VIGITEL foram conferidas no dicionário oficial. Para comparar os países, os modelos usam
apenas as variáveis com **definição equivalente** nas duas pesquisas:

| Variável | VIGITEL | BRFSS |
|---|---|---|
| Idade | `q6` (anos), convertida para as 13 faixas do BRFSS | faixa etária (1 = 18-24 ... 13 = 80+) |
| Sexo | `q7` | `Sex` |
| IMC | `imc` | `BMI` |
| Pressão alta | `q75` (diagnóstico médico) | `HighBP` (diagnóstico médico) |
| Saúde autoavaliada | `q74` (muito bom ... muito ruim) | `GenHlth` (excelente ... ruim) |

**Ficaram de fora dos modelos (definições diferentes):**
- *Colesterol alto*: o VIGITEL não pergunta sobre o diagnóstico (só se fez o exame).
- *Fumante*: BRFSS = já fumou 100 cigarros na vida; VIGITEL = fuma atualmente.
- *Atividade física*: BRFSS = qualquer atividade no último mês; VIGITEL = ≥ 150 min/semana no lazer.
- *Frutas*: BRFSS = ≥ 1 vez/dia; VIGITEL = ≥ 5 dias/semana.

## 2. Perfil das populações

| Indicador | Brasil | EUA |
|---|---|---|
| IMC médio | {B['imc_medio']:.1f} | {U['imc_medio']:.1f} |
| Obesidade (IMC ≥ 30) | {pct(B['obesidade'])} | {pct(U['obesidade'])} |
| Pressão alta | {pct(B['pressao_alta'])} (ponderada: {pct(B['pressao_alta_ponderada'])}) | {pct(U['pressao_alta'])} (ponderada: {pct(U['pressao_alta_ponderada'])}) |
| Saúde ruim/muito ruim | {pct(B['saude_ruim_ou_muito_ruim'])} | {pct(U['saude_ruim_ou_muito_ruim'])} |
| Fumante* | {pct(B['fumante*'])} | {pct(U['fumante*'])} |
| Atividade física* | {pct(B['atividade_fisica*'])} | {pct(U['atividade_fisica*'])} |
| Frutas* | {pct(B['frutas*'])} | {pct(U['frutas*'])} |

\\* Definições diferentes entre as pesquisas (ver seção 1): **não comparar diretamente**.

**Prevalência por faixa etária**

| Faixa | Diabetes Brasil | Diabetes EUA | % da amostra Brasil | % da amostra EUA |
|---|---|---|---|---|
{linhas_faixa}

A diferença de prevalência bruta {'é' if p_prev < 0.05 else 'não é'} estatisticamente significativa (qui-quadrado,
p = {p_prev:.2g}). As amostras, porém, têm composição etária diferente: a dos EUA é mais velha. Padronizando o
Brasil pela distribuição etária dos EUA, a prevalência brasileira fica em **{pct(prev_br_padronizada)}**
(EUA: {pct(U['prevalencia_diabetes'])}): a partir dos 55 anos, a prevalência é maior no Brasil.

![Prevalência por faixa etária](fig_prevalencia_faixa_etaria.png)

## 3. Modelos e transferência entre países

Mesmo pipeline nos dois países: XGBoost, divisão 60/20/20 estratificada, limiar de decisão escolhido no
conjunto de validação (maior F1) e métricas calculadas no conjunto de teste, que o modelo nunca viu.

| Treinado em | Testado em | AUC | F1 | Precisão | Recall |
|---|---|---|---|---|---|
| Brasil | Brasil | {R['Brasil->Brasil']['auc']:.3f} | {R['Brasil->Brasil']['f1']:.3f} | {pct(R['Brasil->Brasil']['precisao'])} | {pct(R['Brasil->Brasil']['recall'])} |
| EUA | EUA | {R['EUA->EUA']['auc']:.3f} | {R['EUA->EUA']['f1']:.3f} | {pct(R['EUA->EUA']['precisao'])} | {pct(R['EUA->EUA']['recall'])} |
| EUA | Brasil | {R['EUA->Brasil']['auc']:.3f} | {R['EUA->Brasil']['f1']:.3f} | {pct(R['EUA->Brasil']['precisao'])} | {pct(R['EUA->Brasil']['recall'])} |
| Brasil | EUA | {R['Brasil->EUA']['auc']:.3f} | {R['Brasil->EUA']['f1']:.3f} | {pct(R['Brasil->EUA']['precisao'])} | {pct(R['Brasil->EUA']['recall'])} |

A **AUC** não depende do limiar e é a melhor métrica para comparar a capacidade de ordenar o risco entre
países. F1, precisão e recall dependem do limiar e da prevalência de cada população.

![AUC no próprio país e no outro](fig_auc_transferencia.png)

## 4. Importância das variáveis

| Variável | Brasil | EUA |
|---|---|---|
{linhas_imp}

![Importância das variáveis](fig_importancia_variaveis.png)

## 5. Limitações

- Pesquisas de anos diferentes (2015 x 2023) e diabetes **autorreferido**, que subestima casos não diagnosticados.
- A escala de saúde autoavaliada tem âncoras diferentes nas duas pesquisas (ex.: "regular" x "good").
- As prevalências ponderadas usam os pesos amostrais de cada pesquisa; os modelos e as demais comparações usam
  as amostras sem ponderação.
- Variáveis importantes nos EUA (colesterol) não puderam ser usadas por não existirem no VIGITEL.
"""

with open(f'{PASTA}/ANALISE_CROSS_CULTURAL.md', 'w', encoding='utf-8') as f:
    f.write(relatorio)

print("Gerados: comparacao_cross_cultural.json, ANALISE_CROSS_CULTURAL.md, 3 figuras, modelo_vigitel_xgboost.pkl")
