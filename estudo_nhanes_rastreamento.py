# -*- coding: utf-8 -*-
"""
ESTUDO COMPLEMENTAR: RASTREAMENTO DE DIABETES NÃO DIAGNOSTICADO (NHANES 2011-2018)

Pergunta: entre adultos sem diagnóstico de diabetes, quem tem diabetes pela HbA1c (>= 6,5%) e não sabe?
A HbA1c e a glicemia NÃO entram como variáveis (definem o desfecho).

Protocolo definido antes de rodar:
- Variáveis: as de uma consulta comum + questionário (lista VARIAVEIS abaixo), fixadas a priori.
- Candidatos: Regressão Logística, XGBoost, Random Forest.
- Validação cruzada aninhada 5x3; hiperparâmetros escolhidos no treino de cada partição (AUC).
- Escolha: maior AUC média; se um modelo mais simples ficar a menos de 1 erro-padrão, fica o mais simples.
- Comparação principal: escore de risco da American Diabetes Association (ADA), calculado nas mesmas pessoas.

Pré-requisito: python baixar_nhanes.py
Saídas em dados_nhanes/: resultados_rastreamento.json, ESTUDO_RASTREAMENTO_NHANES.md, fig_*.png
"""

import json
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
SEED = 42
PASTA = 'dados_nhanes'
CICLOS = {'G': '2011-2012', 'H': '2013-2014', 'I': '2015-2016', 'J': '2017-2018'}

# ---------------------------------------------------------------------------
# 1. Dados
# ---------------------------------------------------------------------------
COLUNAS = {
    'DEMO': ['RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'RIDEXPRG', 'DMDEDUC2', 'INDFMPIR', 'WTMEC2YR'],
    'BMX': ['BMXBMI', 'BMXWAIST', 'BMXHT'], 'BPX': ['BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXDI1', 'BPXDI2', 'BPXDI3'],
    'GHB': ['LBXGH'], 'TCHOL': ['LBXTC'], 'HDL': ['LBDHDD'], 'DIQ': ['DIQ010'], 'BPQ': ['BPQ020'],
    'MCQ': ['MCQ300C'], 'PAQ': ['PAQ605', 'PAQ620', 'PAQ650', 'PAQ665'], 'SMQ': ['SMQ020', 'SMQ040'],
    'RHQ': ['RHQ162'],
}
partes = []
for sufixo in CICLOS:
    d = None
    for arquivo, cols in COLUNAS.items():
        x = pd.read_sas(f'{PASTA}/{arquivo}_{sufixo}.xpt', format='xport')[['SEQN'] + cols]
        d = x if d is None else d.merge(x, on='SEQN', how='left')
    partes.append(d.assign(ciclo=sufixo))
bruto = pd.concat(partes, ignore_index=True)

# Adultos >= 20 anos (idade de rastreamento), não grávidas, com HbA1c e sem diagnóstico prévio de diabetes
df = bruto[(bruto.RIDAGEYR >= 20) & (bruto.RIDEXPRG != 1) & bruto.LBXGH.notna() & (bruto.DIQ010 != 1)].copy()
y = (df.LBXGH >= 6.5).astype(int).to_numpy()
peso = (df.WTMEC2YR / len(CICLOS)).to_numpy()  # peso amostral para 4 ciclos combinados


def sim(serie):
    """1 = sim, 2 = não; recusa (7) e não sabe (9) viram ausente."""
    return serie.map({1: 1.0, 2: 0.0})


df['idade'] = df.RIDAGEYR
df['sexo_masc'] = (df.RIAGENDR == 1).astype(float)
for codigo, nome in [(1, 'mexicano'), (2, 'outro_hispanico'), (4, 'negro'), (6, 'asiatico'), (7, 'outra')]:
    df[f'etnia_{nome}'] = (df.RIDRETH3 == codigo).astype(float)  # referência: branco não hispânico
df['imc'] = df.BMXBMI
df['cintura_altura'] = df.BMXWAIST / df.BMXHT
df['pas'] = df[['BPXSY1', 'BPXSY2', 'BPXSY3']].mean(axis=1)
df['pad'] = df[['BPXDI1', 'BPXDI2', 'BPXDI3']].replace(0, np.nan).mean(axis=1)  # PAD 0 = não medida
df['colesterol'] = df.LBXTC
df['hdl'] = df.LBDHDD
df['hipertensao_dx'] = sim(df.BPQ020)
df['hist_familiar'] = sim(df.MCQ300C)
ativ = df[['PAQ605', 'PAQ620', 'PAQ650', 'PAQ665']]
df['fisicamente_ativo'] = np.where((ativ == 1).any(axis=1), 1.0, np.where((ativ == 2).all(axis=1), 0.0, np.nan))
df['fumante_atual'] = np.where(df.SMQ020 == 2, 0.0, np.where(df.SMQ040.isin([1, 2]), 1.0,
                               np.where(df.SMQ040 == 3, 0.0, np.nan)))
df['ex_fumante'] = np.where(df.SMQ020 == 2, 0.0, np.where(df.SMQ040 == 3, 1.0,
                            np.where(df.SMQ040.isin([1, 2]), 0.0, np.nan)))
df['escolaridade'] = df.DMDEDUC2.where(df.DMDEDUC2.between(1, 5))
df['renda_pobreza'] = df.INDFMPIR
df['diabetes_gestacional'] = np.where(df.sexo_masc == 1, 0.0, sim(df.RHQ162))

VARIAVEIS = ['idade', 'sexo_masc', 'etnia_mexicano', 'etnia_outro_hispanico', 'etnia_negro', 'etnia_asiatico',
             'etnia_outra', 'imc', 'cintura_altura', 'pas', 'pad', 'colesterol', 'hdl', 'hipertensao_dx',
             'hist_familiar', 'fisicamente_ativo', 'fumante_atual', 'ex_fumante', 'escolaridade', 'renda_pobreza',
             'diabetes_gestacional']
NOMES = {
    'idade': 'Idade', 'sexo_masc': 'Sexo masculino', 'etnia_mexicano': 'Mexicano-americano',
    'etnia_outro_hispanico': 'Outro hispânico', 'etnia_negro': 'Negro não hispânico',
    'etnia_asiatico': 'Asiático', 'etnia_outra': 'Outra etnia', 'imc': 'IMC',
    'cintura_altura': 'Cintura/altura', 'pas': 'Pressão sistólica', 'pad': 'Pressão diastólica',
    'colesterol': 'Colesterol total', 'hdl': 'HDL', 'hipertensao_dx': 'Hipertensão diagnosticada',
    'hist_familiar': 'Histórico familiar', 'fisicamente_ativo': 'Fisicamente ativo',
    'fumante_atual': 'Fumante atual', 'ex_fumante': 'Ex-fumante', 'escolaridade': 'Escolaridade',
    'renda_pobreza': 'Renda (razão p/ linha de pobreza)', 'diabetes_gestacional': 'Diabetes gestacional',
}
X = df[VARIAVEIS].to_numpy(dtype=float)
ausentes = {v: round(float(df[v].isna().mean()), 3) for v in VARIAVEIS}
print(f"{len(y):,} adultos sem diagnóstico | {y.sum()} com HbA1c >= 6,5% "
      f"({y.mean()*100:.1f}% na amostra; {np.average(y, weights=peso)*100:.1f}% ponderado)")


# ---------------------------------------------------------------------------
# 2. Escore de risco da ADA (Bang et al., 2009) - referência usada na prática clínica
# ---------------------------------------------------------------------------
def escore_ada(d):
    idade = np.select([d.idade < 40, d.idade < 50, d.idade < 60], [0, 1, 2], 3)
    imc = np.select([d.imc < 25, d.imc < 30, d.imc < 40], [0, 1, 2], 3)
    return (idade + d.sexo_masc.fillna(0) + d.diabetes_gestacional.fillna(0) + d.hist_familiar.fillna(0)
            + d.hipertensao_dx.fillna(0) + (d.fisicamente_ativo == 0).astype(int) + imc).to_numpy(dtype=float, copy=True)


ada = escore_ada(df)
ada[df.imc.isna().to_numpy()] = np.nan


# ---------------------------------------------------------------------------
# 3. Modelos: validação cruzada aninhada
# ---------------------------------------------------------------------------
def pipe(modelo):
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('modelo', modelo)])


CANDIDATOS = {
    'Regressão Logística': (pipe(LogisticRegression(max_iter=3000)), {'modelo__C': [0.01, 0.1, 1]}),
    'XGBoost': (xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, subsample=0.9, eval_metric='logloss',
                                  random_state=SEED, n_jobs=1),
                {'max_depth': [2, 3, 4], 'min_child_weight': [1, 10]}),
    'Random Forest': (pipe(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1)),
                      {'modelo__max_depth': [6, 10], 'modelo__min_samples_leaf': [10, 30]}),
}
SIMPLICIDADE = ['Regressão Logística', 'XGBoost', 'Random Forest']


def fracao_testada(y_, p_, sens=0.80):
    """Fração da população a testar (maior risco primeiro) para encontrar `sens` dos casos."""
    ordem = np.argsort(-p_, kind='stable')
    achados = np.cumsum(y_[ordem]) / y_.sum()
    return float((np.searchsorted(achados, sens) + 1) / len(y_))


def sensibilidade_testando(y_, p_, fracao):
    """Fração dos casos encontrados testando a `fracao` de maior risco."""
    n = int(round(fracao * len(y_)))
    ordem = np.argsort(-p_, kind='stable')[:n]
    return float(y_[ordem].sum() / y_.sum())


validos = ~np.isnan(ada)
X, y, ada, peso = X[validos], y[validos], ada[validos], peso[validos]
externo = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
interno = StratifiedKFold(5, shuffle=True, random_state=SEED)

resultados, oof = {}, {}
for nome, (modelo, grade) in CANDIDATOS.items():
    linhas = []
    oof[nome] = np.zeros((3, len(y)))
    for k, (tr, te) in enumerate(externo.split(X, y)):
        busca = GridSearchCV(modelo, grade, scoring='roc_auc', cv=interno, n_jobs=-1).fit(X[tr], y[tr])
        p = busca.predict_proba(X[te])[:, 1]
        oof[nome][k // 5, te] = p
        ada_ab = ada[te] >= 5
        fr_ada = ada_ab.mean()
        linhas.append({
            'auc': roc_auc_score(y[te], p), 'auc_pr': average_precision_score(y[te], p),
            'brier': brier_score_loss(y[te], p), 'testar_80': fracao_testada(y[te], p),
            'auc_ada': roc_auc_score(y[te], ada[te]),
            'sens_ada5': y[te][ada_ab].sum() / y[te].sum(), 'fracao_ada5': fr_ada,
            'sens_modelo_mesma_fracao': sensibilidade_testando(y[te], p, fr_ada),
        })
    m = pd.DataFrame(linhas)
    m['dif_auc_vs_ada'] = m['auc'] - m['auc_ada']
    resultados[nome] = {
        **{k: round(float(m[k].mean()), 4) for k in m.columns},
        'auc_desvio': round(float(m['auc'].std()), 4),
        'auc_erro_padrao': round(float(m['auc'].std() / np.sqrt(len(m))), 4),
        'dif_auc_vs_ada_desvio': round(float(m['dif_auc_vs_ada'].std()), 4),
    }
    r = resultados[nome]
    print(f"{nome:20s} AUC {r['auc']:.3f} ± {r['auc_desvio']:.3f} (ADA {r['auc_ada']:.3f}) | AUC-PR {r['auc_pr']:.3f} | "
          f"testar {r['testar_80']*100:.0f}% p/ 80% | mesma fração do ADA≥5: {r['sens_modelo_mesma_fracao']*100:.0f}% "
          f"dos casos vs {r['sens_ada5']*100:.0f}%")

melhor = max(resultados, key=lambda n: resultados[n]['auc'])
corte = resultados[melhor]['auc'] - resultados[melhor]['auc_erro_padrao']
escolhido = next(n for n in SIMPLICIDADE if resultados[n]['auc'] >= corte)
print(f"Maior AUC: {melhor} | escolhido (regra de 1 erro-padrão): {escolhido}")

idade_v = df['idade'].to_numpy(dtype=float)[validos]
ref = {
    'ada': {'auc': round(float(roc_auc_score(y, ada)), 4), 'testar_80': round(fracao_testada(y, ada), 4)},
    'idade': {'auc': round(float(roc_auc_score(y, idade_v)), 4), 'testar_80': round(fracao_testada(y, idade_v), 4)},
}

# Razões de chance (Regressão Logística em todos os dados), por desvio-padrão de cada variável
logit = pipe(LogisticRegression(max_iter=3000, C=1)).fit(X, y)
razoes = {v: round(float(np.exp(c)), 3) for v, c in zip(VARIAVEIS, logit.named_steps['modelo'].coef_[0])}

# ---------------------------------------------------------------------------
# 4. Figuras
# ---------------------------------------------------------------------------
SUPERFICIE, TINTA, TINTA_2, GRADE = '#fcfcfb', '#1a1a19', '#5f5e58', '#e4e3dd'
plt.rcParams.update({
    'figure.facecolor': SUPERFICIE, 'axes.facecolor': SUPERFICIE, 'savefig.facecolor': SUPERFICIE,
    'axes.edgecolor': GRADE, 'axes.labelcolor': TINTA_2, 'xtick.color': TINTA_2, 'ytick.color': TINTA_2,
    'text.color': TINTA, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRADE, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
})

# 4.1 Curva de rastreamento: % testado x % de casos encontrados (predições fora da amostra, 1ª repetição)
p_melhor = oof[escolhido][0]
fig, ax = plt.subplots(figsize=(7.5, 4.6))
series = [(f'Modelo ({escolhido})', p_melhor, '#2a78d6', '-'), ('Escore ADA', ada, '#eb6834', '--'),
          ('Só idade', idade_v, '#1baf7a', ':')]
for rotulo, score, cor, estilo in series:
    ordem = np.argsort(-score, kind='stable')
    x_ = np.arange(1, len(y) + 1) / len(y) * 100
    y_ = np.cumsum(y[ordem]) / y.sum() * 100
    ax.plot(x_, y_, color=cor, linewidth=2, linestyle=estilo, label=rotulo)
ax.plot([0, 100], [0, 100], color=GRADE, linewidth=1)
ax.axhline(80, color=TINTA_2, linewidth=0.8, linestyle='--')
ax.text(99, 81.5, '80% dos casos', ha='right', fontsize=8.5, color=TINTA_2)
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.legend(loc='lower right', frameon=False)
ax.set_xlabel('% da população testada (maior risco primeiro)')
ax.set_ylabel('% dos casos não diagnosticados encontrados')
ax.set_title('Rastreamento de diabetes não diagnosticado', loc='left', fontsize=12, color=TINTA, pad=12)
fig.tight_layout(); fig.savefig(f'{PASTA}/fig_curva_rastreamento.png', dpi=200); plt.close(fig)

# 4.2 Razões de chance (série única)
ordem_rc = sorted(VARIAVEIS, key=lambda v: razoes[v])
fig, ax = plt.subplots(figsize=(7.5, 6))
vals = [razoes[v] for v in ordem_rc]
ax.barh([NOMES[v] for v in ordem_rc], [v - 1 for v in vals], left=1, color='#2a78d6', height=0.6)
for i, v in enumerate(vals):
    ax.text(v + (0.02 if v >= 1 else -0.02), i, f'{v:.2f}', va='center', ha='left' if v >= 1 else 'right',
            fontsize=8.5, color=TINTA_2)
ax.axvline(1, color=TINTA_2, linewidth=1)
ax.set_xlim(0.3, max(vals) + 0.3)
ax.grid(axis='y', visible=False)
ax.set_xlabel('Razão de chance por 1 desvio-padrão (1 = sem efeito)')
ax.set_title('Associação com diabetes não diagnosticado', loc='left', fontsize=12, color=TINTA, pad=12)
fig.tight_layout(); fig.savefig(f'{PASTA}/fig_razoes_chance.png', dpi=200); plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Saídas
# ---------------------------------------------------------------------------
saida = {
    'gerado_em': datetime.now().isoformat(timespec='seconds'), 'ciclos': list(CICLOS.values()),
    'n': int(len(y)), 'casos': int(y.sum()), 'prevalencia_amostra': float(y.mean()),
    'prevalencia_ponderada': float(np.average(y, weights=peso)), 'variaveis': VARIAVEIS, 'ausentes': ausentes,
    'resultados': resultados, 'maior_auc': melhor, 'escolhido': escolhido, 'referencias': ref,
    'razoes_de_chance_por_dp': razoes,
}
with open(f'{PASTA}/resultados_rastreamento.json', 'w', encoding='utf-8') as f:
    json.dump(saida, f, indent=2, ensure_ascii=False)

R = resultados


def pct(v):
    return f'{v * 100:.1f}%'


tabela_modelos = '\n'.join(
    f"| {n} | {R[n]['auc']:.3f} ± {R[n]['auc_desvio']:.3f} | {R[n]['auc_pr']:.3f} | {R[n]['brier']:.3f} | "
    f"{pct(R[n]['testar_80'])} | {pct(R[n]['sens_modelo_mesma_fracao'])} |" for n in CANDIDATOS)
tabela_rc = '\n'.join(f"| {NOMES[v]} | {razoes[v]:.2f} | {pct(ausentes[v])} |"
                      for v in sorted(VARIAVEIS, key=lambda v: -abs(np.log(razoes[v]))))
E = R[escolhido]
relatorio = f"""# Estudo complementar: rastreamento de diabetes não diagnosticado (NHANES 2011-2018)

*Gerado por `estudo_nhanes_rastreamento.py` em {datetime.now().strftime('%d/%m/%Y')}.*

## Pergunta

Entre adultos **sem diagnóstico de diabetes**, é possível identificar quem tem diabetes e não sabe usando apenas
informações de uma consulta comum, sem exame de glicose? E o modelo é melhor que o escore de risco da American
Diabetes Association (ADA), usado na prática?

## Dados e desfecho

- NHANES (CDC, EUA), ciclos {', '.join(CICLOS.values())}: inquérito com exame físico e de sangue.
- Adultos ≥ 20 anos, não grávidas, com HbA1c medida e **sem** diagnóstico prévio de diabetes: **{len(y):,} pessoas**.
- Desfecho: **HbA1c ≥ 6,5%** (critério diagnóstico). Casos: **{y.sum()} ({pct(y.mean())} da amostra; {pct(saida['prevalencia_ponderada'])}
  com os pesos amostrais, estimativa para a população adulta dos EUA sem diagnóstico)**.
- HbA1c e glicemia **não** entram no modelo, pois definem o desfecho.

## Método (definido antes de rodar)

- {len(VARIAVEIS)} variáveis fixadas a priori: idade, sexo, etnia, IMC, cintura/altura, pressão sistólica e diastólica,
  colesterol, HDL, hipertensão diagnosticada, histórico familiar, atividade física, tabagismo, escolaridade, renda e
  diabetes gestacional. Ausentes imputados pela mediana dentro de cada partição de treino.
- Regressão Logística, XGBoost e Random Forest em **validação cruzada aninhada** (5 partes × 3 repetições);
  hiperparâmetros escolhidos só com os dados de treino de cada partição.
- Escolha: maior AUC média; um modelo mais simples a menos de 1 erro-padrão do melhor é preferido.
- Referências nas mesmas pessoas: **escore ADA** (7 perguntas: idade, sexo, diabetes gestacional, histórico familiar,
  hipertensão, atividade física, IMC; alto risco = escore ≥ 5) e ordenação só por idade.

## Resultados

| Modelo | AUC | AUC-PR | Brier | Testar p/ achar 80% dos casos | Casos achados testando o mesmo nº que o ADA ≥ 5 |
|---|---|---|---|---|---|
{tabela_modelos}
| Escore ADA (referência) | {ref['ada']['auc']:.3f} | — | — | {pct(ref['ada']['testar_80'])} | {pct(E['sens_ada5'])} (testando {pct(E['fracao_ada5'])}) |
| Só idade (referência) | {ref['idade']['auc']:.3f} | — | — | {pct(ref['idade']['testar_80'])} | — |

AUC-PR do acaso = prevalência ({y.mean():.3f}). Modelo escolhido: **{escolhido}**.
Diferença de AUC em relação ao escore ADA nas mesmas partições: **{E['dif_auc_vs_ada']:+.3f} ± {E['dif_auc_vs_ada_desvio']:.3f}**.

![Curva de rastreamento](fig_curva_rastreamento.png)

## Associação de cada variável (Regressão Logística)

Razão de chance por 1 desvio-padrão da variável (> 1 aumenta o risco; < 1 reduz), ajustada pelas demais.

| Variável | Razão de chance | Ausentes |
|---|---|---|
{tabela_rc}

![Razões de chance](fig_razoes_chance.png)

## Limitações

- Diagnóstico por **uma única** medida de HbA1c (na prática clínica, confirma-se com um segundo exame).
- Modelagem sem os pesos amostrais (a prevalência ponderada é informada à parte); dados dos EUA, sem validação
  externa em população brasileira.
- A pergunta é diferente da do modelo clínico do app (Pima): aqui não se usa glicose, e a prevalência é muito menor,
  por isso F1 e precisão não são comparáveis entre os dois.
"""
with open(f'{PASTA}/ESTUDO_RASTREAMENTO_NHANES.md', 'w', encoding='utf-8') as f:
    f.write(relatorio)
print("Gerados: resultados_rastreamento.json, ESTUDO_RASTREAMENTO_NHANES.md, fig_curva_rastreamento.png, fig_razoes_chance.png")
