# Validação Cross-Cultural: Brasil (VIGITEL 2023) x EUA (BRFSS 2015)

*Gerado por `validacao_cross_cultural.py` em 01/10/2026.*

## 1. Dados e harmonização

| | Brasil | EUA |
|---|---|---|
| Pesquisa | VIGITEL 2023 (telefone) | BRFSS 2015 (telefone, dados originais do CDC) |
| Registros válidos | 19,919 | 360,689 |
| Diabetes diagnosticado | 12.6% na amostra, **10.1% ponderada** | 13.0% na amostra, **10.6% ponderada** |

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
| IMC médio | 27.0 | 27.9 |
| Obesidade (IMC ≥ 30) | 23.9% | 29.6% |
| Pressão alta | 33.3% (ponderada: 27.3%) | 40.7% (ponderada: 32.6%) |
| Saúde ruim/muito ruim | 6.0% | 18.3% |
| Fumante* | 8.2% | 44.0% |
| Atividade física* | 41.5% | 74.1% |
| Frutas* | 60.8% | 62.2% |

\* Definições diferentes entre as pesquisas (ver seção 1): **não comparar diretamente**.

**Prevalência por faixa etária**

| Faixa | Diabetes Brasil | Diabetes EUA | % da amostra Brasil | % da amostra EUA |
|---|---|---|---|---|
| 18-34 | 1.8% | 1.5% | 23.5% | 14.6% |
| 35-44 | 4.8% | 5.1% | 19.3% | 11.4% |
| 45-54 | 10.3% | 10.0% | 18.2% | 16.3% |
| 55-64 | 19.8% | 15.4% | 16.9% | 22.5% |
| 65+ | 27.1% | 20.2% | 22.1% | 35.2% |

A diferença de prevalência bruta não é estatisticamente significativa (qui-quadrado,
p = 0.1). As amostras, porém, têm composição etária diferente: a dos EUA é mais velha. Padronizando o
Brasil pela distribuição etária dos EUA, a prevalência brasileira fica em **16.5%**
(EUA: 13.0%): a partir dos 55 anos, a prevalência é maior no Brasil.

![Prevalência por faixa etária](fig_prevalencia_faixa_etaria.png)

## 3. Modelos e transferência entre países

Mesmo pipeline nos dois países: XGBoost, divisão 60/20/20 estratificada, limiar de decisão escolhido no
conjunto de validação (maior F1) e métricas calculadas no conjunto de teste, que o modelo nunca viu.

| Treinado em | Testado em | AUC | F1 | Precisão | Recall |
|---|---|---|---|---|---|
| Brasil | Brasil | 0.813 | 0.419 | 29.8% | 70.5% |
| EUA | EUA | 0.825 | 0.445 | 36.8% | 56.4% |
| EUA | Brasil | 0.818 | 0.375 | 41.9% | 33.9% |
| Brasil | EUA | 0.787 | 0.375 | 24.3% | 82.1% |

A **AUC** não depende do limiar e é a melhor métrica para comparar a capacidade de ordenar o risco entre
países. F1, precisão e recall dependem do limiar e da prevalência de cada população.

![AUC no próprio país e no outro](fig_auc_transferencia.png)

## 4. Importância das variáveis

| Variável | Brasil | EUA |
|---|---|---|
| Pressão alta | 41.9% | 63.0% |
| Saúde autoavaliada | 17.8% | 20.7% |
| Idade | 25.6% | 8.7% |
| IMC | 9.0% | 5.7% |
| Sexo (masc.) | 5.7% | 1.8% |

![Importância das variáveis](fig_importancia_variaveis.png)

## 5. Limitações

- Pesquisas de anos diferentes (2015 x 2023) e diabetes **autorreferido**, que subestima casos não diagnosticados.
- A escala de saúde autoavaliada tem âncoras diferentes nas duas pesquisas (ex.: "regular" x "good").
- As prevalências ponderadas usam os pesos amostrais de cada pesquisa; os modelos e as demais comparações usam
  as amostras sem ponderação.
- Variáveis importantes nos EUA (colesterol) não puderam ser usadas por não existirem no VIGITEL.
