# 5 RESULTADOS

> **Rascunho.** Revise e reescreva com a sua voz. Números extraídos de `resultados_otimizacao_clinico.json`,
> `modelo_comportamental_meta.json`, `dados_vigitel/comparacao_cross_cultural.json` e
> `dados_nhanes/resultados_rastreamento.json`. Ajuste a numeração de tabelas e figuras ao seu documento.

Este capítulo apresenta os resultados dos quatro estudos descritos no Capítulo 3. A interpretação e as
limitações são discutidas no Capítulo 6.

## 5.1 Modelo clínico

A Tabela 1 apresenta o desempenho médio dos quatro algoritmos nas 15 avaliações da validação cruzada aninhada.

**Tabela 1 – Desempenho dos algoritmos no modelo clínico (média ± desvio-padrão em 15 avaliações)**

| Algoritmo | F1 | AUC | Precisão | Recall | Brier |
|---|---|---|---|---|---|
| **Random Forest** | **0,691 ± 0,038** | **0,839** | 59,9% | 82,2% | 0,158 |
| XGBoost | 0,678 ± 0,037 | 0,832 | 59,2% | 80,3% | 0,159 |
| Regressão Logística | 0,669 ± 0,042 | 0,833 | 57,8% | 80,2% | 0,159 |
| SVM | 0,669 ± 0,041 | 0,827 | 57,5% | 81,0% | 0,163 |

O Random Forest obteve o maior F1 médio (0,691) e também a maior AUC (0,839). O erro-padrão do seu F1 foi de
0,010; o XGBoost, segundo colocado, ficou 0,013 abaixo, fora da margem de um erro-padrão. Pela regra de
escolha definida previamente, o Random Forest foi selecionado. A diferença entre o melhor e o pior algoritmo
foi de 0,022 em F1 e de 0,012 em AUC.

O modelo final, ajustado com todas as 768 pacientes, utilizou 300 árvores com profundidade máxima 4, sem
indicador de valor ausente, e limiar de decisão de 0,39. Com esse modelo, o sistema identifica corretamente
cerca de 82% das pacientes que desenvolveram diabetes (recall), com precisão de 60%.

## 5.2 Modelo comportamental

Na grade de 16 configurações do XGBoost, a AUC no conjunto de validação variou entre 0,819 e 0,822; a melhor
configuração (profundidade 3, 600 árvores, peso mínimo por nó 1) foi selecionada. A Regressão Logística de
referência obteve AUC de 0,815 na validação.

No conjunto de teste (50.736 entrevistas não utilizadas no treinamento), com limiar de 0,23, o modelo
apresentou os resultados da Tabela 2.

**Tabela 2 – Desempenho do modelo comportamental no conjunto de teste**

| Métrica | Valor |
|---|---|
| AUC | 0,821 |
| F1 | 0,458 |
| Precisão | 36,7% |
| Recall | 61,1% |
| Brier | 0,099 |

## 5.3 Validação entre Brasil e Estados Unidos

### 5.3.1 Perfil das populações

A Tabela 3 compara as amostras do VIGITEL 2023 e do BRFSS 2015.

**Tabela 3 – Características das amostras**

| Indicador | Brasil (VIGITEL 2023) | EUA (BRFSS 2015) |
|---|---|---|
| Registros | 19.919 | 253.680 |
| Diabetes | 12,6% (10,1% ponderado) | 13,9% |
| IMC médio (kg/m²) | 27,0 | 28,4 |
| Obesidade (IMC ≥ 30) | 23,9% | 34,6% |
| Pressão alta | 33,3% (27,3% ponderado) | 42,9% |
| Saúde ruim ou muito ruim | 6,0% | 17,2% |
| Pessoas com 65 anos ou mais | 22,1% | 35,1% |
| Pessoas de 18 a 34 anos | 23,5% | 9,6% |

A diferença de prevalência de diabetes entre as amostras foi estatisticamente significativa (qui-quadrado,
p = 1,2 × 10⁻⁷). As amostras, porém, têm composição etária diferente: a do BRFSS é mais velha. A Figura X
mostra a prevalência por faixa etária.

**Figura X – Prevalência de diabetes por faixa etária** (`dados_vigitel/fig_prevalencia_faixa_etaria.png`)

**Tabela 4 – Prevalência de diabetes por faixa etária**

| Faixa etária | Brasil | EUA |
|---|---|---|
| 18–34 | 1,8% | 2,2% |
| 35–44 | 4,8% | 5,6% |
| 45–54 | 10,3% | 10,5% |
| 55–64 | 19,8% | 15,6% |
| 65 ou mais | 27,1% | 20,6% |

Até os 54 anos, as prevalências são semelhantes; a partir dos 55 anos, a prevalência brasileira é maior.
Padronizando a amostra brasileira pela distribuição etária do BRFSS, a prevalência brasileira passa a 17,1%,
contra 13,9% nos EUA.

### 5.3.2 Desempenho e transferência dos modelos

A Tabela 5 apresenta o desempenho de cada modelo no conjunto de teste do próprio país e do outro país.

**Tabela 5 – Desempenho dos modelos no próprio país e no outro país**

| Treinado em | Testado em | AUC | F1 | Precisão | Recall |
|---|---|---|---|---|---|
| Brasil | Brasil | 0,813 | 0,419 | 29,8% | 70,5% |
| EUA | EUA | 0,814 | 0,449 | 35,2% | 61,8% |
| EUA | Brasil | 0,814 | 0,372 | 39,3% | 35,3% |
| Brasil | EUA | 0,773 | 0,378 | 24,7% | 80,9% |

O modelo treinado com dados dos EUA obteve no Brasil a mesma AUC (0,814) que o modelo treinado com dados
brasileiros (0,813). No sentido inverso, o modelo brasileiro teve AUC de 0,773 nos EUA. Ao ser transferido, o
modelo americano apresentou recall mais baixo no Brasil (35,3%), enquanto o brasileiro apresentou recall mais
alto nos EUA (80,9%), variações que acompanham o limiar de decisão de cada modelo (0,22 nos EUA e 0,17 no
Brasil).

**Figura X – AUC no próprio país e no outro país** (`dados_vigitel/fig_auc_transferencia.png`)

### 5.3.3 Importância das variáveis

**Tabela 6 – Importância das variáveis (proporção do ganho total)**

| Variável | Brasil | EUA |
|---|---|---|
| Pressão alta | 41,9% | 58,9% |
| Saúde autoavaliada | 17,8% | 25,7% |
| Idade | 25,6% | 7,3% |
| IMC | 9,0% | 6,1% |
| Sexo | 5,7% | 1,9% |

Pressão alta foi a variável mais importante nos dois países. A idade teve peso maior no modelo brasileiro.

**Figura X – Importância das variáveis** (`dados_vigitel/fig_importancia_variaveis.png`)

## 5.4 Diabetes não diagnosticado (NHANES)

Das 17.306 pessoas sem diagnóstico prévio, 648 (3,7%) apresentaram HbA1c ≥ 6,5%. Com os pesos amostrais, a
estimativa para a população adulta dos EUA sem diagnóstico é de 2,4%.

**Tabela 7 – Identificação de diabetes não diagnosticado (média em 15 avaliações)**

| Método | AUC | AUC-PR | Fração testada para achar 80% dos casos | Casos achados testando o mesmo número que o ADA ≥ 5 |
|---|---|---|---|---|
| **Regressão Logística** | **0,827 ± 0,018** | 0,148 | **31,2%** | **88,9%** |
| XGBoost | 0,820 ± 0,018 | 0,148 | 33,6% | 87,7% |
| Random Forest | 0,815 ± 0,017 | 0,133 | 33,2% | 87,8% |
| Escore ADA (referência) | 0,762 | — | 40,4% | 81,9% |
| Somente idade (referência) | 0,673 | — | 52,9% | — |

A AUC-PR do acaso, igual à prevalência, é 0,037. A Regressão Logística obteve a maior AUC e foi selecionada
pela regra definida previamente. Em relação ao escore da ADA, a diferença de AUC, calculada em cada partição,
foi de +0,065 ± 0,018. Para encontrar 80% dos casos, o modelo exige testar 31,2% da população, contra 40,4%
com o escore da ADA, o que corresponde a 23% menos exames. Testando o mesmo número de pessoas indicado pelo
escore da ADA (41,5% da população), o modelo encontra 88,9% dos casos, contra 81,9% do escore.

**Figura X – Curva de rastreamento** (`dados_nhanes/fig_curva_rastreamento.png`)

A Tabela 8 apresenta as razões de chance da Regressão Logística ajustada com todos os dados.

**Tabela 8 – Razões de chance por desvio-padrão de cada variável (principais)**

| Variável | Razão de chance |
|---|---|
| Idade | 2,18 |
| HDL | 0,54 |
| Etnia asiática (vs. branco não hispânico) | 1,75 |
| Etnia negra não hispânica | 1,56 |
| Razão cintura/altura | 1,44 |
| Histórico familiar | 1,35 |
| IMC | 1,26 |
| Colesterol total | 1,26 |
| Diabetes gestacional | 1,18 |
| Pressão sistólica | 1,16 |
| Atividade física | 0,89 |
| Escolaridade | 0,90 |

As maiores associações com diabetes não diagnosticado foram idade e HDL baixo, seguidos de etnia, razão
cintura/altura e histórico familiar. Atividade física e escolaridade se associaram a menor risco.

**Figura X – Razões de chance** (`dados_nhanes/fig_razoes_chance.png`)

## 5.5 Sistema desenvolvido

A aplicação foi publicada no Streamlit Community Cloud. Os testes funcionais automatizados percorreram todas as
páginas e executaram os dois modelos com perfis de baixo e de alto risco, sem erros (Tabela 9). O relatório em
PDF foi gerado nos dois casos do modelo clínico.

**Tabela 9 – Testes funcionais dos modelos na aplicação**

| Modelo | Perfil de teste | Probabilidade | Classificação |
|---|---|---|---|
| Clínico | Glicemia TOTG 90 mg/dL, insulina 60, IMC 22, 25 anos | 4,8% | Risco baixo |
| Clínico | Glicemia TOTG 185 mg/dL, insulina 250, IMC 36, 55 anos | 72,0% | Risco elevado |
| Comportamental | 25 anos, IMC 22, saúde excelente, sem fatores de risco | 0,2% | Risco baixo |
| Comportamental | 65 anos, IMC 36, saúde ruim, com todos os fatores de risco | 69,0% | Risco elevado |

As previsões da aplicação foram conferidas com as do modelo original: para o modelo clínico, a diferença
máxima entre as probabilidades calculadas pela aplicação e pela biblioteca de treinamento, nas 768 pacientes,
foi zero.

*(Incluir aqui capturas de tela das telas de login, modelo clínico com resultado, histórico e PDF.)*
