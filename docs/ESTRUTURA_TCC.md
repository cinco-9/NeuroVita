# Estrutura proposta do TCC

**Tema:** Agente de IA para estimativa de risco de diabetes tipo 2, com validação entre populações.
Para cada seção: o que escrever, os números a usar e onde estão no projeto. Páginas são uma estimativa.

---

## Elementos pré-textuais

Capa, folha de rosto, dedicatória/agradecimentos (opcionais), **resumo** e **abstract**, listas de figuras,
tabelas e siglas, sumário.

**Resumo (escrever por último, ~250 palavras):** problema → objetivo → o que foi feito (dois modelos, app,
validação Brasil x EUA, estudo NHANES) → principais números → conclusão principal.

Números para o resumo: clínico F1 0,69 / AUC 0,84; comportamental AUC 0,82; modelo americano no Brasil
AUC 0,81; NHANES AUC 0,83 contra 0,76 do escore ADA.

---

## 1. Introdução (4–6 páginas)

- **Contexto:** diabetes tipo 2 é frequente e muitas vezes silencioso; no Brasil, cerca de 10% dos adultos
  relatam diagnóstico (VIGITEL 2023, ponderado: 10,1%); parte dos casos não é diagnosticada.
- **Problema:** como identificar cedo quem tem risco, inclusive sem exames de laboratório?
- **Objetivo geral:** desenvolver e avaliar um sistema web com modelos de aprendizado de máquina para estimar o
  risco de diabetes tipo 2.
- **Objetivos específicos:**
  1. Treinar e comparar modelos com dados clínicos (Pima) e de questionário (BRFSS);
  2. Avaliar se um modelo treinado nos EUA funciona na população brasileira (VIGITEL);
  3. Avaliar o uso dos modelos para rastrear diabetes não diagnosticado (NHANES);
  4. Disponibilizar os modelos em uma aplicação web com autenticação, histórico e relatório em PDF.
- **Justificativa:** baixo custo da triagem por questionário; falta de validação de modelos em população
  brasileira.
- **Organização do trabalho:** um parágrafo descrevendo os capítulos.

---

## 2. Fundamentação teórica (12–18 páginas)

### 2.1 Diabetes tipo 2
Definição, fatores de risco (idade, obesidade, hipertensão, histórico familiar, sedentarismo, HDL baixo).
**Critérios diagnósticos:** HbA1c ≥ 6,5%, glicemia de jejum ≥ 126 mg/dL, TOTG 2h ≥ 200 mg/dL. Pré-diabetes.
→ Importante para explicar depois por que HbA1c e glicemia não podem ser variáveis no estudo NHANES.

### 2.2 Rastreamento e escores de risco
O que é rastreamento; escores de risco por questionário; **escore de risco da ADA** (7 perguntas).

### 2.3 Aprendizado de máquina para classificação
- Regressão logística (razão de chance, interpretabilidade);
- Árvores de decisão; **Random Forest** (bagging, média de árvores independentes);
- **XGBoost** (boosting: árvores em sequência corrigindo erros);
- SVM (breve, pois foi um dos candidatos).

### 2.4 Avaliação de modelos
- Matriz de confusão, **precisão, recall, F1**, e por que dependem do limiar e da prevalência;
- **AUC-ROC** (independe do limiar), **AUC-PR** (útil com poucos casos), **Brier** (calibração);
- Divisão treino/validação/teste, validação cruzada, **validação cruzada aninhada**;
- **Vazamento de dados** (ex.: escolher limiar ou hiperparâmetros olhando o conjunto de teste).

### 2.5 Trabalhos relacionados
Estudos com Pima, BRFSS e NHANES; faixa de desempenho relatada (AUC ~0,82–0,85 no Pima; ~0,82 no BRFSS);
estudos de validação de escores entre populações.

---

## 3. Materiais e métodos (12–16 páginas)

### 3.1 Bases de dados
Tabela com as quatro bases:

| Base | País/ano | N usado | Tipo | Uso no trabalho |
|---|---|---|---|---|
| Pima Indians | EUA | 768 | Exames clínicos | Modelo clínico |
| BRFSS 2015 | EUA | 253.680 | Questionário telefônico | Modelo comportamental; validação Brasil x EUA |
| VIGITEL 2023 | Brasil | 19.919 | Questionário telefônico | Validação Brasil x EUA |
| NHANES 2011-2018 | EUA | 17.504 | Exame físico e de sangue + questionário | Estudo de diabetes não diagnosticado |

Para cada base: origem, variáveis, definição do desfecho e limitações (Pima: só mulheres de uma etnia;
BRFSS: pré-diabetes e diabetes agrupados e autorreferidos; VIGITEL: autorreferido, sem colesterol;
NHANES: uma única HbA1c).

### 3.2 Modelo clínico
Zeros como "não medido" e imputação pela mediana; candidatos (Regressão Logística, XGBoost, Random Forest,
SVM); validação cruzada aninhada 5×3; **regra de escolha definida antes de rodar** (maior F1; mais simples se
dentro de 1 erro-padrão). Fonte: `otimizar_modelo_clinico.py`.

### 3.3 Modelo comportamental
8 variáveis; divisão 60/20/20; grade de hiperparâmetros na validação; limiar pela maior F1 na validação.
Fonte: `treinar_modelo_comportamental.py`.

### 3.4 Validação Brasil x EUA
Harmonização das variáveis (tabela da seção 1 de `dados_vigitel/ANALISE_CROSS_CULTURAL.md`); conferência no
dicionário oficial do VIGITEL; variáveis excluídas por definição diferente; mesmo pipeline nos dois países;
teste de transferência (modelo de um país avaliado no outro). Fonte: `validacao_cross_cultural.py`.

### 3.5 Estudo de diabetes não diagnosticado (NHANES)
População sem diagnóstico; desfecho HbA1c ≥ 6,5%; 21 variáveis fixadas a priori; validação cruzada aninhada;
comparação com o escore ADA e com a idade. Fonte: `estudo_nhanes_rastreamento.py`.

### 3.6 Ferramentas
Python, scikit-learn, XGBoost, pandas, Streamlit, Supabase, ReportLab.

---

## 4. Desenvolvimento do sistema (6–10 páginas)

- **Arquitetura:** Streamlit (interface) + Supabase (autenticação e banco) + modelos em JSON; deploy no
  Streamlit Cloud. Um diagrama simples ajuda.
- **Funcionalidades:** cadastro/login com sessão persistente; perfil; os dois modelos; medidor de risco;
  fatores de risco e recomendações; PDF; histórico.
- **Decisões de engenharia que valem citar:**
  - modelos salvos em JSON (sem pickle) para funcionarem em versões diferentes das bibliotecas, com
    conferência de que o resultado é idêntico ao do treino;
  - uma conexão com o banco por sessão de usuário;
  - campos do formulário alinhados ao que o modelo espera (glicemia TOTG 2h, pressão diastólica).
- **Capturas de tela** das principais telas (login, modelo clínico com resultado, histórico, PDF).

---

## 5. Resultados (10–14 páginas)

### 5.1 Modelo clínico
Tabela dos 4 candidatos (`resultados_otimizacao_clinico.json`):

| Modelo | F1 | AUC | Recall | Precisão |
|---|---|---|---|---|
| Random Forest | 0,691 ± 0,038 | 0,839 | 82% | 60% |
| XGBoost | 0,678 ± 0,037 | 0,832 | 80% | 59% |
| Regressão Logística | 0,669 ± 0,042 | 0,833 | 80% | 58% |
| SVM | 0,669 ± 0,041 | 0,827 | 81% | 58% |

### 5.2 Modelo comportamental
F1 0,46; AUC 0,82; recall 61%; precisão 37%; todas as 16 configurações de XGBoost entre 0,819 e 0,822 de AUC
(`modelo_comportamental_meta.json`).

### 5.3 Validação Brasil x EUA
Figuras `dados_vigitel/fig_prevalencia_faixa_etaria.png`, `fig_auc_transferencia.png`,
`fig_importancia_variaveis.png` e a tabela de transferência (AUC: Brasil→Brasil 0,813; EUA→EUA 0,814;
EUA→Brasil 0,814; Brasil→EUA 0,773).

### 5.4 Diabetes não diagnosticado (NHANES)
Figuras `dados_nhanes/fig_curva_rastreamento.png` e `fig_razoes_chance.png`. Regressão Logística
AUC 0,827 contra 0,762 do escore ADA; testa 31% da população para achar 80% dos casos, contra 40% do ADA.

### 5.5 Sistema
Resultado dos testes funcionais (todas as páginas, casos de baixo e alto risco, geração de PDF).

---

## 6. Discussão (5–8 páginas)

- **O limite está nos dados, não no algoritmo:** quatro algoritmos ficaram a menos de 0,02 de F1 no Pima, e
  16 configurações a menos de 0,004 de AUC no BRFSS; o que mais pesa é ter a glicose medida (Pima 0,84 com
  TOTG; NHANES 0,83 sem glicose, mas com mais variáveis e dados).
- **Comparabilidade:** por que o F1 do clínico (35% de prevalência) não se compara ao do NHANES (3,8%).
- **Generalização entre países:** os mesmos fatores (pressão, idade, saúde autoavaliada, IMC) ordenam o
  risco igualmente bem no Brasil e nos EUA.
- **Rigor metodológico:** escolhas feitas só com dados de treino; regras definidas antes de rodar; em versões
  anteriores do projeto, o limiar escolhido no teste inflava o F1 (0,72 contra 0,69 com método correto).
- **Limitações:** dados autorreferidos; Pima restrito; pré-diabetes agrupado no BRFSS; anos diferentes entre
  pesquisas; modelagem do NHANES sem pesos amostrais; ausência de validação clínica prospectiva.

---

## 7. Conclusão (2–3 páginas)

Retomar cada objetivo específico e dizer como foi atingido. **Trabalhos futuros:** dados brasileiros com
exames (PNS, ELSA-Brasil), HbA1c e glicemia de jejum, avaliação com usuários reais, calibração por população.

---

## Referências para buscar e conferir

- Smith et al. (1988), artigo original do dataset Pima (algoritmo ADAP).
- Breiman (2001), *Random Forests*.
- Chen e Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*.
- Varma e Simon (2006), viés na estimativa de erro com validação cruzada (base da validação aninhada).
- Bang et al. (2009), desenvolvimento do escore de risco da ADA.
- American Diabetes Association, *Standards of Care in Diabetes* (critérios diagnósticos).
- CDC: documentação do BRFSS e do NHANES; Ministério da Saúde: relatório VIGITEL 2023.
- Teboul, *Diabetes Health Indicators Dataset* (Kaggle), versão do BRFSS 2015 usada.

---

## Apêndices sugeridos

- Dicionário de variáveis de cada base (harmonização Brasil x EUA).
- Hiperparâmetros finais dos modelos.
- Instruções para reproduzir os resultados (tabela "Retreinar os modelos" do README).
