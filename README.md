# Sistema de Predição de Risco de Diabetes Tipo 2

Aplicação web que estima o risco de diabetes tipo 2 com dois modelos de Machine Learning: um baseado em
**exames laboratoriais** e outro baseado em **questionário de hábitos e saúde**, sem exames. O projeto inclui uma
validação dos modelos entre populações do **Brasil (VIGITEL 2023)** e dos **EUA (BRFSS 2015)**.

Autor: David Reis · 2026

**Demo online:** https://neurovita-9jmmlcrm2ewvcxrvunmyhj.streamlit.app

> Sistema com finalidade educacional e de triagem. Não substitui avaliação médica.

---

## Funcionalidades

- **Cadastro e login** (Supabase Auth), com a sessão mantida ao recarregar a página e recuperação de senha por código (ativada quando o envio de email do Supabase estiver configurado).
- **Perfil do usuário** (idade, sexo, altura, peso, IMC, histórico familiar, alergias, medicações), usado para preencher os formulários.
- **Modelo Clínico**: risco a partir de exames (glicemia do TOTG 2h, insulina, pressão diastólica, IMC etc.).
- **Modelo Comportamental**: risco a partir de 8 perguntas (idade, sexo, IMC, saúde geral, pressão alta, colesterol, atividade física, fumo).
- **Resultado** com probabilidade, medidor visual, fatores de risco identificados e recomendações.
- **Relatório em PDF** para levar ao médico.
- **Histórico** das avaliações de cada usuário.
- **Privacidade (LGPD)**: termo de consentimento no cadastro, cada usuário só acessa os próprios dados (RLS no Supabase) e pode excluir a conta com todos os dados pelo próprio app.

## Modelos

| Modelo | Algoritmo | Dados | Entrada | F1 | Precisão | Recall | AUC |
|---|---|---|---|---|---|---|---|
| Clínico | Random Forest | Pima Indians (768 pacientes) | 8 exames | 0,69 ± 0,04 | 60% | 82% | 0,84 |
| Comportamental | XGBoost | BRFSS 2015 (253.680 entrevistas) | 8 respostas de questionário | 0,46 | 37% | 61% | 0,82 |

Em nenhum dos dois o conjunto que mede o desempenho é usado para escolher modelo, hiperparâmetros ou limiar:

- **Clínico** ([`otimizar_modelo_clinico.py`](otimizar_modelo_clinico.py)): validação cruzada aninhada (5 partes × 3
  repetições) comparando Regressão Logística, XGBoost, Random Forest e SVM. Zeros em glicemia, pressão, dobra
  cutânea, insulina e IMC são tratados como "não medido" e imputados pela mediana. A regra de escolha foi definida
  antes de rodar: maior F1 médio, preferindo o modelo mais simples se estiver a menos de 1 erro-padrão.
  Resultados de todos os candidatos: [`resultados_otimizacao_clinico.json`](resultados_otimizacao_clinico.json).
- **Comportamental** ([`treinar_modelo_comportamental.py`](treinar_modelo_comportamental.py)): divisão 60/20/20;
  hiperparâmetros e limiar escolhidos na validação, métricas no teste.

Os dois ficam na faixa que a literatura metodologicamente cuidadosa relata para esses dados (AUC ~0,82–0,85 no
Pima; ~0,82–0,83 no BRFSS).

**Importante sobre o modelo clínico:** o dataset Pima mede a glicemia **2 horas após ingerir glicose** (TOTG) e a
pressão **diastólica**. Informar a glicemia de jejum subestima o risco.

## Validação Brasil x EUA

[`validacao_cross_cultural.py`](validacao_cross_cultural.py) treina o mesmo pipeline nos dois países, usando só as
variáveis com definição equivalente nas duas pesquisas (idade, sexo, IMC, pressão alta, saúde autoavaliada), e
testa cada modelo também no outro país:

| Treinado em → testado em | AUC |
|---|---|
| Brasil → Brasil | 0,813 |
| EUA → EUA | 0,825 |
| EUA → Brasil | 0,818 |
| Brasil → EUA | 0,787 |

O modelo treinado nos EUA ordena o risco da população brasileira tão bem quanto um modelo treinado no Brasil.
Relatório completo, figuras e limitações: [dados_vigitel/ANALISE_CROSS_CULTURAL.md](dados_vigitel/ANALISE_CROSS_CULTURAL.md).

## Estudo complementar: diabetes não diagnosticado (NHANES)

[`estudo_nhanes_rastreamento.py`](estudo_nhanes_rastreamento.py) usa 17.306 adultos **sem diagnóstico** de diabetes
do NHANES 2011-2018 (inquérito dos EUA com exame de sangue) para identificar quem tem HbA1c ≥ 6,5% e não sabe,
usando só informações de uma consulta comum (sem exame de glicose). Protocolo definido antes de rodar, validação
cruzada aninhada e comparação com o escore de risco da American Diabetes Association (ADA) nas mesmas pessoas:

| | AUC | Testar para achar 80% dos casos |
|---|---|---|
| **Regressão Logística (modelo)** | **0,827 ± 0,018** | **31%** |
| Escore ADA | 0,762 | 40% |
| Só idade | 0,673 | 53% |

Para achar 80% dos casos, o modelo precisa testar **23% menos pessoas** que o escore da ADA (31% contra 40% da população).
Relatório: [dados_nhanes/ESTUDO_RASTREAMENTO_NHANES.md](dados_nhanes/ESTUDO_RASTREAMENTO_NHANES.md).

## Como rodar localmente

Requisitos: Python 3.11+ e um projeto no [Supabase](https://supabase.com) (veja [SUPABASE_SETUP.md](SUPABASE_SETUP.md)).

```bash
pip install -r requirements.txt
```

Crie um arquivo `.env` na raiz:

```
SUPABASE_URL=https://<seu-projeto>.supabase.co
SUPABASE_KEY=<chave anon>
# opcional: mostra "Esqueci minha senha" (exige SMTP configurado no Supabase)
RECUPERACAO_SENHA_ATIVA=1
# opcional: email exibido no termo de privacidade para pedidos sobre os dados
CONTATO_PRIVACIDADE=
```

No Supabase SQL Editor, rode `create_table.sql`, `create_users_table.sql` e `privacidade_lgpd.sql`, nessa ordem.

```bash
python -m streamlit run app_diabetes.py
```

O app abre em http://localhost:8501. Deploy no Streamlit Cloud: [DEPLOY_STREAMLIT.md](DEPLOY_STREAMLIT.md).

## Retreinar os modelos

| O quê | Comando | Gera |
|---|---|---|
| Modelo clínico | `python otimizar_modelo_clinico.py` | `modelo_clinico.json`, `modelo_clinico_meta.json`, `resultados_otimizacao_clinico.json` |
| Modelo comportamental | `python treinar_modelo_comportamental.py` | `modelo_comportamental.json`, `modelo_comportamental_meta.json` |
| Dados do VIGITEL | `python mapear_vigitel_completo.py` | `dados_vigitel/vigitel_2023_processado.csv` |
| Validação Brasil x EUA | `python baixar_brfss.py` e `python validacao_cross_cultural.py` | relatório, JSON e figuras em `dados_vigitel/` |
| Estudo NHANES | `python baixar_nhanes.py` e `python estudo_nhanes_rastreamento.py` | relatório, JSON e figuras em `dados_nhanes/` |
| Comparação com a literatura | `python comparacao_literatura.py` | [COMPARACAO_LITERATURA.md](COMPARACAO_LITERATURA.md) |

Os scripts de treino precisam também de `scikit-learn`, `imbalanced-learn`, `scipy` e `openpyxl`, que não estão no
`requirements.txt` (ele lista só o necessário para o app). Os modelos são salvos em JSON, sem pickle: o
comportamental no formato nativo do XGBoost (o app exige `xgboost>=3.4`, versão usada no treino) e o clínico
como as árvores do Random Forest, avaliadas pelo app com numpy (resultado idêntico ao do scikit-learn, conferido
no próprio script de treino).

## Estrutura

```
app_diabetes.py                  Aplicação Streamlit (páginas, formulários, resultados)
auth.py                          Login, cadastro, sessão persistente, recuperação de senha, perfil
supabase_db.py                   Acesso ao banco (uma conexão por sessão de usuário)
pdf_generator.py                 Relatório em PDF
gauge_component.py               Medidor de risco
analise_temporal.py              Gráficos do histórico
modelo_*.json                    Modelos treinados e metadados (features, limiar, imputação, normalização)
otimizar_modelo_clinico.py       Seleção e treino do modelo clínico (validação cruzada aninhada)
melhorar_precisao.py             Experimento anterior do modelo clínico (histórico)
treinar_modelo_comportamental.py Treino do modelo comportamental
mapear_vigitel_completo.py       Preparação dos dados do VIGITEL
validacao_cross_cultural.py      Validação Brasil x EUA
baixar_nhanes.py                 Download dos dados do NHANES
estudo_nhanes_rastreamento.py    Estudo de diabetes não diagnosticado (NHANES)
dados_vigitel/                   Dados, dicionário e resultados do VIGITEL
```

## Dados

- **Pima Indians Diabetes**: 768 mulheres de origem Pima (EUA), com exames clínicos.
- **BRFSS 2015**: inquérito telefônico do CDC (EUA). O modelo comportamental usa a versão tratada do Kaggle
  (pré-diabetes e diabetes agrupados); a validação Brasil x EUA usa os dados originais do CDC, baixados por
  `baixar_brfss.py`, com o desfecho só de diabetes.
- **VIGITEL 2023**: inquérito telefônico do Ministério da Saúde (Brasil), com dicionário oficial em `dados_vigitel/`.
- **NHANES 2011-2018**: inquérito do CDC (EUA) com exame físico e de sangue, baixado por `baixar_nhanes.py`.

## Limitações

- O dataset clínico é pequeno (768 pacientes) e restrito a mulheres de uma etnia.
- Nos inquéritos telefônicos o diabetes é **autorreferido**, o que subestima casos não diagnosticados.
- O VIGITEL não pergunta sobre colesterol alto, então essa variável não entra na comparação entre países.
- Na versão do BRFSS usada no modelo comportamental, **pré-diabetes e diabetes formam um único grupo**, então esse
  modelo estima o risco de "pré-diabetes ou diabetes".
