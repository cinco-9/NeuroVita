# 3 MATERIAIS E MÉTODOS

> **Rascunho.** Revise, ajuste ao modelo de formatação da sua instituição e reescreva com a sua voz.
> Trechos `[CITAR: ...]` indicam onde inserir referências. Os números vêm dos scripts e arquivos citados.

Este capítulo descreve as bases de dados utilizadas, o tratamento aplicado a cada uma, o desenvolvimento dos
modelos de classificação e os protocolos de avaliação. O trabalho é composto por quatro estudos: (i) o modelo
clínico, baseado em exames laboratoriais; (ii) o modelo comportamental, baseado em questionário; (iii) a
validação dos modelos entre as populações do Brasil e dos Estados Unidos; e (iv) um estudo complementar sobre
a identificação de diabetes não diagnosticado. Todos os procedimentos foram implementados em Python e podem
ser reproduzidos a partir dos scripts do repositório do projeto (Apêndice X).

Em todos os estudos adotou-se o mesmo princípio: nenhuma decisão de modelagem (escolha de algoritmo,
hiperparâmetros ou limiar de decisão) foi tomada com base nos dados usados para medir o desempenho final,
evitando o vazamento de informação que tende a superestimar os resultados [CITAR: Varma e Simon, 2006].

## 3.1 Bases de dados

Foram utilizadas quatro bases de dados públicas, resumidas no Quadro 1.

**Quadro 1 – Bases de dados utilizadas**

| Base | Origem | Registros utilizados | Tipo de informação | Uso no trabalho |
|---|---|---|---|---|
| Pima Indians Diabetes | NIDDK, EUA | 768 | Exames clínicos | Modelo clínico |
| BRFSS 2015 | CDC, EUA | 253.680 | Questionário telefônico | Modelo comportamental e validação Brasil x EUA |
| VIGITEL 2023 | Ministério da Saúde, Brasil | 19.919 | Questionário telefônico | Validação Brasil x EUA |
| NHANES 2011–2018 | CDC, EUA | 17.306 | Exame físico, exames de sangue e questionário | Estudo de diabetes não diagnosticado |

### 3.1.1 Pima Indians Diabetes

A base contém registros de 768 mulheres com 21 anos ou mais, de origem indígena Pima, residentes no Arizona
[CITAR: Smith et al., 1988]. Cada registro possui oito variáveis: número de gestações, concentração de glicose
plasmática duas horas após o teste oral de tolerância à glicose (TOTG), pressão arterial diastólica, espessura
da dobra cutânea do tríceps, insulina sérica, índice de massa corporal (IMC), função de pedigree de diabetes
(índice de histórico familiar) e idade. O desfecho indica o desenvolvimento de diabetes, presente em 34,9% das
pacientes.

Nessa base, valores iguais a zero em glicose, pressão diastólica, dobra cutânea, insulina e IMC são
fisiologicamente impossíveis e representam medidas não realizadas. A proporção desses valores foi de 1% para
glicose e IMC, 5% para pressão diastólica, 30% para dobra cutânea e 49% para insulina.

### 3.1.2 BRFSS 2015

O *Behavioral Risk Factor Surveillance System* (BRFSS) é um inquérito telefônico anual conduzido pelo Centers
for Disease Control and Prevention (CDC) [CITAR: CDC, BRFSS]. Foi utilizada a versão de 2015 disponibilizada
de forma tratada na plataforma Kaggle [CITAR: Teboul, Diabetes Health Indicators Dataset], com 253.680
entrevistas. Nessa versão, o desfecho agrupa pré-diabetes e diabetes autorreferidos, com prevalência de 13,9%.
Por isso, o modelo comportamental estima o risco de "pré-diabetes ou diabetes".

### 3.1.3 VIGITEL 2023

O VIGITEL (Vigilância de Fatores de Risco e Proteção para Doenças Crônicas por Inquérito Telefônico) é o
inquérito telefônico anual do Ministério da Saúde realizado nas capitais brasileiras [CITAR: Brasil, VIGITEL
2023]. A base de 2023 contém 21.690 entrevistas. As variáveis foram identificadas no dicionário oficial do
inquérito (Quadro 2). Foram excluídos os registros com resposta ausente ("não sabe" ou "não quis informar")
nas variáveis utilizadas, com IMC fora do intervalo de 10 a 60 kg/m² ou idade fora de 18 a 100 anos,
resultando em 19.919 registros.

**Quadro 2 – Variáveis do VIGITEL 2023 utilizadas**

| Variável | Código no VIGITEL | Definição |
|---|---|---|
| Diabetes | `q76` | Diagnóstico médico de diabetes (1 = sim) |
| Idade | `q6` | Idade em anos |
| Sexo | `q7` | 1 = masculino |
| IMC | `imc` | Calculado pelo próprio inquérito |
| Pressão alta | `q75` | Diagnóstico médico de hipertensão |
| Saúde autoavaliada | `q74` | 1 = muito bom a 5 = muito ruim |
| Fumante | `fumante` | Indicador de fumante atual |
| Atividade física | `ativo_livre` | Indicador de atividade física suficiente no tempo livre |
| Frutas | `frutareg` | Consumo de frutas em 5 ou mais dias da semana |
| Peso amostral | `pesorake` | Peso de pós-estratificação |

O VIGITEL não pergunta sobre diagnóstico de colesterol elevado, apenas sobre a realização do exame; essa
variável, portanto, não pôde ser utilizada. A prevalência de diabetes foi de 12,6% na amostra e de 10,1%
quando ponderada pelos pesos amostrais.

### 3.1.4 NHANES 2011–2018

O *National Health and Nutrition Examination Survey* (NHANES) é um inquérito do CDC que combina entrevista,
exame físico e coleta de sangue em amostra representativa da população dos Estados Unidos [CITAR: CDC, NHANES].
Foram utilizados quatro ciclos (2011–2012, 2013–2014, 2015–2016 e 2017–2018). Foram incluídos adultos com 20
anos ou mais, não gestantes, com hemoglobina glicada (HbA1c) medida e **sem diagnóstico prévio de diabetes**
(17.504 pessoas). Foram excluídas 198 pessoas sem IMC medido, necessário para o cálculo do escore de risco
usado como comparação (Seção 3.5.3), resultando em 17.306 pessoas.

## 3.2 Modelo clínico

### 3.2.1 Pré-processamento

Os valores zero nas cinco variáveis citadas na Seção 3.1.1 foram tratados como ausentes. Para Regressão
Logística, Random Forest e SVM, eles foram imputados pela mediana e as variáveis foram padronizadas (média zero
e desvio-padrão um); imputação e padronização foram ajustadas somente com os dados de treino de cada partição,
dentro de um *pipeline*, para evitar vazamento de informação. O XGBoost recebeu os valores ausentes diretamente,
pois os trata de forma nativa.

### 3.2.2 Algoritmos candidatos

Foram comparados quatro algoritmos, com as grades de hiperparâmetros do Quadro 3: Regressão Logística,
Random Forest [CITAR: Breiman, 2001], XGBoost [CITAR: Chen e Guestrin, 2016] e Máquina de Vetores de Suporte
(SVM) com kernel RBF. Para alguns candidatos também se avaliou, como hiperparâmetro, a inclusão de um indicador
de valor ausente e de cinco variáveis combinadas (produtos de IMC × idade, glicose × IMC, glicose × idade,
insulina × glicose e pressão × IMC).

**Quadro 3 – Grades de hiperparâmetros do modelo clínico**

| Algoritmo | Hiperparâmetros avaliados |
|---|---|
| Regressão Logística | C ∈ {0,01; 0,1; 1; 10}; indicador de ausente (sim/não); variáveis combinadas (sim/não) |
| XGBoost (200 árvores, taxa 0,05) | profundidade ∈ {2, 3, 4}; peso mínimo por nó ∈ {1, 5}; variáveis combinadas (sim/não) |
| Random Forest (300 árvores) | profundidade ∈ {4, 6, sem limite}; amostras mínimas por folha ∈ {1, 5}; indicador de ausente (sim/não) |
| SVM (kernel RBF) | C ∈ {0,3; 1; 3}; indicador de ausente (sim/não) |

### 3.2.3 Protocolo de avaliação

Utilizou-se validação cruzada aninhada [CITAR: Varma e Simon, 2006]. O laço externo consistiu em validação
cruzada estratificada de 5 partições repetida 3 vezes (15 avaliações). Em cada partição externa, apenas os
dados de treino foram usados para:

1. escolher os hiperparâmetros, pela maior AUC em validação cruzada interna de 5 partições;
2. escolher o limiar de decisão, pelo maior F1 nas predições fora da amostra da validação interna, em uma
   grade de 0,10 a 0,70.

O modelo resultante foi então avaliado na partição de teste externa, calculando-se F1, precisão, recall, AUC
e escore de Brier.

### 3.2.4 Critério de escolha

A regra de escolha foi definida antes da execução: seleciona-se o algoritmo com maior F1 médio; caso um
algoritmo mais simples apresente F1 a menos de um erro-padrão do melhor, ele é preferido. A ordem de
simplicidade considerada foi: Regressão Logística, XGBoost, Random Forest e SVM. O modelo final foi ajustado
com todos os dados pelo mesmo procedimento interno (hiperparâmetros e limiar).

## 3.3 Modelo comportamental

Foram utilizadas oito variáveis do BRFSS que podem ser respondidas sem exames: pressão alta, colesterol alto,
IMC, tabagismo, atividade física, saúde autoavaliada, faixa etária e sexo. Devido ao tamanho da base, foi
adotada divisão estratificada em treino (60%), validação (20%) e teste (20%).

Avaliou-se o XGBoost em uma grade de 16 combinações (profundidade ∈ {3, 4, 5, 6}; número de árvores ∈ {300, 600};
peso mínimo por nó ∈ {1, 10}; taxa de aprendizado 0,05; subamostragem de 90% das linhas e colunas), escolhendo
a configuração com maior AUC no conjunto de validação. Uma Regressão Logística foi treinada como referência. O
limiar de decisão foi escolhido pelo maior F1 no conjunto de validação, e as métricas finais foram calculadas
uma única vez no conjunto de teste.

## 3.4 Validação entre Brasil e Estados Unidos

### 3.4.1 Harmonização das variáveis

Para comparar as duas populações, foram usadas apenas as variáveis com definição equivalente no VIGITEL e no
BRFSS: idade, sexo, IMC, diagnóstico de pressão alta e saúde autoavaliada. A idade do VIGITEL, registrada em
anos, foi convertida para as 13 faixas do BRFSS (18–24, 25–29, ..., 75–79, 80 ou mais). Foram excluídas as
variáveis cujas definições diferem entre os inquéritos (Quadro 4).

**Quadro 4 – Variáveis excluídas da comparação entre países**

| Variável | BRFSS | VIGITEL |
|---|---|---|
| Colesterol alto | Diagnóstico autorreferido | Não perguntado |
| Fumante | Fumou 100 cigarros na vida | Fuma atualmente |
| Atividade física | Qualquer atividade no último mês | ≥ 150 min/semana no tempo livre |
| Frutas | Consumo ≥ 1 vez/dia | Consumo ≥ 5 dias/semana |

### 3.4.2 Modelos e transferência

O mesmo procedimento foi aplicado aos dois países: XGBoost (400 árvores, profundidade 5, taxa 0,05), divisão
estratificada 60/20/20 e limiar escolhido pelo maior F1 no conjunto de validação. Cada modelo foi avaliado no
conjunto de teste do próprio país e no conjunto de teste do outro país (transferência). A AUC foi adotada como
métrica principal, por não depender do limiar nem da prevalência. A importância das variáveis foi medida pelo
ganho médio nas árvores.

### 3.4.3 Análise descritiva

Compararam-se prevalências, distribuição etária, IMC e fatores de risco. A diferença de prevalência foi
testada pelo teste qui-quadrado. Como as amostras têm composições etárias diferentes, calculou-se também a
prevalência brasileira padronizada pela distribuição etária do BRFSS.

## 3.5 Estudo de diabetes não diagnosticado (NHANES)

### 3.5.1 Desfecho e variáveis

O desfecho foi definido como HbA1c ≥ 6,5%, critério diagnóstico de diabetes [CITAR: ADA, Standards of Care],
na população sem diagnóstico prévio. Como a HbA1c e a glicemia definem o desfecho, elas **não** foram usadas
como variáveis preditoras, o que tornaria a análise circular.

Foram definidas *a priori* 21 variáveis disponíveis em uma consulta comum: idade, sexo, raça/etnia (cinco
categorias, com branco não hispânico como referência), IMC, razão cintura/altura, pressão sistólica e
diastólica (média de até três medidas), colesterol total, HDL, diagnóstico de hipertensão, histórico familiar
de diabetes, atividade física (prática de atividade moderada ou vigorosa no trabalho ou no lazer), tabagismo
atual, ex-tabagismo, escolaridade, razão da renda familiar sobre a linha de pobreza e diabetes gestacional.
Para Regressão Logística e Random Forest, valores ausentes foram imputados pela mediana e as variáveis
padronizadas, com ajuste feito dentro de cada partição de treino; o XGBoost tratou os ausentes de forma nativa.

### 3.5.2 Modelos e avaliação

Foram comparados Regressão Logística (C ∈ {0,01; 0,1; 1}), XGBoost (profundidade ∈ {2, 3, 4}; peso mínimo
por nó ∈ {1, 10}) e Random Forest (profundidade ∈ {6, 10}; amostras mínimas por folha ∈ {10, 30}), com
validação cruzada aninhada (5 partições × 3 repetições no laço externo, 5 partições no interno). A escolha seguiu
regra definida antes da execução: maior AUC média, preferindo o modelo mais simples dentro de um erro-padrão.

Além da AUC, foram calculados a AUC-PR, o escore de Brier e duas métricas de rastreamento: (i) a fração da
população que precisa ser testada, em ordem decrescente de risco, para encontrar 80% dos casos; e (ii) a
fração de casos encontrada ao testar o mesmo número de pessoas indicado pelo escore da ADA.

### 3.5.3 Comparação com o escore de risco da ADA

O escore de risco da American Diabetes Association [CITAR: Bang et al., 2009] foi calculado nas mesmas pessoas:
idade (0 a 3 pontos), sexo masculino (1), diabetes gestacional (1), histórico familiar (1), hipertensão (1),
inatividade física (1) e IMC (0 a 3 pontos); escore ≥ 5 indica alto risco. A diferença de AUC entre o modelo e
o escore foi calculada em cada partição externa.

### 3.5.4 Associação das variáveis

Para interpretação, ajustou-se uma Regressão Logística com todos os dados e calcularam-se as razões de chance
por desvio-padrão de cada variável. A prevalência populacional foi estimada com os pesos amostrais do NHANES; a
modelagem foi feita sem ponderação.

## 3.6 Ferramentas

Os experimentos foram implementados em Python 3.14, com as bibliotecas pandas 3.0, NumPy 2.5, scikit-learn 1.9
e XGBoost 3.4. A aplicação web foi desenvolvida com Streamlit 1.63, com autenticação e banco de dados no
Supabase, relatórios em PDF gerados com ReportLab e publicação no Streamlit Community Cloud. As figuras foram
produzidas com Matplotlib.
