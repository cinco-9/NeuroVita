# 4 DESENVOLVIMENTO DO SISTEMA

> **Rascunho.** Revise e reescreva com a sua voz. Os trechos marcados com **[CONFERIR]** dependem da
> configuração do seu projeto no Supabase. Substitua as indicações de figura por capturas de tela reais.

Este capítulo descreve a aplicação web desenvolvida para disponibilizar os modelos do Capítulo 3: sua
arquitetura, as funcionalidades oferecidas ao usuário, a forma como os modelos são executados e as decisões de
engenharia adotadas para garantir que os resultados da aplicação sejam idênticos aos obtidos no treinamento.

## 4.1 Visão geral

O sistema é uma aplicação web em que o usuário cria uma conta, preenche um perfil de saúde e estima o seu risco
de diabetes tipo 2 por dois caminhos: o **modelo clínico**, a partir de exames laboratoriais, e o **modelo
comportamental**, a partir de um questionário que dispensa exames. Cada avaliação apresenta a probabilidade
estimada, os fatores de risco identificados e recomendações, pode ser exportada em PDF e fica registrada em um
histórico pessoal.

A aplicação está publicada em https://neurovita-9jmmlcrm2ewvcxrvunmyhj.streamlit.app e o código-fonte está
disponível em repositório público (Apêndice X).

## 4.2 Arquitetura

A arquitetura é composta por três partes (Figura X):

1. **Interface e lógica da aplicação**, desenvolvidas com o *framework* Streamlit, em Python. O Streamlit
   gera a interface web a partir de código Python e executa, no servidor, tanto as telas quanto os modelos.
2. **Autenticação e banco de dados**, fornecidos pelo Supabase, plataforma baseada em PostgreSQL. O Supabase
   gerencia contas de usuário (cadastro, login e sessões) e armazena perfis e avaliações.
3. **Modelos de aprendizado de máquina**, armazenados em arquivos JSON no próprio repositório e carregados pela
   aplicação na primeira utilização.

A publicação é feita no Streamlit Community Cloud, conectado ao repositório no GitHub: cada alteração enviada ao
ramo principal é implantada automaticamente. As credenciais do Supabase ficam na área de segredos da
plataforma, fora do código.

**Figura X – Arquitetura do sistema** *(sugestão de diagrama)*

```
 Navegador do usuário
        │  HTTPS
        ▼
 Streamlit Community Cloud ──────────────► Supabase
 ┌──────────────────────────────┐          ┌───────────────────────────┐
 │ app_diabetes.py  (telas)     │  login   │ Auth (contas e sessões)   │
 │ auth.py          (sessão)    │◄────────►│ PostgreSQL                │
 │ supabase_db.py   (banco)     │  dados   │  • user_profiles (perfis) │
 │ modelos JSON + numpy/XGBoost │          │  • predicoes (avaliações) │
 │ pdf_generator.py (PDF)       │          └───────────────────────────┘
 └──────────────────────────────┘
        ▲
        │ implantação automática
   Repositório GitHub
```

O Quadro X resume os módulos do código.

**Quadro X – Módulos da aplicação**

| Módulo | Responsabilidade |
|---|---|
| `app_diabetes.py` | Páginas, formulários, execução dos modelos e exibição dos resultados |
| `auth.py` | Cadastro, login, logout, sessão persistente, recuperação de senha e perfil |
| `supabase_db.py` | Acesso ao banco: gravação e consulta de avaliações |
| `pdf_generator.py` | Geração do relatório em PDF |
| `gauge_component.py` | Medidor visual de risco |
| `analise_temporal.py` | Estatísticas e gráficos de evolução no histórico |

## 4.3 Banco de dados

O banco possui duas tabelas, além das tabelas de autenticação gerenciadas pelo próprio Supabase:

- **`user_profiles`**: dados do perfil (nome, idade, sexo, altura, peso, IMC, histórico familiar de diabetes,
  alergias e medicações), vinculados ao identificador do usuário;
- **`predicoes`**: cada avaliação realizada, com o tipo de modelo, os dados informados (em JSON), o resultado
  (risco baixo ou elevado), a probabilidade, a data e o identificador do usuário.

O script de criação das tabelas (`create_users_table.sql`) habilita *Row Level Security* (RLS) nas duas
tabelas, com políticas que permitem a cada usuário ler e gravar apenas os próprios registros. **[CONFERIR: se
o script foi executado no projeto do Supabase.]**

## 4.4 Funcionalidades

### 4.4.1 Cadastro, login e sessão

O usuário cria uma conta com nome, e-mail e senha (mínimo de seis caracteres) e entra com e-mail e senha. Para
que a sessão não se perca ao recarregar a página, a aplicação guarda no navegador um token de renovação de
sessão emitido pelo Supabase, válido por 30 dias. Ao abrir a aplicação, esse token é usado para restaurar o
login; a cada renovação o Supabase emite um novo token, que substitui o anterior. O botão "Sair" encerra a
sessão apenas no navegador em uso e apaga o token.

A recuperação de senha por código enviado por e-mail está implementada, mas só é exibida quando o envio de
e-mails do Supabase está configurado com um servidor SMTP próprio. **[CONFERIR: se a recuperação foi ativada.]**

**Figura X – Tela de login**

### 4.4.2 Perfil

A página "Meu Perfil" registra dados pessoais e de saúde, calcula o IMC automaticamente e os utiliza para
preencher os formulários dos modelos (idade, sexo e IMC). Alergias informadas são destacadas junto às
recomendações.

### 4.4.3 Modelo clínico

O formulário recebe as oito variáveis do modelo: glicemia do TOTG de 2 horas, insulina, pressão diastólica,
espessura da dobra cutânea, IMC, idade, número de gestações e função de pedigree. Textos de ajuda orientam o
preenchimento, por exemplo, que a glicemia esperada é a do teste de tolerância e não a de jejum, que a pressão
é o valor diastólico, e que insulina e dobra cutânea podem ser deixadas em zero quando o exame não foi feito.

O resultado apresenta (Figura X):

- a classificação (risco baixo ou elevado) e a probabilidade estimada;
- um medidor visual da probabilidade;
- os fatores de risco identificados, com limites clínicos (por exemplo, glicemia do TOTG de 140 a 199 mg/dL
  indica tolerância à glicose diminuída, e 200 mg/dL ou mais, faixa de diabetes);
- recomendações personalizadas;
- a opção de baixar o relatório em PDF.

**Figura X – Resultado do modelo clínico**

### 4.4.4 Modelo comportamental

O questionário possui oito perguntas: idade, sexo, IMC, saúde geral, pressão alta, colesterol alto, atividade
física e tabagismo. A tela informa que o modelo estima o risco de pré-diabetes ou diabetes, conforme a
definição da base de treinamento. O resultado segue o mesmo formato do modelo clínico.

### 4.4.5 Comparação e histórico

A página de comparação apresenta o desempenho dos dois modelos e propõe seu uso combinado: o modelo
comportamental como triagem inicial, sem custo, e o clínico quando há exames. A página de histórico lista as
avaliações do usuário, contabiliza avaliações por modelo e de risco elevado, e mostra a evolução da
probabilidade entre a primeira e a última avaliação.

**Figura X – Histórico**

### 4.4.6 Relatório em PDF

O relatório contém os dados do paciente, o resultado com a probabilidade, os dados informados e as
recomendações, e um aviso de que o sistema não substitui avaliação médica.

## 4.5 Execução dos modelos na aplicação

Um cuidado central do desenvolvimento foi garantir que a aplicação produza exatamente as mesmas probabilidades
obtidas no treinamento. Três decisões foram tomadas com esse objetivo.

**Formato portável dos modelos.** Modelos de aprendizado de máquina costumam ser salvos com o formato *pickle*
do Python, que depende das versões exatas das bibliotecas. Durante o desenvolvimento, verificou-se que modelos
salvos com uma versão do NumPy não podiam ser abertos com a versão instalada no servidor. Por isso, os modelos
foram exportados em JSON:

- o modelo comportamental, no formato nativo do XGBoost, carregado pela biblioteca sem depender do
  scikit-learn. Testou-se a leitura em versões diferentes do XGBoost: na versão 3.0 as probabilidades
  divergiam, e na versão 3.4, usada no treinamento, eram idênticas; por isso a aplicação exige a versão 3.4 ou
  superior;
- o modelo clínico (Random Forest), exportado como a estrutura das suas 300 árvores (variável, limite e
  probabilidade de cada nó), junto com os parâmetros de imputação e padronização. A aplicação percorre as
  árvores com NumPy, sem precisar do scikit-learn no servidor.

**Reprodução exata do pré-processamento.** A aplicação repete, na mesma ordem, os passos do treinamento: valores
zero nos exames são tratados como não medidos e substituídos pela mediana do treinamento; as variáveis são
padronizadas; e os valores são arredondados para precisão simples (32 bits) antes da comparação com os limites
das árvores, como faz a biblioteca original.

**Verificação.** As probabilidades calculadas pela aplicação foram comparadas com as da biblioteca de treinamento
para as 768 pacientes da base Pima, com diferença máxima igual a zero.

## 4.6 Outras decisões de engenharia

- **Isolamento entre usuários:** o cliente do Supabase, que guarda a sessão de login, é criado separadamente
  para cada sessão da aplicação, evitando que usuários simultâneos compartilhem a mesma sessão.
- **Interface:** tema escuro com paleta de cores definida em variáveis CSS e layout compacto, com o resultado e
  o medidor lado a lado.
- **Tratamento de erros:** se um modelo não puder ser executado, a aplicação exibe uma mensagem de erro em vez de
  apresentar uma estimativa substituta.

## 4.7 Testes

A aplicação foi testada de duas formas:

- **testes funcionais automatizados**, com a ferramenta de testes do Streamlit, que percorrem todas as páginas
  e executam os dois modelos com perfis de baixo e de alto risco, incluindo a geração do PDF (resultados na
  Seção 5.5);
- **inspeção visual**, com capturas de tela automáticas de todas as páginas em navegador, para verificar o
  layout.
