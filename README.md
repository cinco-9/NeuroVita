# 🩺 Sistema de Predição de Diabetes Tipo 2

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **TCC: Agente de IA para Estimativa de Risco de Diabetes Tipo 2 e Doenças Cardiovasculares**  
> Autor: David Reis | 2026

Sistema completo de predição de diabetes tipo 2 utilizando Machine Learning, com interface web interativa, explicabilidade de predições (SHAP) e geração de relatórios em PDF.

## 🌐 Demo Online

**Acesse o sistema funcionando:** [EM BREVE - Seguir DEPLOY.md]

> 📱 **Responsivo:** Funciona em desktop, tablet e mobile!

---

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Descoberta Científica](#-descoberta-científica)
- [Funcionalidades](#-funcionalidades)
- [Tecnologias](#-tecnologias)
- [Resultados](#-resultados)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Datasets](#-datasets)
- [Contribuição](#-contribuição)
- [Licença](#-licença)

---

## 🎯 Sobre o Projeto

Este projeto desenvolveu um sistema inteligente para **predição de risco de diabetes tipo 2** utilizando técnicas avançadas de Machine Learning. O sistema oferece:

- **2 modelos preditivos**: Clínico (baseado em exames) e Comportamental (baseado em hábitos)
- **Interface web completa** com autenticação e histórico personalizado
- **Explicabilidade** através de gráficos SHAP
- **Relatórios em PDF** para levar ao médico
- **Recomendações personalizadas** baseadas em fatores de risco individuais

### 🎓 Contexto Acadêmico

Trabalho de Conclusão de Curso focado em aplicações de Inteligência Artificial na área da saúde, com ênfase em medicina preventiva e diagnóstico precoce.

---

## 🔬 Descoberta Científica

### **Features Clínicas são 53% Superiores!**

Um dos principais achados deste trabalho foi a **comparação entre dois tipos de features**:

| Tipo | Dataset | Registros | F1-Score | Precisão | Recall |
|------|---------|-----------|----------|----------|--------|
| **🔬 Clínico** | Pima Indians | 768 | **0.72** | **63.4%** | **83.3%** |
| **📋 Comportamental** | BRFSS 2015 | 253,680 | 0.47 | 37.1% | 63.8% |
| **📊 Diferença** | - | - | **+53.4%** | **+70.8%** | **+30.5%** |

### Por quê?

- **Features clínicas** (glicose, insulina, pressão) medem **diretamente** o estado fisiológico
- **Features comportamentais** (hábitos de vida) medem apenas **fatores de risco indiretos**
- **Qualidade > Quantidade**: 768 registros clínicos superam 253k registros comportamentais!

**Conclusão:** Para diagnóstico preciso, **exames laboratoriais são insubstituíveis**. Hábitos de vida são úteis para triagem inicial.

---

## ✨ Funcionalidades

### 🔐 Autenticação e Perfil
- Sistema de login/cadastro com Supabase Auth
- Perfil pessoal com cálculo automático de IMC
- Histórico individual de predições
- Row Level Security (RLS) - cada usuário vê apenas seus dados

### 🔬 Modelo Clínico (Pima Indians)
- Predição baseada em 8 exames laboratoriais
- Feature Engineering: 16 features totais
- **F1-Score: 0.72** (Excelente!)
- Threshold otimizado: 0.35
- **Gráficos SHAP**: explicação visual da predição
- **Exportar PDF**: relatório profissional

### 📋 Modelo Comportamental (BRFSS)
- Predição baseada em hábitos de vida (SEM exames)
- Útil para triagem inicial
- **F1-Score: 0.47**
- Threshold otimizado: 0.24
- **Exportar PDF**: relatório completo

### 📊 Dashboard & Estatísticas
- **5 gráficos interativos**:
  1. Distribuição por risco (pizza)
  2. Predições por modelo (barras)
  3. Evolução temporal (linha)
  4. Evolução de probabilidades (barras)
  5. Distribuição de probabilidades (histograma)
- Comparação Clínico vs Comportamental
- Histórico completo de predições
- Estatísticas em tempo real
- **📊 Gauge de Risco Visual**: Medidor semicircular 0-100% 🆕
- **📈 Análise Temporal**: Evolução do risco ao longo do tempo 🆕

### 📊 Gauge de Risco Visual 🆕
- Medidor semicircular colorido (verde→amarelo→laranja→vermelho)
- Mostra probabilidade de 0-100% com agulha indicadora
- Zonas de risco claramente demarcadas
- Integrado em ambos os modelos
- **Visual impactante e fácil de entender!**

### 📈 Análise Temporal Pessoal 🆕
- **Gráfico de evolução**: linha do tempo mostrando progresso
- **Comparação antes vs agora**: primeira vs última avaliação
- **4 métricas de evolução**: primeira avaliação, atual, mudança total, período
- **Mensagens motivacionais**: feedback personalizado sobre melhora/piora
- **Funciona automaticamente**: basta ter 2+ predições

### 🎯 Recomendações Personalizadas
- Baseadas em **fatores de risco individuais**
- Ajustadas por **idade, sexo e histórico familiar**
- Diferenciam entre **alto e baixo risco**
- Incluem **alergias e medicações** do perfil

### 📄 Geração de PDF
- Relatório profissional completo
- Dados do paciente
- Resultado colorido (verde/vermelho)
- Gráficos SHAP (quando disponível)
- Recomendações personalizadas
- Aviso médico

---

## 🛠️ Tecnologias

### Machine Learning
- **XGBoost** - Modelo principal (F1=0.72)
- **LightGBM** - Ensemble
- **CatBoost** - Ensemble
- **Scikit-learn** - Pré-processamento e métricas
- **SMOTE** - Balanceamento de classes
- **SHAP** - Explicabilidade

### Otimização
- **Optuna** - Otimização Bayesiana de hiperparâmetros
- **GridSearchCV** - Busca em grade
- **Stratified K-Fold** - Validação cruzada robusta

### Frontend/Backend
- **Streamlit** - Interface web interativa
- **Supabase** - Banco de dados PostgreSQL + Autenticação
- **ReportLab** - Geração de PDF
- **Matplotlib/Seaborn** - Visualizações

### Bibliotecas Python
```python
pandas==3.0.5
numpy==2.5.3
scikit-learn==1.9.0
xgboost==3.4.1
lightgbm==4.6.0
catboost==1.2.10
streamlit==1.42.0
supabase==3.0.0
shap==0.52.0
reportlab==5.0.1
matplotlib==3.11.1
optuna==5.0.0
imbalanced-learn==0.13.0
```

---

## 📊 Resultados

### Modelo Clínico (MELHOR)

```
Dataset: Pima Indians (768 registros)
Features: 16 (8 clínicas + 8 engenhadas)
Algoritmo: XGBoost

Métricas:
✅ F1-Score:    0.72
✅ Precisão:    63.4%
✅ Recall:      83.3%
✅ ROC-AUC:     0.82
✅ Threshold:   0.35
```

**Top 5 Features (SHAP):**
1. Glucose_BMI (1.12)
2. Age_Glucose (0.60)
3. DiabetesPedigreeFunction (0.39)
4. Glucose (0.35)
5. BMI (0.28)

### Modelos Testados

| Modelo | F1-Score | Status |
|--------|----------|--------|
| XGBoost (Feature Eng) | **0.72** | 🏆 Melhor |
| XGBoost Baseline | 0.69 | ✅ |
| Stacking Ensemble | 0.69 | ✅ |
| CatBoost | 0.69 | ✅ |
| Neural Network (MLP) | 0.68 | ✅ |
| Logistic Regression | 0.44 | ⚠️ |
| Modelo Comportamental | 0.47 | ⚠️ |

### Validação Cruzada (5-fold)

```
F1-Score CV: 0.80 ± 0.02
```

Modelo é **estável** e **bem generalizado**!

---

## 🚀 Instalação

### Pré-requisitos

- Python 3.14+
- pip
- Git

### Passo a Passo

1. **Clone o repositório**
```bash
git clone https://github.com/seu-usuario/tcc-diabetes.git
cd tcc-diabetes
```

2. **Crie ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Instale dependências**
```bash
pip install -r requirements.txt
```

4. **Configure Supabase** (Opcional - para histórico)

Crie arquivo `.env` na raiz:
```env
SUPABASE_URL=sua_url_aqui
SUPABASE_KEY=sua_chave_aqui
```

Siga instruções em `SUPABASE_SETUP.md` para configurar tabelas.

5. **Execute a aplicação**
```bash
streamlit run app_diabetes.py
```

6. **Acesse no navegador**
```
http://localhost:8501
```

---

## 📖 Como Usar

### 1️⃣ Criar Conta
- Clique em "Cadastro"
- Preencha email e senha
- Confirme email (se configurado)

### 2️⃣ Completar Perfil
- Vá em "Meu Perfil"
- Preencha: nome, idade, sexo, altura, peso
- IMC é calculado automaticamente
- Adicione alergias e histórico familiar (opcional)

### 3️⃣ Fazer Predição

**Modelo Clínico** (requer exames):
- Preencha: glicose, pressão, insulina, IMC, etc.
- Clique "Analisar Risco"
- Veja resultado + gráfico SHAP explicando
- Baixe relatório PDF

**Modelo Comportamental** (sem exames):
- Preencha hábitos de vida
- Clique "Analisar Risco"
- Veja recomendações personalizadas
- Baixe relatório PDF

### 4️⃣ Ver Histórico
- Vá em "Histórico & Estatísticas"
- Veja evolução ao longo do tempo
- Compare resultados
- Analise gráficos

---

## 📁 Estrutura do Projeto

```
tcc-diabetes/
│
├── app_diabetes.py              # Aplicação principal Streamlit
├── auth.py                      # Sistema de autenticação
├── supabase_db.py              # Integração banco de dados
├── pdf_generator.py            # Geração de relatórios PDF
├── shap_explicabilidade.py     # Gráficos SHAP
│
├── treinar_modelo.py           # Baseline Logistic Regression
├── treinar_xgboost.py          # XGBoost básico
├── modelo_final_otimizado.py   # XGBoost + Feature Engineering
├── ensemble_definitivo.py      # Voting + Stacking
├── modelo_hibrido_definitivo.py # Comparação Pima vs BRFSS
├── analise_shap_explicabilidade.py # Análise SHAP completa
├── melhorar_precisao.py        # Deep Learning + Calibração
├── otimizacao_bayesiana.py     # Optuna
├── ensemble_stacking_final.py  # Ensemble avançado
│
├── pima_diabetes.csv           # Dataset Pima Indians
├── diabetes_binary_health_indicators_BRFSS2015.csv # Dataset BRFSS
│
├── create_table.sql            # SQL: tabela predições
├── create_users_table.sql      # SQL: tabela usuários
├── .env                        # Credenciais Supabase
├── requirements.txt            # Dependências Python
├── README.md                   # Este arquivo
└── SUPABASE_SETUP.md          # Guia configuração Supabase
```

---

## 📊 Datasets

### 🔬 Pima Indians Diabetes Database
- **Fonte:** UCI Machine Learning Repository
- **Registros:** 768
- **Features:** 8 clínicas
- **Target:** Diabetes (0/1)
- **Prevalência:** 34.9%
- **Uso:** Modelo Clínico

**Features:**
- Pregnancies (Gestações)
- Glucose (Glicose em jejum)
- BloodPressure (Pressão arterial)
- SkinThickness (Espessura da pele)
- Insulin (Insulina sérica)
- BMI (Índice de Massa Corporal)
- DiabetesPedigreeFunction (Histórico familiar)
- Age (Idade)

### 📋 BRFSS 2015 Diabetes Health Indicators
- **Fonte:** CDC Behavioral Risk Factor Surveillance System
- **Registros:** 253,680
- **Features:** 21 comportamentais
- **Target:** Diabetes (0/1)
- **Prevalência:** 13.9%
- **Uso:** Modelo Comportamental

**Features principais:**
- HighBP, HighChol (Condições de saúde)
- BMI, Smoker, PhysActivity (Hábitos)
- Fruits, Veggies (Alimentação)
- GenHlth, Age (Saúde geral)

---

## 🤝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fork o projeto
2. Criar uma branch (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abrir um Pull Request

### Ideias para Contribuir:
- 🌍 Tradução para outros idiomas
- 📱 Versão mobile
- 🔗 Integração com wearables
- 📈 Mais visualizações
- 🧪 Novos modelos de ML
- 📚 Mais datasets

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 📧 Contato

**David Reis**  
📧 Email: daaviidreeis@gmail.com  
💼 LinkedIn: [seu-linkedin]  
🐙 GitHub: [seu-github]

---

## 🙏 Agradecimentos

- **UCI Machine Learning Repository** - Dataset Pima Indians
- **CDC** - Dataset BRFSS 2015
- **Streamlit** - Framework web incrível
- **Supabase** - Backend completo e gratuito
- **SHAP** - Explicabilidade de ML
- **Comunidade Python** - Bibliotecas fantásticas

---

## ⚠️ Aviso Importante

**Este sistema é para fins educacionais e de pesquisa.**

Os resultados **NÃO substituem** uma consulta médica profissional. Sempre consulte um médico endocrinologista para diagnóstico e tratamento adequados.

---

## 📚 Referências

1. Smith, J.W., et al. (1988). "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus"
2. CDC. (2015). "Behavioral Risk Factor Surveillance System"
3. Lundberg, S.M., & Lee, S.I. (2017). "A Unified Approach to Interpreting Model Predictions" (SHAP)
4. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"

---

<div align="center">

**⭐ Se este projeto foi útil, deixe uma estrela! ⭐**

Made with ❤️ by David Reis

</div>
