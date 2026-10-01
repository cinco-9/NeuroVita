# 🗓️ ROADMAP TCC - 3 MESES (Set→Dez 2026)

**Objetivo:** TCC nota 9.5-10.0 + Artigo publicável  
**Status Atual:** App funcionando + Validação + Comparação Literatura ✅

---

## 📊 VISÃO GERAL

### **Já Concluído (70%):**
- ✅ Modelo Clínico (Pima): F1=0.72
- ✅ Modelo Comportamental (BRFSS): F1=0.47
- ✅ App web funcionando (Streamlit Cloud)
- ✅ Validação cruzada 10-fold
- ✅ Comparação com 8 modelos literatura
- ✅ Calibração de probabilidades
- ✅ Análise de erros
- ✅ Interface completa (login, perfil, histórico, PDF)
- ✅ Deploy online

### **Falta Fazer (30%):**
- ⏳ Validação em dados brasileiros (VIGITEL)
- ⏳ Análise comparativa cross-cultural
- ⏳ Deep Learning / AutoML (opcional)
- ⏳ Otimizações finais
- ⏳ Documentação completa
- ⏳ Preparação artigo científico

---

## 📅 CRONOGRAMA DETALHADO

### **🗓️ SEMANA 1-2 (23 Set - 6 Out): Dados Brasileiros**

#### Objetivos:
1. Baixar dados VIGITEL reais
2. Processar e limpar dados
3. Mapear features VIGITEL → modelo
4. Análise exploratória comparativa

#### Tarefas:
- [ ] **Dia 1-2:** Download VIGITEL 2023 + 2024
  - Acessar: https://svs.aids.gov.br/download/Vigitel/
  - Baixar CSV mais recentes (2023, 2024)
  - Verificar integridade dos dados
  
- [ ] **Dia 3-4:** Processamento e limpeza
  - Ler documentação VIGITEL
  - Mapear variáveis para features do modelo
  - Tratar valores faltantes
  - Normalizar formato
  
- [ ] **Dia 5-7:** Análise exploratória
  - Estatísticas descritivas
  - Comparar com BRFSS (USA)
  - Identificar diferenças culturais
  - Documentar prevalências

#### Entregável:
- `vigitel_processado.csv` (dataset limpo)
- `analise_vigitel.py` (script de análise)
- `VIGITEL_ANALISE.md` (relatório comparativo)

#### Tempo estimado: **10-15 horas**

---

### **🗓️ SEMANA 3-4 (7 Out - 20 Out): Treinar Modelo em Dados Brasileiros**

#### Objetivos:
1. Adaptar modelo para VIGITEL
2. Treinar e validar
3. Comparar performance USA vs Brasil
4. Análise de diferenças

#### Tarefas:
- [ ] **Dia 1-3:** Adaptação do modelo
  - Feature engineering para VIGITEL
  - Ajustar pipeline de pré-processamento
  - Balanceamento de classes (SMOTE)
  
- [ ] **Dia 4-6:** Treinamento e validação
  - Treinar XGBoost em VIGITEL
  - 10-fold cross-validation
  - Otimizar hiperparâmetros
  - Calibração de probabilidades
  
- [ ] **Dia 7-10:** Análise comparativa
  - Pima vs BRFSS vs VIGITEL
  - Identificar padrões culturais
  - Fatores de risco Brasil vs USA
  - Generalização do modelo

#### Entregável:
- `modelo_vigitel.pkl` (modelo treinado)
- `comparacao_cross_cultural.py` (análise)
- `ANALISE_CROSS_CULTURAL.md` (relatório)
- Gráficos comparativos (3-4 figuras)

#### Tempo estimado: **15-20 horas**

---

### **🗓️ SEMANA 5-6 (21 Out - 3 Nov): Deep Learning & AutoML**

#### Objetivos:
1. Implementar Neural Network
2. Testar AutoML (TPOT/AutoSklearn)
3. Comparar com modelos tradicionais
4. Ensemble avançado

#### Tarefas:
- [ ] **Dia 1-3:** Deep Learning
  - Implementar MLP (Multi-Layer Perceptron)
  - Testar arquiteturas diferentes
  - Regularização (Dropout, L2)
  - Early stopping
  
- [ ] **Dia 4-6:** AutoML
  - Configurar TPOT
  - Busca automática de pipelines
  - Comparar com modelos manuais
  - Identificar melhores combinações
  
- [ ] **Dia 7-10:** Ensemble Avançado
  - Stacking: RF + GB + XGB
  - Voting Classifier
  - Meta-learner otimizado
  - Validação final

#### Entregável:
- `deep_learning_diabetes.py` (implementação)
- `automl_experiment.py` (TPOT)
- `ensemble_avancado.pkl` (melhor modelo)
- `MODELOS_AVANCADOS.md` (relatório)

#### Tempo estimado: **15-20 horas**

---

### **🗓️ SEMANA 7-8 (4 Nov - 17 Nov): Otimizações & Refinamentos**

#### Objetivos:
1. Otimizar interface do app
2. Melhorar visualizações
3. Adicionar features extras
4. Testes de usabilidade

#### Tarefas:
- [ ] **Interface:**
  - Botão "Preencher Exemplo"
  - Tooltips explicativos
  - Validação de campos
  - Tutorial de boas-vindas
  
- [ ] **Visualizações:**
  - Melhorar gráficos SHAP
  - Dashboard mais interativo
  - Comparação temporal
  - Gauge de risco visual
  
- [ ] **Features Extras:**
  - Exportar histórico CSV
  - Comparar com população geral
  - Recomendações personalizadas
  - Integração com VIGITEL (stats)
  
- [ ] **Testes:**
  - Testar em 5+ usuários
  - Coletar feedback
  - Corrigir bugs
  - Otimizar performance

#### Entregável:
- App otimizado (versão 2.0)
- Relatório de testes de usabilidade
- Documentação de features

#### Tempo estimado: **10-15 horas**

---

### **🗓️ SEMANA 9-10 (18 Nov - 1 Dez): Documentação & Artigo**

#### Objetivos:
1. Documentação completa do projeto
2. Preparar artigo científico
3. Slides da apresentação
4. Vídeo de demonstração

#### Tarefas:
- [ ] **Documentação Técnica:**
  - README.md completo
  - Guia de instalação
  - Manual do usuário
  - Documentação de código
  
- [ ] **Artigo Científico:**
  - Estrutura: Intro + Métodos + Resultados + Discussão
  - Foco: Validação cross-cultural (Brasil vs USA)
  - Tabelas e figuras profissionais
  - Referências bibliográficas
  - Target: Revista brasileira de saúde pública
  
- [ ] **Apresentação TCC:**
  - Slides PowerPoint (20-30 slides)
  - Roteiro de apresentação
  - Ensaiar (15-20 minutos)
  - Preparar respostas para perguntas comuns
  
- [ ] **Vídeo Demo:**
  - Gravar demonstração do app (5 min)
  - Explicar funcionalidades
  - Mostrar resultados
  - Upload no YouTube

#### Entregável:
- Documentação completa
- Artigo científico (draft)
- Slides apresentação
- Vídeo demonstração

#### Tempo estimado: **20-25 horas**

---

### **🗓️ SEMANA 11-12 (2 Dez - 15 Dez): Revisão Final & Submissão**

#### Objetivos:
1. Revisão completa do TCC
2. Correções finais
3. Preparação para defesa
4. Submissão

#### Tarefas:
- [ ] **Revisão TCC:**
  - Ler documento completo
  - Verificar formatação ABNT
  - Revisar ortografia/gramática
  - Ajustar referências
  
- [ ] **Correções:**
  - Implementar feedback do orientador
  - Corrigir inconsistências
  - Melhorar redação
  - Atualizar gráficos
  
- [ ] **Defesa:**
  - Ensaiar apresentação (5x)
  - Preparar FAQ
  - Revisar conceitos técnicos
  - Preparar demo ao vivo
  
- [ ] **Submissão:**
  - Imprimir cópias
  - Upload versão digital
  - Enviar para banca
  - Agendar defesa

#### Entregável:
- TCC final (PDF)
- Apresentação ensaiada
- Demo funcionando
- Pronto para defender!

#### Tempo estimado: **15-20 horas**

---

## ⏱️ RESUMO TEMPORAL

| Fase | Período | Horas | Prioridade |
|------|---------|-------|------------|
| **Dados Brasileiros** | Sem 1-2 | 10-15h | 🔴 Alta |
| **Treinar VIGITEL** | Sem 3-4 | 15-20h | 🔴 Alta |
| **Deep Learning** | Sem 5-6 | 15-20h | 🟡 Média |
| **Otimizações** | Sem 7-8 | 10-15h | 🟡 Média |
| **Documentação** | Sem 9-10 | 20-25h | 🔴 Alta |
| **Revisão Final** | Sem 11-12 | 15-20h | 🔴 Alta |
| **TOTAL** | **3 meses** | **85-115h** | - |

**Média:** ~7-10 horas/semana (1-2h por dia)

---

## 🎯 METAS POR MÊS

### **📅 OUTUBRO (Mês 1):**
- ✅ VIGITEL baixado e processado
- ✅ Modelo treinado em dados brasileiros
- ✅ Análise comparativa USA vs Brasil
- ✅ Deep Learning implementado

### **📅 NOVEMBRO (Mês 2):**
- ✅ AutoML testado
- ✅ App otimizado (v2.0)
- ✅ Artigo científico (draft)
- ✅ Slides prontos

### **📅 DEZEMBRO (Mês 3):**
- ✅ TCC revisado
- ✅ Apresentação ensaiada
- ✅ Defesa marcada
- ✅ APROVADO! 🎓

---

## 📊 RESULTADOS ESPERADOS

### **Métricas do Modelo:**
- F1-Score Pima: 0.72 ✅ (atual)
- F1-Score VIGITEL: **0.45-0.55** (esperado)
- Generalização cross-cultural: **Documentada**
- Ensemble: **0.75-0.78** (esperado)

### **Publicações:**
- TCC completo: **100-150 páginas**
- Artigo científico: **8-12 páginas**
- Apresentação: **25-30 slides**
- Vídeo demo: **5 minutos**

### **Qualidade:**
- Nota esperada: **9.5-10.0** 🏆
- Publicação em revista: **Possível** 📄
- Destaque no curso: **Provável** ⭐
- Referência para futuros TCCs: **Sim** 📚

---

## 💡 DICAS PARA OS 3 MESES

### **Organização:**
- ✅ Reserve 1-2h por dia para o TCC
- ✅ Sábados: sessão longa (4-6h)
- ✅ Domingos: revisão/planejamento
- ✅ Use Notion/Trello para tracking

### **Qualidade:**
- ✅ Documente TUDO enquanto faz
- ✅ Faça git commit frequente
- ✅ Peça feedback do orientador mensalmente
- ✅ Mostre para colegas/amigos

### **Evite:**
- ❌ Deixar para última semana
- ❌ Não fazer backups
- ❌ Trabalhar sem planejar
- ❌ Ignorar feedback

---

## 🚀 PRÓXIMO PASSO IMEDIATO

**Esta semana (23-29 Set):**

1. [ ] Ler documentação VIGITEL
2. [ ] Baixar dados 2023 + 2024
3. [ ] Explorar primeiras colunas
4. [ ] Começar mapeamento de variáveis

**Vou te ajudar com isso AGORA!**

---

## 📞 SUPORTE CONTÍNUO

**Me chame quando:**
- ❓ Tiver dúvidas técnicas
- 🐛 Encontrar bugs
- 💡 Precisar de ideias
- ✅ Completar cada fase (para revisão)

**Vamos fazer o MELHOR TCC do curso!** 🎓🚀

---

*Roadmap criado em: 23 Set 2026*
*Previsão de conclusão: 15 Dez 2026*
