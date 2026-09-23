# 📥 GUIA COMPLETO: Download Dados VIGITEL

**Atualizado:** 23 Set 2026  
**Tempo estimado:** 15-30 minutos

---

## 🎯 OBJETIVO

Baixar dados VIGITEL (Sistema de Vigilância de Fatores de Risco do Ministério da Saúde) para validar seu modelo em população brasileira.

---

## 📋 OPÇÃO 1: Download via Portal Oficial (RECOMENDADO)

### **Passo 1: Acessar Portal VIGITEL**

🔗 **Link:** https://www.gov.br/saude/pt-br/composicao/svsa/inqueritos-de-saude/vigitel

### **Passo 2: Localizar Dados**

Procure por uma das opções:
- "Base de Dados"
- "Microdados"
- "Download"
- "Dados Abertos"

### **Passo 3: Escolher Ano**

Baixe preferencialmente:
- ✅ **VIGITEL 2023** (mais recente completo)
- ✅ **VIGITEL 2022** (backup)

### **Passo 4: Formato do Arquivo**

Priorize nesta ordem:
1. **CSV** (mais fácil)
2. **DBF** (converte facilmente)
3. **XLSX** (também OK)
4. **SAV/SPSS** (requer conversão)

### **Passo 5: Arquivos Complementares**

Também baixe (se disponível):
- 📖 **Dicionário de Variáveis** (essencial!)
- 📄 **Nota Técnica** (pesos amostrais)
- 📊 **Relatório Descritivo**

### **Passo 6: Salvar Arquivos**

Coloque TUDO em:
```
F:\dev\tcc-diabetes\dados_vigitel\
```

---

## 📋 OPÇÃO 2: Portal SVS Antigo

Se o link acima não funcionar:

🔗 **Link:** https://svs.aids.gov.br/daent/cgdnt/vigitel/

**Navegue:**
1. Procure menu "Download" ou "Dados"
2. Entre na pasta do ano desejado
3. Baixe todos os arquivos disponíveis

---

## 📋 OPÇÃO 3: Contato Direto (Se nada funcionar)

**Email:** vigitel@saude.gov.br (provável)

**Mensagem modelo:**
```
Assunto: Solicitação de acesso aos microdados VIGITEL para TCC

Prezados,

Sou estudante de [SUA UNIVERSIDADE] e estou desenvolvendo meu Trabalho de 
Conclusão de Curso sobre predição de diabetes utilizando Machine Learning.

Gostaria de solicitar acesso aos microdados do VIGITEL (anos 2022 e/ou 2023) 
para validação do modelo em população brasileira.

Poderiam me orientar sobre como acessar esses dados ou me enviar os arquivos 
em formato CSV/DBF?

Atenciosamente,
David Reis
Email: daaviidreeis@gmail.com
```

---

## ✅ CHECKLIST PÓS-DOWNLOAD

Após baixar, verifique se tem:

- [ ] Arquivo de dados (.csv, .dbf, .xlsx ou .sav)
- [ ] Dicionário de variáveis (nomes das colunas)
- [ ] Nota técnica (pesos amostrais)
- [ ] Tamanho do arquivo > 10 MB (indica dados completos)

---

## 🔍 VERIFICAR SE BAIXOU CERTO

Execute este script:

```bash
python baixar_dados_vigitel.py
```

Ele irá:
- ✅ Procurar arquivos na pasta dados_vigitel/
- ✅ Mostrar informações básicas
- ✅ Validar formato

---

## 📊 O QUE ESPERAR

### **Arquivo de Dados:**
- **Linhas:** ~27.000-54.000 (depende do ano)
- **Colunas:** ~100-150 variáveis
- **Tamanho:** 15-50 MB

### **Variáveis Importantes:**
- `diabetes` ou `Q060` - Diagnóstico de diabetes
- `peso` ou `imc` - IMC
- `sexo` - Sexo
- `idade` - Idade
- `escolaridade` - Escolaridade
- `ativ_fisica` - Atividade física
- `frutas`, `verduras` - Alimentação

---

## 🚨 PROBLEMAS COMUNS

### **Problema 1: Link quebrado**
**Solução:** Tente opção 2 ou 3

### **Problema 2: Precisa cadastro**
**Solução:** Cadastre-se no portal (é gratuito)

### **Problema 3: Arquivo muito grande**
**Solução:** Normal! VIGITEL tem ~50k registros

### **Problema 4: Formato estranho (DBF, SAV)**
**Solução:** Temos script de conversão (próximo passo)

---

## 📞 SUPORTE

Se travar em qualquer passo, me avise!

Vou te ajudar a:
- Converter formatos
- Interpretar dicionário
- Processar os dados

---

## ➡️ PRÓXIMO PASSO

Depois de baixar:

```bash
python processar_vigitel.py
```

Esse script irá:
1. Ler o arquivo baixado
2. Identificar variáveis de diabetes
3. Mapear para o formato do modelo
4. Gerar análise exploratória

---

**IMPORTANTE:** Não se preocupe se demorar 1-2 dias para conseguir os dados. 
É normal! Dados governamentais às vezes são difíceis de acessar.

O importante é que você TEM 3 MESES, então sem pressa! 😊

---

*Guia criado: 23 Set 2026*
