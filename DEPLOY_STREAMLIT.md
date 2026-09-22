# ☁️ Guia de Deploy - Streamlit Cloud

## 📋 Pré-requisitos

- ✅ Repositório no GitHub criado e com push feito
- ✅ Conta no Supabase configurada
- ✅ Modelos treinados incluídos no repositório

---

## 🚀 Passo 1: Acessar Streamlit Cloud

1. Acesse: https://share.streamlit.io
2. Faça login com sua **conta GitHub**
3. Clique em **"New app"** (botão azul no canto superior direito)

---

## ⚙️ Passo 2: Configurar a Aplicação

### 2.1 Informações do Repositório

Preencha os campos:

| Campo | Valor |
|-------|-------|
| **Repository** | `SEU-USUARIO/tcc-diabetes` |
| **Branch** | `main` |
| **Main file path** | `app_diabetes.py` |
| **App URL** | Escolha um nome único (ex: `davidreis-diabetes` ou `tcc-diabetes-2026`) |

⚠️ **Importante:** O App URL será parte da sua URL final: `https://[nome-escolhido].streamlit.app`

---

## 🔐 Passo 3: Configurar Secrets (IMPORTANTE!)

**ANTES de clicar em "Deploy"**, configure as variáveis de ambiente:

### 3.1 Abrir Advanced Settings

1. Clique em **"Advanced settings"** (abaixo dos campos)
2. Vá para a aba **"Secrets"**

### 3.2 Adicionar Secrets do Supabase

Cole EXATAMENTE o seguinte no campo de secrets (formato TOML):

```toml
SUPABASE_URL = "https://duwhggpxipnnilgqgvev.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR1d2hnZ3B4aXBubmlsZ3FndmV2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODg4ODY0OTcsImV4cCI6MjEwNDQ2MjQ5N30.L-4eSfeQ_SltNk2YP9i-mX8-CH2XWTVNtcBI6zJ80mY"
```

⚠️ **CRÍTICO:** 
- Use aspas duplas `"` (não simples)
- Mantenha o formato TOML exatamente como mostrado
- NÃO adicione espaços extras ou quebras de linha dentro das strings

### 3.3 Salvar

Clique em **"Save"** na seção de secrets

---

## 🎯 Passo 4: Deploy!

1. Clique no botão **"Deploy"** (azul, canto inferior direito)
2. Aguarde ~3-5 minutos (primeira vez é mais demorada)
3. Você verá os logs de instalação das dependências

### O que acontece durante o deploy:

```
⏳ Cloning repository...
⏳ Installing Python 3.14...
⏳ Installing requirements.txt...
   ├─ numpy, pandas, scikit-learn
   ├─ xgboost, lightgbm, catboost
   ├─ streamlit, supabase
   └─ shap, reportlab, matplotlib
⏳ Starting app...
✅ Your app is live! 🎉
```

---

## ✅ Passo 5: Testar a Aplicação

Quando o deploy terminar, seu app estará em:

```
https://[nome-escolhido].streamlit.app
```

### 5.1 Checklist de Testes

Teste TODAS as funcionalidades:

- [ ] **Página carrega sem erros**
- [ ] **Cadastro de usuário funciona**
- [ ] **Login funciona**
- [ ] **Perfil pode ser preenchido e salvo**
- [ ] **Modelo Clínico faz predições**
- [ ] **Modelo Comportamental faz predições**
- [ ] **Gráficos SHAP aparecem**
- [ ] **Gauge de risco visual aparece**
- [ ] **PDF é gerado e baixado**
- [ ] **Histórico de predições aparece**
- [ ] **Análise temporal funciona (após 2+ predições)**

### 5.2 Teste em Múltiplos Dispositivos

- [ ] Desktop (Chrome, Firefox, Edge)
- [ ] Tablet
- [ ] Mobile (Android/iOS)

---

## 🔄 Passo 6: Atualizar README com Link

Após o deploy bem-sucedido, atualize o README.md:

### 6.1 Editar localmente

No arquivo `README.md`, linha 14, substitua:

```markdown
**Acesse o sistema funcionando:** [EM BREVE - Seguir DEPLOY.md]
```

Por:

```markdown
**Acesse o sistema funcionando:** [https://[SEU-APP].streamlit.app](https://[SEU-APP].streamlit.app)
```

### 6.2 Commit e Push

```bash
git add README.md
git commit -m "Update README with deployed app URL"
git push
```

---

## 🎓 Para a Apresentação do TCC

### 📱 QR Code

1. Acesse: https://www.qr-code-generator.com
2. Cole a URL do seu app
3. Baixe o QR Code
4. Adicione nos slides da apresentação

### 📸 Screenshots

Capture screenshots para a documentação:

1. Tela de login/cadastro
2. Dashboard principal
3. Modelo Clínico com resultado
4. Gráfico SHAP explicando predição
5. Gauge de risco visual
6. Histórico com evolução temporal
7. PDF gerado
8. Versão mobile

---

## 🐛 Troubleshooting

### Erro: "ModuleNotFoundError"

**Problema:** Biblioteca faltando no `requirements.txt`

**Solução:**
1. Adicione a biblioteca no `requirements.txt`
2. Commit e push
3. Streamlit detecta automaticamente e redeploy

### Erro: "Supabase not connected" ou "KeyError: 'SUPABASE_URL'"

**Problema:** Secrets não configuradas corretamente

**Solução:**
1. Vá em "Manage app" (⚙️ no canto superior direito do app)
2. Clique em "⋮" > "Settings"
3. Vá para "Secrets"
4. Verifique se as variáveis estão corretas (formato TOML com aspas duplas)
5. Clique em "Save"
6. Clique em "Reboot app"

### Erro: "This app has encountered an error"

**Problema:** Erro no código ou dependência

**Solução:**
1. Vá em "Manage app" > "Logs"
2. Veja o erro específico nos logs
3. Corrija o código localmente
4. Commit e push (redeploy automático)

### Erro 503: "Service Unavailable"

**Problema:** App "dormindo" por inatividade (normal no plano gratuito)

**Solução:** Apenas recarregue a página (F5), o app acorda em ~30 segundos

### App muito lento

**Problema:** Recursos limitados do plano gratuito

**Soluções:**
1. Adicione cache nas funções pesadas:
```python
@st.cache_data
def carregar_modelo():
    return pickle.load(...)
```

2. Otimize carregamento de dados

3. Ou faça upgrade para Streamlit Cloud Pro ($200/mês)

---

## 📊 Monitoramento

### Ver Logs em Tempo Real

1. Acesse seu app
2. Clique em "Manage app" (⚙️)
3. Vá em "Logs"
4. Veja requisições, erros e warnings

### Métricas Disponíveis

- CPU/RAM usage
- Número de viewers ativos
- Uptime
- Erros e exceções

### Receber Notificações

1. Configure email no Streamlit Cloud
2. Receba alertas de:
   - Deploy failed
   - App crashed
   - High resource usage

---

## 🔄 Como Atualizar o App

Sempre que fizer mudanças no código:

```bash
git add .
git commit -m "Descrição das mudanças"
git push
```

O Streamlit Cloud:
- ✅ Detecta automaticamente o push
- ✅ Faz redeploy automaticamente (~2 minutos)
- ✅ Mantém os secrets configurados
- ✅ Mantém a mesma URL

**Você NÃO precisa fazer nada manualmente!** 🎉

---

## 💰 Custos

### Streamlit Cloud (Plano Gratuito)

- ✅ 1 app público
- ✅ Viewers ilimitados
- ✅ 1 GB RAM
- ✅ 1 vCPU
- ✅ HTTPS automático
- ✅ Custom subdomain
- ⚠️ App dorme após inatividade (~10 min)

### Supabase (Plano Gratuito)

- ✅ 500 MB database
- ✅ 50k requisições/mês
- ✅ 2 GB bandwidth
- ✅ Autenticação incluída
- ✅ Row Level Security

### Total: R$ 0,00 por mês! 🎉

---

## 🔐 Segurança

### ✅ Implementado

- HTTPS automático (certificado SSL)
- Secrets não expostas no código
- Row Level Security (RLS) no Supabase
- Senhas criptografadas (bcrypt via Supabase Auth)
- .env no .gitignore

### ⚠️ Para Produção Real (Recomendações)

- Ativar confirmação de email no Supabase
- Implementar rate limiting
- Adicionar backup automático do banco
- Integrar monitoring (Sentry)
- Adicionar logs estruturados
- Implementar testes automatizados

---

## 📞 Suporte

### Documentação Oficial

- **Streamlit Cloud:** https://docs.streamlit.io/streamlit-community-cloud
- **Supabase:** https://supabase.com/docs
- **Streamlit:** https://docs.streamlit.io

### Comunidades

- **Streamlit Forum:** https://discuss.streamlit.io
- **Supabase Discord:** https://discord.supabase.com

---

## 🎉 Parabéns!

Seu TCC está online e acessível para qualquer pessoa no mundo!

**Compartilhe o link:**
- Com a banca do TCC
- No seu LinkedIn
- No seu portfólio
- Com potenciais empregadores

---

**URL do App:** `https://[SEU-APP].streamlit.app`  
**Repositório GitHub:** `https://github.com/SEU-USUARIO/tcc-diabetes`

*Criado em: 22/09/2026*
*Autor: David Reis*
