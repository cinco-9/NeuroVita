# 🚀 Guia de Deploy - Streamlit Cloud

## 📋 Pré-requisitos

- Conta no [GitHub](https://github.com)
- Conta no [Streamlit Cloud](https://share.streamlit.io) (gratuita!)
- Conta no [Supabase](https://supabase.com) (já criada!)

---

## 🔧 Passo 1: Preparar o Repositório Git

### 1.1 Inicializar Git (se ainda não fez)

```bash
cd c:\dev\tcc-diabetes
git init
git add .
git commit -m "Initial commit - Sistema de Predição de Diabetes"
```

### 1.2 Criar Repositório no GitHub

1. Acesse [GitHub](https://github.com/new)
2. Crie um repositório **público** chamado `tcc-diabetes`
3. **NÃO** inicialize com README (já temos!)

### 1.3 Conectar e Enviar

```bash
git remote add origin https://github.com/SEU-USUARIO/tcc-diabetes.git
git branch -M main
git push -u origin main
```

⚠️ **Importante:** Substitua `SEU-USUARIO` pelo seu nome de usuário do GitHub!

---

## 🌐 Passo 2: Deploy no Streamlit Cloud

### 2.1 Acessar Streamlit Cloud

1. Acesse [share.streamlit.io](https://share.streamlit.io)
2. Faça login com sua conta GitHub
3. Clique em **"New app"**

### 2.2 Configurar o App

Preencha os campos:

- **Repository:** `SEU-USUARIO/tcc-diabetes`
- **Branch:** `main`
- **Main file path:** `app_diabetes.py`
- **App URL:** Escolha um nome único (ex: `seu-nome-diabetes`)

### 2.3 Configurar Secrets (Supabase)

Antes de clicar em "Deploy", configure as variáveis de ambiente:

1. Clique em **"Advanced settings"** > **"Secrets"**
2. Cole o seguinte (com seus valores reais):

```toml
SUPABASE_URL = "https://duwhggpxipnnilgqgvev.supabase.co"
SUPABASE_KEY = "sua_chave_aqui"
```

⚠️ **IMPORTANTE:** Pegue suas credenciais do arquivo `.env` local!

### 2.4 Deploy!

Clique em **"Deploy"** e aguarde (~5 minutos)

---

## ✅ Passo 3: Verificar Deploy

Quando terminar, seu app estará disponível em:

```
https://seu-nome-diabetes.streamlit.app
```

### 3.1 Teste Completo

- ✅ Login/Cadastro funciona?
- ✅ Perfil salva?
- ✅ Predições funcionam?
- ✅ Gráficos aparecem?
- ✅ PDF é gerado?

---

## 🔄 Passo 4: Atualizar o App

Sempre que fizer mudanças no código:

```bash
git add .
git commit -m "Descrição das mudanças"
git push
```

O Streamlit Cloud **detecta automaticamente** e redeploy em ~2 minutos!

---

## 🐛 Troubleshooting

### Erro: "ModuleNotFoundError"

**Problema:** Falta alguma biblioteca no `requirements.txt`

**Solução:**
1. Adicione a biblioteca no `requirements.txt`
2. Commit e push

### Erro: "Supabase not connected"

**Problema:** Secrets não configuradas

**Solução:**
1. Vá em Settings > Secrets no Streamlit Cloud
2. Adicione SUPABASE_URL e SUPABASE_KEY
3. Clique em "Save"
4. Reboot app

### App muito lento

**Problema:** Streamlit Cloud Free tem recursos limitados

**Solução:**
- Adicione `@st.cache_data` em funções pesadas
- Ou faça upgrade para Streamlit Cloud Pro

### Erro 503 - Service Unavailable

**Problema:** App dormindo (inatividade)

**Solução:** Apenas recarregue a página, ele acorda automaticamente

---

## 📊 Monitoramento

### Ver Logs

1. Acesse seu app no Streamlit Cloud
2. Clique em "Manage app" (⚙️)
3. Vá em "Logs"

### Métricas

- CPU/RAM usage
- Número de viewers
- Uptime

---

## 🎓 Para a Apresentação do TCC

### Link Permanente

Adicione no README.md:

```markdown
## 🌐 Demo Online

Acesse o sistema funcionando: https://seu-nome-diabetes.streamlit.app
```

### QR Code

Gere um QR Code do link em [qr-code-generator.com](https://www.qr-code-generator.com)

Coloque nos slides para a banca testar!

---

## 💰 Custos

**Streamlit Cloud:**
- ✅ Gratuito até 1 app
- ✅ Ilimitado viewers
- ✅ CPU/RAM suficientes

**Supabase:**
- ✅ Gratuito até 500MB
- ✅ 50k requisições/mês
- ✅ Autenticação incluída

**Total: R$ 0,00** 🎉

---

## 🔐 Segurança

### ✅ O que está seguro:

- SUPABASE_KEY não exposta no código
- Row Level Security (RLS) ativo
- Senha criptografada (Supabase Auth)
- HTTPS automático

### ⚠️ Para produção real:

- Ativar confirmação de email
- Rate limiting
- Backup automático do banco
- Monitoring (Sentry)

---

## 🎯 Checklist Final

Antes de apresentar:

- [ ] App no ar e funcionando
- [ ] README.md atualizado com link
- [ ] QR Code criado para slides
- [ ] Testado em mobile
- [ ] Testado com usuário teste
- [ ] Link funciona de outra rede
- [ ] Banco de dados com dados de exemplo

---

## 📞 Suporte

- **Streamlit Cloud:** [docs.streamlit.io](https://docs.streamlit.io)
- **Supabase:** [supabase.com/docs](https://supabase.com/docs)
- **GitHub Issues:** [Criar issue](https://github.com/SEU-USUARIO/tcc-diabetes/issues)

---

**🎉 Parabéns! Seu TCC está na nuvem!** ☁️

---

## 📸 Screenshots para Documentação

Capture screenshots para o TCC:

1. Tela de login
2. Dashboard principal
3. Modelo Clínico com gauge
4. Gráfico SHAP
5. PDF gerado
6. Histórico com evolução temporal

Use [lightshot.com](https://lightshot.com) ou print screen!

---

*Última atualização: 14/09/2026*
