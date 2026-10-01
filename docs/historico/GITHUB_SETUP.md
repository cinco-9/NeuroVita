# 🚀 Guia Rápido - Upload para GitHub

## ✅ Status Atual
- ✅ Git inicializado
- ✅ Primeiro commit realizado (45 arquivos)
- ✅ Modelos treinados incluídos (177KB total)
- ✅ Arquivos temporários removidos

---

## 📝 Passo 1: Criar Repositório no GitHub

### Opção A: Via Interface Web (Recomendado)

1. Acesse: https://github.com/new
2. Preencha:
   - **Repository name:** `tcc-diabetes`
   - **Description:** `Sistema de IA para Predição de Diabetes Tipo 2 - TCC 2026`
   - **Visibility:** ✅ Public (para Streamlit Cloud gratuito)
   - ❌ **NÃO** marque "Add a README file"
   - ❌ **NÃO** marque "Add .gitignore"
3. Clique em **"Create repository"**

### Opção B: Via GitHub CLI (Se instalado)

```bash
gh repo create tcc-diabetes --public --description "Sistema de IA para Predição de Diabetes Tipo 2 - TCC 2026"
```

---

## 🔗 Passo 2: Conectar e Fazer Push

Após criar o repositório no GitHub, execute os comandos abaixo **na ordem**:

### 2.1 Conectar ao Repositório Remoto

```bash
git remote add origin https://github.com/SEU-USUARIO/tcc-diabetes.git
```

⚠️ **Substitua `SEU-USUARIO`** pelo seu nome de usuário do GitHub!

### 2.2 Renomear Branch para Main

```bash
git branch -M main
```

### 2.3 Fazer Push (Upload)

```bash
git push -u origin main
```

🔐 **Autenticação GitHub:**
- Se pedir senha, use um **Personal Access Token** (não a senha da conta)
- Crie em: https://github.com/settings/tokens
- Permissões necessárias: `repo` (acesso total)

---

## ✅ Passo 3: Verificar Upload

Acesse: `https://github.com/SEU-USUARIO/tcc-diabetes`

Você deve ver:
- ✅ 45 arquivos
- ✅ README.md renderizado
- ✅ Commit "Initial commit - Sistema de Predição de Diabetes Tipo 2"
- ✅ Badge verde "Python 3.14" no README

---

## 🎯 Próximos Passos (Após Push)

### 1. Atualizar README com Link do GitHub

No arquivo `README.md`, linha 238, substitua:

```markdown
git clone https://github.com/seu-usuario/tcc-diabetes.git
```

Por:

```markdown
git clone https://github.com/SEU-USUARIO/tcc-diabetes.git
```

### 2. Deploy no Streamlit Cloud

Siga o arquivo `DEPLOY_STREAMLIT.md` que será criado em seguida.

---

## 🐛 Troubleshooting

### Erro: "remote origin already exists"

```bash
git remote remove origin
git remote add origin https://github.com/SEU-USUARIO/tcc-diabetes.git
```

### Erro: "authentication failed"

1. Gere um Personal Access Token: https://github.com/settings/tokens
2. Clique em "Generate new token (classic)"
3. Marque: `repo` (Full control of private repositories)
4. Copie o token
5. Use o token como senha no `git push`

### Erro: "repository not found"

- Verifique se o nome do repositório está correto
- Confirme que o repositório foi criado no GitHub
- Verifique se o nome de usuário está correto na URL

---

## 📊 Informações do Commit

**Commit ID:** 96c844d  
**Arquivos:** 45 files  
**Linhas:** +262,821 insertions  
**Modelos:** modelo_melhorado_clinico.pkl (177KB), scaler_melhorado.pkl (1KB)

---

## 🔄 Comandos Úteis

### Ver status do repositório
```bash
git status
```

### Ver histórico de commits
```bash
git log --oneline
```

### Ver repositórios remotos
```bash
git remote -v
```

### Fazer novos commits (após mudanças)
```bash
git add .
git commit -m "Descrição das mudanças"
git push
```

---

**🎉 Depois do push, você estará pronto para o deploy no Streamlit Cloud!**

*Criado em: 22/09/2026*
