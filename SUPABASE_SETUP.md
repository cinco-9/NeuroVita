# 🚀 CONFIGURAÇÃO DO SUPABASE - TCC DIABETES

## Passo 1: Criar Conta no Supabase

1. Acesse: **https://supabase.com**
2. Clique em **"Start your project"**
3. Faça login com GitHub ou crie conta com email
4. É **GRÁTIS** (não precisa cartão de crédito!)

---

## Passo 2: Criar Novo Projeto

1. No dashboard, clique em **"New Project"**
2. Preencha:
   - **Name:** `tcc-diabetes` (ou outro nome)
   - **Database Password:** Crie uma senha forte e **ANOTE**
   - **Region:** Escolha `South America (São Paulo)` (mais próximo)
   - **Pricing Plan:** FREE (já vem selecionado)
3. Clique em **"Create new project"**
4. Aguarde ~2 minutos (criação do banco)

---

## Passo 3: Obter as Credenciais

Após o projeto ser criado:

1. No menu lateral, vá em **"Settings"** (ícone de engrenagem)
2. Clique em **"API"**
3. **COPIE e ANOTE** essas 2 informações:

   **URL do Projeto:**
   ```
   https://xxxxxxxxxxxxx.supabase.co
   ```

   **anon/public key:**
   ```
   eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSI...
   ```

   (É uma chave LONGA começando com `eyJ...`)

---

## Passo 4: Criar a Tabela no Banco

1. No menu lateral, clique em **"SQL Editor"**
2. Clique em **"New query"**
3. **COLE** o código SQL abaixo:

```sql
-- Criar tabela de predições
CREATE TABLE predicoes (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Tipo de modelo usado
    tipo_modelo VARCHAR(50) NOT NULL, -- 'clinico' ou 'comportamental'
    
    -- Dados do input (JSON para flexibilidade)
    dados_input JSONB NOT NULL,
    
    -- Resultado da predição
    resultado INTEGER NOT NULL, -- 0 = baixo risco, 1 = alto risco
    probabilidade FLOAT NOT NULL, -- 0.0 a 1.0
    
    -- Metadados
    session_id VARCHAR(100), -- identificar usuário sem login
    user_agent TEXT,
    
    -- Índices para queries rápidas
    CONSTRAINT predicoes_resultado_check CHECK (resultado IN (0, 1)),
    CONSTRAINT predicoes_probabilidade_check CHECK (probabilidade >= 0 AND probabilidade <= 1)
);

-- Criar índices para performance
CREATE INDEX idx_predicoes_created_at ON predicoes(created_at DESC);
CREATE INDEX idx_predicoes_tipo_modelo ON predicoes(tipo_modelo);
CREATE INDEX idx_predicoes_session_id ON predicoes(session_id);

-- Comentários na tabela
COMMENT ON TABLE predicoes IS 'Histórico de predições de diabetes do sistema TCC';
COMMENT ON COLUMN predicoes.tipo_modelo IS 'Tipo de modelo: clinico (Pima) ou comportamental (BRFSS)';
COMMENT ON COLUMN predicoes.dados_input IS 'Dados de entrada do usuário em formato JSON';
COMMENT ON COLUMN predicoes.resultado IS '0 = baixo risco, 1 = alto risco de diabetes';
COMMENT ON COLUMN predicoes.probabilidade IS 'Probabilidade de diabetes (0.0 a 1.0)';
```

4. Clique em **"Run"** (ou F5)
5. Você deve ver: **"Success. No rows returned"**

---

## Passo 5: Verificar se Funcionou

1. No menu lateral, clique em **"Table Editor"**
2. Você deve ver a tabela **"predicoes"** na lista
3. Clique nela - deve aparecer vazia (0 rows)

---

## Passo 6: Configurar o App

1. **Crie um arquivo** chamado `.env` na pasta do projeto:
   ```
   C:\dev\tcc-diabetes\.env
   ```

2. **Cole** suas credenciais nesse formato:
   ```env
   SUPABASE_URL=https://xxxxxxxxxxxxx.supabase.co
   SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSI...
   ```

3. **Substitua** pelos seus valores reais do Passo 3

---

## ✅ Pronto!

Após seguir todos os passos, me avise que eu integro o Supabase no app!

**Dúvidas?** Me chame! 😊

---

## 🔒 Segurança

- **NUNCA** compartilhe suas credenciais
- **NUNCA** commite o arquivo `.env` no Git
- A chave `anon/public` pode ser exposta no frontend (é segura)
- Para produção real, use RLS (Row Level Security) do Supabase
