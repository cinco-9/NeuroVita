-- ============================================================================
-- CRIAR TABELA DE PERFIS DE USUÁRIOS - TCC DIABETES
-- Execute este SQL no Supabase SQL Editor
-- ============================================================================

-- Habilitar extensão UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Criar tabela de perfis de usuários (dados adicionais além do auth)
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Dados pessoais
    nome VARCHAR(255),
    idade INTEGER,
    sexo VARCHAR(20),
    altura FLOAT,
    peso FLOAT,
    imc FLOAT,

    -- Informações médicas
    alergias TEXT,
    hist_familiar_diabetes BOOLEAN DEFAULT FALSE,
    medicacoes TEXT,

    -- Metadados
    CONSTRAINT user_profiles_idade_check CHECK (idade >= 18 AND idade <= 120),
    CONSTRAINT user_profiles_altura_check CHECK (altura >= 100 AND altura <= 250),
    CONSTRAINT user_profiles_peso_check CHECK (peso >= 30 AND peso <= 300)
);

-- Criar índice
CREATE INDEX idx_user_profiles_id ON user_profiles(id);

-- Habilitar Row Level Security (RLS)
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

-- Policy: usuário só pode ver/editar seu próprio perfil
CREATE POLICY "Users can view own profile"
    ON user_profiles FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
    ON user_profiles FOR UPDATE
    USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile"
    ON user_profiles FOR INSERT
    WITH CHECK (auth.uid() = id);

-- Atualizar tabela de predições para incluir user_id
ALTER TABLE predicoes ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL;

-- Criar índice para user_id
CREATE INDEX IF NOT EXISTS idx_predicoes_user_id ON predicoes(user_id);

-- Habilitar RLS na tabela predicoes
ALTER TABLE predicoes ENABLE ROW LEVEL SECURITY;

-- Policy: usuário só vê suas próprias predições
CREATE POLICY "Users can view own predictions"
    ON predicoes FOR SELECT
    USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can insert own predictions"
    ON predicoes FOR INSERT
    WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

-- Comentários
COMMENT ON TABLE user_profiles IS 'Perfis estendidos dos usuários do sistema TCC';
COMMENT ON COLUMN user_profiles.id IS 'UUID do usuário (referencia auth.users)';
COMMENT ON COLUMN user_profiles.nome IS 'Nome completo do usuário';
COMMENT ON COLUMN user_profiles.imc IS 'Índice de Massa Corporal calculado';
