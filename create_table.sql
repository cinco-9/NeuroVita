-- ============================================================================
-- CRIAR TABELA DE PREDIÇÕES - TCC DIABETES
-- Execute este SQL no Supabase SQL Editor
-- ============================================================================

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

    -- Constraints
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
