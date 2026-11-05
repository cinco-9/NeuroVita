-- ============================================================================
-- DIÁRIO DE SAÚDE, ALIMENTAÇÃO E ACESSO DO MÉDICO - TCC DIABETES
-- Execute este SQL uma vez no Supabase SQL Editor, depois de privacidade_lgpd.sql.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Tipo de conta (paciente ou médico)
-- ----------------------------------------------------------------------------
ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS tipo_conta TEXT NOT NULL DEFAULT 'paciente';
ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS crm TEXT;
ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS crm_uf TEXT;
ALTER TABLE user_profiles DROP CONSTRAINT IF EXISTS user_profiles_tipo_conta_check;
ALTER TABLE user_profiles ADD CONSTRAINT user_profiles_tipo_conta_check
    CHECK (tipo_conta IN ('paciente', 'medico'));

-- Cria o perfil no cadastro, com nome e tipo de conta informados no formulário.
-- Necessário porque, com confirmação de email ligada, o app não tem sessão no
-- momento do cadastro e não consegue gravar o perfil ele mesmo.
CREATE OR REPLACE FUNCTION public.criar_perfil_novo_usuario()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    tipo TEXT := COALESCE(NEW.raw_user_meta_data->>'tipo_conta', 'paciente');
BEGIN
    IF tipo NOT IN ('paciente', 'medico') THEN
        tipo := 'paciente';
    END IF;
    INSERT INTO public.user_profiles (id, nome, tipo_conta, crm, crm_uf)
    VALUES (
        NEW.id,
        NULLIF(NEW.raw_user_meta_data->>'nome', ''),
        tipo,
        NULLIF(NEW.raw_user_meta_data->>'crm', ''),
        NULLIF(NEW.raw_user_meta_data->>'crm_uf', '')
    )
    ON CONFLICT (id) DO NOTHING;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.criar_perfil_novo_usuario();

-- ----------------------------------------------------------------------------
-- 2. Compartilhamento paciente -> médico (o paciente decide e pode revogar)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS compartilhamentos (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    paciente_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    medico_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    UNIQUE (paciente_id, medico_id),
    CHECK (paciente_id <> medico_id)
);
CREATE INDEX IF NOT EXISTS idx_compartilhamentos_medico ON compartilhamentos(medico_id);

ALTER TABLE compartilhamentos ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Participants can view sharing" ON compartilhamentos;
CREATE POLICY "Participants can view sharing"
    ON compartilhamentos FOR SELECT
    USING (auth.uid() = paciente_id OR auth.uid() = medico_id);

-- Paciente revoga; médico também pode encerrar o acompanhamento
DROP POLICY IF EXISTS "Participants can delete sharing" ON compartilhamentos;
CREATE POLICY "Participants can delete sharing"
    ON compartilhamentos FOR DELETE
    USING (auth.uid() = paciente_id OR auth.uid() = medico_id);

-- Sem policy de INSERT: o compartilhamento só é criado pela função abaixo,
-- que confere se o email é de uma conta de médico.
CREATE OR REPLACE FUNCTION public.compartilhar_com_medico(email_medico TEXT)
RETURNS TEXT
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    id_medico UUID;
    nome_medico TEXT;
BEGIN
    IF auth.uid() IS NULL THEN
        RAISE EXCEPTION 'não autenticado';
    END IF;

    SELECT u.id, p.nome INTO id_medico, nome_medico
    FROM auth.users u
    JOIN public.user_profiles p ON p.id = u.id
    WHERE lower(u.email) = lower(trim(email_medico))
      AND p.tipo_conta = 'medico';

    IF id_medico IS NULL THEN
        RAISE EXCEPTION 'medico_nao_encontrado';
    END IF;
    IF id_medico = auth.uid() THEN
        RAISE EXCEPTION 'medico_nao_encontrado';
    END IF;

    INSERT INTO public.compartilhamentos (paciente_id, medico_id)
    VALUES (auth.uid(), id_medico)
    ON CONFLICT (paciente_id, medico_id) DO NOTHING;

    RETURN COALESCE(nome_medico, 'médico');
END;
$$;

-- Médicos com quem o paciente logado compartilha
CREATE OR REPLACE FUNCTION public.meus_medicos()
RETURNS TABLE (medico_id UUID, nome TEXT, crm TEXT, crm_uf TEXT, email TEXT, desde TIMESTAMPTZ)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT c.medico_id, p.nome, p.crm, p.crm_uf, u.email::TEXT, c.created_at
    FROM public.compartilhamentos c
    JOIN public.user_profiles p ON p.id = c.medico_id
    JOIN auth.users u ON u.id = c.medico_id
    WHERE c.paciente_id = auth.uid()
    ORDER BY c.created_at;
$$;

-- Pacientes que compartilham com o médico logado
CREATE OR REPLACE FUNCTION public.meus_pacientes()
RETURNS TABLE (paciente_id UUID, nome TEXT, email TEXT, desde TIMESTAMPTZ)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT c.paciente_id, p.nome, u.email::TEXT, c.created_at
    FROM public.compartilhamentos c
    LEFT JOIN public.user_profiles p ON p.id = c.paciente_id
    JOIN auth.users u ON u.id = c.paciente_id
    WHERE c.medico_id = auth.uid()
    ORDER BY p.nome;
$$;

REVOKE ALL ON FUNCTION public.compartilhar_com_medico(TEXT) FROM PUBLIC, anon;
REVOKE ALL ON FUNCTION public.meus_medicos() FROM PUBLIC, anon;
REVOKE ALL ON FUNCTION public.meus_pacientes() FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.compartilhar_com_medico(TEXT) TO authenticated;
GRANT EXECUTE ON FUNCTION public.meus_medicos() TO authenticated;
GRANT EXECUTE ON FUNCTION public.meus_pacientes() TO authenticated;

-- Função auxiliar para as policies: o usuário logado é médico deste paciente?
CREATE OR REPLACE FUNCTION public.e_medico_do_paciente(paciente UUID)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT EXISTS (
        SELECT 1 FROM public.compartilhamentos
        WHERE paciente_id = paciente AND medico_id = auth.uid()
    );
$$;
REVOKE ALL ON FUNCTION public.e_medico_do_paciente(UUID) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.e_medico_do_paciente(UUID) TO authenticated;

-- Médico lê (só lê) perfil e avaliações dos pacientes que compartilharam com ele
DROP POLICY IF EXISTS "Doctors can view shared profiles" ON user_profiles;
CREATE POLICY "Doctors can view shared profiles"
    ON user_profiles FOR SELECT
    USING (public.e_medico_do_paciente(id));

DROP POLICY IF EXISTS "Doctors can view shared predictions" ON predicoes;
CREATE POLICY "Doctors can view shared predictions"
    ON predicoes FOR SELECT
    USING (public.e_medico_do_paciente(user_id));

-- ----------------------------------------------------------------------------
-- 3. Diário de saúde
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS diario_registros (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    user_id UUID NOT NULL DEFAULT auth.uid() REFERENCES auth.users(id) ON DELETE CASCADE,
    data DATE NOT NULL DEFAULT CURRENT_DATE,
    tipo TEXT NOT NULL CHECK (tipo IN (
        'refeicao', 'atividade', 'glicemia', 'peso', 'pressao', 'sono', 'sintomas', 'habitos'
    )),
    dados JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_diario_user_data ON diario_registros(user_id, data DESC);

ALTER TABLE diario_registros ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own diary" ON diario_registros;
CREATE POLICY "Users can view own diary"
    ON diario_registros FOR SELECT
    USING (auth.uid() = user_id OR public.e_medico_do_paciente(user_id));

DROP POLICY IF EXISTS "Users can insert own diary" ON diario_registros;
CREATE POLICY "Users can insert own diary"
    ON diario_registros FOR INSERT
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete own diary" ON diario_registros;
CREATE POLICY "Users can delete own diary"
    ON diario_registros FOR DELETE
    USING (auth.uid() = user_id);

-- ----------------------------------------------------------------------------
-- 4. Produtos cadastrados pelo próprio usuário (código de barras não encontrado
--    no Open Food Facts). Valores por 100 g ou 100 ml.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS produtos_usuario (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    user_id UUID NOT NULL DEFAULT auth.uid() REFERENCES auth.users(id) ON DELETE CASCADE,
    codigo_barras TEXT NOT NULL,
    nome TEXT NOT NULL,
    kcal FLOAT NOT NULL CHECK (kcal >= 0),
    carboidratos_g FLOAT NOT NULL CHECK (carboidratos_g >= 0),
    acucares_g FLOAT CHECK (acucares_g >= 0),
    proteinas_g FLOAT NOT NULL CHECK (proteinas_g >= 0),
    gorduras_g FLOAT NOT NULL CHECK (gorduras_g >= 0),
    fibra_g FLOAT CHECK (fibra_g >= 0),
    sodio_mg FLOAT CHECK (sodio_mg >= 0),
    UNIQUE (user_id, codigo_barras)
);

ALTER TABLE produtos_usuario ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users manage own products" ON produtos_usuario;
CREATE POLICY "Users manage own products"
    ON produtos_usuario FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);
