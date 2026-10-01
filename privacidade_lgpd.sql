-- ============================================================================
-- PRIVACIDADE / LGPD - TCC DIABETES
-- Execute este SQL uma vez no Supabase SQL Editor (projeto já existente).
-- Instalações novas: rode create_table.sql, create_users_table.sql e depois este.
-- ============================================================================

-- 1. Cada usuário só lê e grava as próprias avaliações.
--    A regra antiga ("OR user_id IS NULL") deixava qualquer pessoa ler
--    e inserir avaliações sem dono usando a chave pública (anon).
DROP POLICY IF EXISTS "Users can view own predictions" ON predicoes;
DROP POLICY IF EXISTS "Users can insert own predictions" ON predicoes;
DROP POLICY IF EXISTS "Users can delete own predictions" ON predicoes;

CREATE POLICY "Users can view own predictions"
    ON predicoes FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own predictions"
    ON predicoes FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own predictions"
    ON predicoes FOR DELETE
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete own profile" ON user_profiles;
CREATE POLICY "Users can delete own profile"
    ON user_profiles FOR DELETE
    USING (auth.uid() = id);

-- 2. Ao excluir a conta, as avaliações são apagadas junto
--    (antes ficavam no banco com user_id NULL, visíveis para todos).
ALTER TABLE predicoes DROP CONSTRAINT IF EXISTS predicoes_user_id_fkey;
ALTER TABLE predicoes
    ADD CONSTRAINT predicoes_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE;

-- 3. Exclusão da própria conta pelo app (direito de eliminação, LGPD art. 18, VI).
--    A chave anon não pode apagar de auth.users; esta função roda com permissão
--    do dono do banco, mas só apaga o usuário que está chamando (auth.uid()).
CREATE OR REPLACE FUNCTION public.excluir_minha_conta()
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    IF auth.uid() IS NULL THEN
        RAISE EXCEPTION 'não autenticado';
    END IF;
    DELETE FROM public.predicoes WHERE user_id = auth.uid();
    DELETE FROM public.user_profiles WHERE id = auth.uid();
    DELETE FROM auth.users WHERE id = auth.uid();
END;
$$;

REVOKE ALL ON FUNCTION public.excluir_minha_conta() FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.excluir_minha_conta() TO authenticated;

-- ============================================================================
-- 4. LIMPEZA DAS AVALIAÇÕES SEM DONO (opcional, irreversível)
--    Depois do passo 1 elas ficam invisíveis para o app, mas continuam no banco.
--    Confira antes quantas são:
--        SELECT count(*) FROM predicoes WHERE user_id IS NULL;
--    Para apagá-las e impedir novas, descomente as duas linhas abaixo:
-- ============================================================================
-- DELETE FROM predicoes WHERE user_id IS NULL;
-- ALTER TABLE predicoes ALTER COLUMN user_id SET NOT NULL;
