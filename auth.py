# -*- coding: utf-8 -*-
"""
MÓDULO DE AUTENTICAÇÃO - SUPABASE AUTH
Sistema de login, cadastro e gerenciamento de usuários
"""

import json
from datetime import datetime, timezone
import streamlit as st
from supabase_db import db as supabase_db
from privacidade import VERSAO_TERMO
from typing import Optional, Dict

COOKIE_SESSAO = "dp_refresh_token"
COOKIE_DURACAO_S = 30 * 24 * 3600
# Manter igual ao mínimo configurado no Supabase (Authentication > Policies)
SENHA_MINIMA = 8


class Auth:
    """Classe para gerenciar autenticação com Supabase"""

    @staticmethod
    def _iniciar_sessao(user, session):
        metadata = user.user_metadata or {}
        st.session_state.user = {
            'id': user.id,
            'email': user.email,
            'consentimento_versao': metadata.get('consentimento_versao'),
        }
        st.session_state.logout_feito = False
        if session and session.refresh_token:
            st.session_state.cookie_pendente = session.refresh_token

    @staticmethod
    def sincronizar_cookie():
        """Grava/apaga o cookie de sessão no navegador; chamar em toda execução do app"""
        if 'cookie_pendente' not in st.session_state:
            return
        token = st.session_state.pop('cookie_pendente')
        max_age = COOKIE_DURACAO_S if token else 0
        st.html(
            f"""<script>
            document.cookie = "{COOKIE_SESSAO}=" + encodeURIComponent({json.dumps(token or "")})
                + "; path=/; max-age={max_age}; SameSite=Strict"
                + (location.protocol === "https:" ? "; Secure" : "");
            </script>""",
            unsafe_allow_javascript=True,
        )

    @staticmethod
    def restaurar_sessao():
        """Restaura o login a partir do cookie, se houver"""
        if Auth.is_logged_in() or st.session_state.get('logout_feito'):
            return
        # st.context.cookies reflete o carregamento da página; não muda após logout nesta sessão
        token = st.context.cookies.get(COOKIE_SESSAO)
        if not token or not supabase_db.conectado or not supabase_db.client:
            return

        try:
            response = supabase_db.client.auth.refresh_session(token)
            if response.user:
                # O Supabase gira o refresh token a cada uso, então o cookie é regravado
                Auth._iniciar_sessao(response.user, response.session)
                Auth.carregar_perfil()
                return
        except Exception:
            pass
        st.session_state.cookie_pendente = None

    @staticmethod
    def is_logged_in() -> bool:
        """Verifica se usuário está logado"""
        return 'user' in st.session_state and st.session_state.user is not None

    @staticmethod
    def get_user() -> Optional[Dict]:
        """Retorna usuário logado ou None"""
        return st.session_state.get('user', None)

    @staticmethod
    def get_user_id() -> Optional[str]:
        """Retorna ID do usuário logado"""
        user = Auth.get_user()
        return user['id'] if user else None

    @staticmethod
    def get_user_email() -> Optional[str]:
        """Retorna email do usuário logado"""
        user = Auth.get_user()
        return user['email'] if user else None

    @staticmethod
    def _dados_consentimento() -> Dict:
        return {
            'consentimento_versao': VERSAO_TERMO,
            'consentimento_em': datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def consentimento_em_dia() -> bool:
        """True se o usuário logado aceitou a versão atual do termo de privacidade"""
        user = Auth.get_user()
        return bool(user) and user.get('consentimento_versao') == VERSAO_TERMO

    @staticmethod
    def registrar_consentimento() -> tuple[bool, str]:
        """Grava o aceite do termo nos metadados do usuário (contas criadas antes do termo)"""
        if not supabase_db.conectado or not supabase_db.client:
            return False, "Erro: Supabase não está conectado"
        try:
            supabase_db.client.auth.update_user({"data": Auth._dados_consentimento()})
            st.session_state.user['consentimento_versao'] = VERSAO_TERMO
            return True, "Consentimento registrado"
        except Exception as e:
            print(f"Erro ao registrar consentimento: {e}")
            return False, "Não foi possível registrar o consentimento. Tente novamente."

    @staticmethod
    def excluir_conta() -> tuple[bool, str]:
        """Apaga a conta, o perfil e todas as avaliações do usuário logado"""
        if not Auth.is_logged_in():
            return False, "Usuário não está logado"
        if not supabase_db.conectado or not supabase_db.client:
            return False, "Erro: Supabase não está conectado"
        try:
            # Função excluir_minha_conta() criada em privacidade_lgpd.sql
            supabase_db.client.rpc('excluir_minha_conta').execute()
        except Exception as e:
            print(f"Erro ao excluir conta: {e}")
            return False, "Não foi possível excluir a conta. Tente novamente mais tarde."
        Auth.logout()
        return True, "Sua conta e todos os seus dados foram excluídos."

    @staticmethod
    def login(email: str, password: str) -> tuple[bool, str]:
        """
        Faz login do usuário

        Returns:
            tuple: (sucesso, mensagem)
        """
        if not supabase_db.conectado or not supabase_db.client:
            return False, "Erro: Supabase não está conectado"

        try:
            # Fazer login
            response = supabase_db.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })

            if response.user:
                Auth._iniciar_sessao(response.user, response.session)
                return True, "Login realizado com sucesso!"
            else:
                return False, "Email ou senha incorretos"

        except Exception as e:
            error_msg = str(e)
            if "Invalid login credentials" in error_msg:
                return False, "Email ou senha incorretos"
            elif "Email not confirmed" in error_msg:
                return False, "Email não confirmado. Verifique sua caixa de entrada."
            else:
                print(f"Erro ao fazer login: {error_msg}")
                return False, "Não foi possível fazer login. Tente novamente mais tarde."

    @staticmethod
    def signup(email: str, password: str, nome: str = "") -> tuple[bool, str]:
        """
        Cadastra novo usuário

        Returns:
            tuple: (sucesso, mensagem)
        """
        if not supabase_db.conectado or not supabase_db.client:
            return False, "Erro: Supabase não está conectado"

        try:
            # Criar usuário
            # Só é chamado depois do aceite do termo; o registro fica nos metadados do usuário
            response = supabase_db.client.auth.sign_up({
                "email": email,
                "password": password,
                "options": {"data": Auth._dados_consentimento()},
            })

            if response.user:
                # Fazer login automático (se não precisar confirmar email)
                if response.session:
                    Auth._iniciar_sessao(response.user, response.session)

                    # Criar perfil inicial
                    if nome:
                        try:
                            supabase_db.client.table('user_profiles').insert({
                                'id': response.user.id,
                                'nome': nome
                            }).execute()
                        except:
                            pass  # Perfil pode ser criado depois

                    return True, "Cadastro realizado com sucesso!"
                else:
                    return True, "Cadastro realizado! Verifique seu email para confirmar."
            else:
                return False, "Erro ao criar conta"

        except Exception as e:
            error_msg = str(e)
            if "already registered" in error_msg.lower() or "already been registered" in error_msg.lower():
                return False, "Este email já está cadastrado. Faça login!"
            elif "Password should be at least" in error_msg:
                return False, f"Senha deve ter no mínimo {SENHA_MINIMA} caracteres"
            else:
                print(f"Erro ao cadastrar: {error_msg}")
                return False, "Não foi possível criar a conta. Tente novamente mais tarde."

    @staticmethod
    def solicitar_reset_senha(email: str) -> tuple[bool, str]:
        """Envia email com código de recuperação de senha"""
        if not supabase_db.conectado or not supabase_db.client:
            return False, "Erro: Supabase não está conectado"

        try:
            supabase_db.client.auth.reset_password_for_email(email)
            # Mensagem neutra para não revelar se o email está cadastrado
            return True, "Se este email estiver cadastrado, você receberá um código de recuperação."
        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower() or "seconds" in error_msg.lower():
                return False, "Aguarde alguns instantes antes de pedir um novo código."
            print(f"Erro ao enviar código: {error_msg}")
            return False, "Não foi possível enviar o código. Tente novamente mais tarde."

    @staticmethod
    def redefinir_senha(email: str, codigo: str, nova_senha: str) -> tuple[bool, str]:
        """Valida o código recebido por email e define a nova senha"""
        if not supabase_db.conectado or not supabase_db.client:
            return False, "Erro: Supabase não está conectado"

        try:
            response = supabase_db.client.auth.verify_otp({
                "email": email,
                "token": codigo,
                "type": "recovery"
            })
            if not response.user:
                return False, "Código inválido ou expirado"
        except Exception:
            return False, "Código inválido ou expirado"

        try:
            supabase_db.client.auth.update_user({"password": nova_senha})
            Auth._iniciar_sessao(response.user, response.session)
            return True, "Senha alterada com sucesso!"
        except Exception as e:
            error_msg = str(e)
            if "should be different" in error_msg.lower():
                return False, "A nova senha deve ser diferente da anterior"
            if "Password should be at least" in error_msg:
                return False, f"Senha deve ter no mínimo {SENHA_MINIMA} caracteres"
            print(f"Erro ao alterar senha: {error_msg}")
            return False, "Não foi possível alterar a senha. Tente novamente mais tarde."

    @staticmethod
    def logout():
        """Faz logout do usuário"""
        try:
            if supabase_db.conectado and supabase_db.client:
                # "local": sai só deste navegador; o padrão "global" derrubaria os outros aparelhos
                supabase_db.client.auth.sign_out({"scope": "local"})
        except:
            pass

        # Limpar session_state
        if 'user' in st.session_state:
            del st.session_state.user
        st.session_state.logout_feito = True
        st.session_state.cookie_pendente = None

        # Limpar perfil
        keys_to_clear = [
            'perfil_nome', 'perfil_idade', 'perfil_sexo', 'perfil_altura',
            'perfil_peso', 'perfil_imc', 'perfil_alergias', 'perfil_hist_familiar',
            'perfil_medicacoes', 'perfil_preenchido'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

    @staticmethod
    def carregar_perfil():
        """Carrega perfil do usuário do banco"""
        user_id = Auth.get_user_id()
        if not user_id:
            return False

        try:
            if supabase_db.conectado and supabase_db.client:
                response = supabase_db.client.table('user_profiles').select('*').eq('id', user_id).execute()

                if response.data and len(response.data) > 0:
                    perfil = response.data[0]

                    # Carregar no session_state
                    st.session_state.perfil_nome = perfil.get('nome') or ''
                    # O banco pode devolver 175 em vez de 175.0; os st.number_input exigem tipos consistentes
                    st.session_state.perfil_idade = int(perfil.get('idade') or 30)
                    st.session_state.perfil_sexo = perfil.get('sexo') or 'Masculino'
                    st.session_state.perfil_altura = float(perfil.get('altura') or 170.0)
                    st.session_state.perfil_peso = float(perfil.get('peso') or 70.0)
                    st.session_state.perfil_imc = float(perfil.get('imc') or 0.0)
                    st.session_state.perfil_alergias = perfil.get('alergias') or ''
                    st.session_state.perfil_hist_familiar = 'Sim' if perfil.get('hist_familiar_diabetes') else 'Não'
                    st.session_state.perfil_medicacoes = perfil.get('medicacoes') or ''
                    st.session_state.perfil_preenchido = True

                    return True
        except Exception as e:
            print(f"Erro ao carregar perfil: {e}")

        return False

    @staticmethod
    def salvar_perfil(nome: str, idade: int, sexo: str, altura: float, peso: float,
                     imc: float, alergias: str, hist_familiar: str, medicacoes: str) -> tuple[bool, str]:
        """Salva perfil do usuário no banco"""
        user_id = Auth.get_user_id()
        if not user_id:
            return False, "Usuário não está logado"

        if not supabase_db.conectado or not supabase_db.client:
            return False, "Banco de dados indisponível"

        try:
            if supabase_db.conectado and supabase_db.client:
                dados = {
                    'id': user_id,
                    'nome': nome,
                    'idade': idade,
                    'sexo': sexo,
                    'altura': altura,
                    'peso': peso,
                    'imc': imc,
                    'alergias': alergias,
                    'hist_familiar_diabetes': (hist_familiar == 'Sim'),
                    'medicacoes': medicacoes
                }

                # Tentar inserir ou atualizar
                response = supabase_db.client.table('user_profiles').upsert(dados).execute()

                return True, "Perfil salvo no banco de dados!"
        except Exception as e:
            print(f"Erro ao salvar perfil: {e}")
            return False, "não foi possível salvar no banco de dados"

# Instância global
auth = Auth()
