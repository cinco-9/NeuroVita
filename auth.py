# -*- coding: utf-8 -*-
"""
MÓDULO DE AUTENTICAÇÃO - SUPABASE AUTH
Sistema de login, cadastro e gerenciamento de usuários
"""

import streamlit as st
from supabase_db import db as supabase_db
from typing import Optional, Dict

class Auth:
    """Classe para gerenciar autenticação com Supabase"""

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
                # Salvar usuário no session_state
                st.session_state.user = {
                    'id': response.user.id,
                    'email': response.user.email
                }
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
                return False, f"Erro ao fazer login: {error_msg}"

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
            response = supabase_db.client.auth.sign_up({
                "email": email,
                "password": password
            })

            if response.user:
                # Fazer login automático (se não precisar confirmar email)
                if response.session:
                    st.session_state.user = {
                        'id': response.user.id,
                        'email': response.user.email
                    }

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
                return False, "Senha deve ter no mínimo 6 caracteres"
            else:
                return False, f"Erro ao cadastrar: {error_msg}"

    @staticmethod
    def logout():
        """Faz logout do usuário"""
        try:
            if supabase_db.conectado and supabase_db.client:
                supabase_db.client.auth.sign_out()
        except:
            pass

        # Limpar session_state
        if 'user' in st.session_state:
            del st.session_state.user

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
                    st.session_state.perfil_nome = perfil.get('nome', '')
                    st.session_state.perfil_idade = perfil.get('idade', 30)
                    st.session_state.perfil_sexo = perfil.get('sexo', 'Masculino')
                    st.session_state.perfil_altura = perfil.get('altura', 170.0)
                    st.session_state.perfil_peso = perfil.get('peso', 70.0)
                    st.session_state.perfil_imc = perfil.get('imc', 0.0)
                    st.session_state.perfil_alergias = perfil.get('alergias', '')
                    st.session_state.perfil_hist_familiar = 'Sim' if perfil.get('hist_familiar_diabetes', False) else 'Não'
                    st.session_state.perfil_medicacoes = perfil.get('medicacoes', '')
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
            return False, f"Erro ao salvar perfil: {str(e)}"

# Instância global
auth = Auth()
