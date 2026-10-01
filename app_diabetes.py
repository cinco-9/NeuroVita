# -*- coding: utf-8 -*-
"""
INTERFACE WEB - SISTEMA DE PREDICAO DE DIABETES
TCC: Agente de IA para Estimativa de Risco de Diabetes Tipo 2
Autor: David Reis
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
# import lightgbm as lgb  # Desabilitado para economizar RAM
# from catboost import CatBoostClassifier  # Desabilitado para economizar RAM
import matplotlib.pyplot as plt
# import seaborn as sns  # Desabilitado - usa matplotlib
import warnings
import os
import re
import tempfile
import uuid
from datetime import datetime
warnings.filterwarnings('ignore')

# Importar módulos
from supabase_db import db as supabase_db
from auth import auth, SENHA_MINIMA
from privacidade import TERMO_PRIVACIDADE
from pdf_generator import gerar_relatorio_predicao
# from shap_explicabilidade import criar_grafico_barras_shap, obter_top_features_shap  # Desabilitado - muito pesado
from gauge_component import criar_gauge_risco
from analise_temporal import criar_grafico_evolucao, criar_grafico_progresso, calcular_estatisticas_evolucao
import json

# Configuração da página
st.set_page_config(
    page_title="Predição de Diabetes",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Dark Mode Profissional - SEM EMOJIS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #141414;
        --sidebar: #171717;
        --surface: #1b1b1b;
        --surface-2: #222222;
        --border: #2c2c2c;
        --border-strong: #3a3a3a;
        --text: #e8e8e8;
        --muted: #a3a3a3;
        --accent: #4fb9c4;
        --accent-hover: #6cc8d1;
        --accent-soft: rgba(79, 185, 196, 0.14);
        --on-accent: #0e1f22;
        --success: #4ade80;
        --success-soft: rgba(34, 197, 94, 0.14);
        --danger: #f87171;
        --danger-soft: rgba(239, 68, 68, 0.14);
        --warning: #fbbf24;
        --warning-soft: rgba(245, 158, 11, 0.14);
    }

    .stApp {
        background: var(--bg);
    }

    .block-container {
        max-width: 1150px !important;
        padding-top: 3.5rem !important;  /* abaixo da barra do topo do Streamlit */
        padding-bottom: 1.5rem !important;
    }

    /* Menos espaço entre elementos */
    [data-testid="stVerticalBlock"] {
        gap: 0.6rem !important;
    }

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text);
    }

    .main-header {
        font-size: 22px;
        font-weight: 800;
        text-align: center;
        color: var(--text);
        padding: 0 8px 2px;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }

    .sub-header {
        font-size: 12px;
        text-align: center;
        color: var(--muted);
        padding-bottom: 6px;
        font-weight: 400;
        letter-spacing: 0.5px;
    }


    /* Logo Icon */
    .logo-icon {
        text-align: center;
        font-size: 50px;
        margin-bottom: 5px;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    .metric-box {
        background: var(--surface);
        padding: 16px 20px;
        border-radius: 20px;
        border: 1px solid var(--border);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }

    .metric-box:hover {
        transform: translateY(-5px);
        border-color: var(--accent);
    }

    .result-high {
        background: var(--danger-soft);
        padding: 16px 20px;
        border-radius: 20px;
        border: 1px solid var(--danger);
    }

    .result-high h3 {
        color: var(--danger) !important;
        -webkit-text-fill-color: var(--danger) !important;
        font-weight: 700;
        font-size: 18px;
    }

    .result-low {
        background: var(--success-soft);
        padding: 16px 20px;
        border-radius: 20px;
        border: 1px solid var(--success);
    }

    .result-low h3 {
        color: var(--success) !important;
        -webkit-text-fill-color: var(--success) !important;
        font-weight: 700;
        font-size: 18px;
    }

    .stButton>button {
        background: var(--surface-2);
        color: var(--text);
        border: 1px solid var(--border-strong);
        border-radius: 10px;
        padding: 6px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        border-color: var(--accent);
        color: var(--accent);
    }

    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        background-color: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
        padding: 8px 12px !important;
        font-size: 13px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        -webkit-text-fill-color: var(--text) !important;
    }

    /* Ícone de mostrar/esconder senha */
    .stTextInput button[kind="icon"] {
        background: transparent !important;
        border: none !important;
        color: var(--muted) !important;
    }

    .stTextInput button[kind="icon"]:hover {
        background: var(--accent-soft) !important;
        color: var(--accent) !important;
    }

    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
        transform: translateY(-1px) scale(1.01);
    }

    /* -webkit-text-fill-color do input sobrescreve a cor do placeholder no Chrome */
    input::placeholder {
        color: #8a8a8a !important;
        -webkit-text-fill-color: #8a8a8a !important;
        opacity: 1 !important;
        font-weight: 500 !important;
    }

    /* Labels dos inputs */
    .stTextInput label,
    .stNumberInput label,
    .stSelectbox label {
        color: var(--muted) !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        margin-bottom: 2px !important;
        letter-spacing: 0.3px;
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid var(--border);
    }

    /* Tabs melhoradas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: var(--surface);
        padding: 6px;
        border-radius: 14px;
        margin-bottom: 15px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        color: var(--muted) !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.5px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: var(--surface-2) !important;
        transform: translateY(-2px);
        color: var(--text) !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--accent-soft) !important;
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        transform: translateY(-1px);
    }

    /* Botão melhorado */
    button[kind="primary"],
    button[kind="primaryFormSubmit"] {
        background: var(--accent) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 8px 20px !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        color: var(--on-accent) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }

    button[kind="primary"] *,
    button[kind="primaryFormSubmit"] * {
        color: var(--on-accent) !important;
    }

    button[kind="primary"]:hover,
    button[kind="primaryFormSubmit"]:hover {
        transform: translateY(-3px) scale(1.02) !important;
        background: var(--accent-hover) !important;
    }

    button[kind="primary"]:active,
    button[kind="primaryFormSubmit"]:active {
        transform: translateY(-1px) scale(0.98) !important;
    }

    /* Container do form */
    [data-testid="stForm"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 14px !important;
    }

    /* Headings melhorados */
    h1, h2, h3, h4 {
        color: var(--text) !important;
        font-weight: 700 !important;
    }

    h2 {
        font-size: 18px !important;
        margin: 4px 0 !important;
        padding: 0 !important;
    }

    h3 {
        font-size: 15px !important;
        margin-bottom: 4px !important;
        padding: 4px 0 !important;
        color: var(--accent) !important;
        font-weight: 700 !important;
    }

    h4 {
        font-size: 16px !important;
        margin-bottom: 8px !important;
        margin-top: 0 !important;
        color: var(--text) !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
    }

    p, li {
        color: var(--text) !important;
        font-size: 13px !important;
    }

    p {
        margin-bottom: 4px !important;
    }

    p.prob {
        font-size: 20px !important;
        margin: 2px 0 4px !important;
    }

    .auth-footer {
        text-align: center;
        margin-top: 8px;
    }

    .auth-footer p {
        margin-bottom: 4px;
    }

    /* Linha separadora */
    hr {
        border-color: var(--border) !important;
        margin: 4px 0 !important;
    }

    /* Efeitos de erro e sucesso melhorados */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 14px !important;
        padding: 8px 12px !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        animation: slideInRight 0.4s ease-out;
    }

    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    .stSuccess {
        background: var(--success-soft) !important;
        border: 1px solid var(--success) !important;
    }

    .stError {
        background: var(--danger-soft) !important;
        border: 1px solid var(--danger) !important;
    }

    .stWarning {
        background: var(--warning-soft) !important;
        border: 1px solid var(--warning) !important;
    }

    .stInfo {
        background: var(--accent-soft) !important;
        border: 1px solid var(--accent) !important;
    }

    /* Melhorar selectbox */
    .stSelectbox>div>div>select {
        background-color: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
        padding: 12px !important;
    }

    /* Animação de fade-in */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .main-header {
        animation: fadeInUp 0.8s ease-out;
    }

    .sub-header {
        animation: fadeInUp 1s ease-out;
    }

    [data-testid="stForm"] {
        animation: fadeInUp 1.2s ease-out;
    }

    /* O Streamlit compensa a margem de 16px dos parágrafos com -16px no container; como a margem foi reduzida, zera a compensação */
    [data-testid="stMarkdownContainer"] {
        margin-bottom: 0 !important;
    }

    [data-testid="stAlertContainer"] p {
        margin-bottom: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Modelos em JSON nativo do XGBoost, carregados pelo Booster: não dependem de pickle nem do scikit-learn,
# cujas versões variam entre esta máquina e o Streamlit Cloud
@st.cache_resource
def carregar_modelo(nome):
    """Devolve (modelo, meta) de <nome>.json / <nome>_meta.json; meta['tipo'] = 'xgboost' ou 'random_forest'"""
    with open(f'{nome}_meta.json', encoding='utf-8') as f:
        meta = json.load(f)
    if meta['tipo'] == 'xgboost':
        modelo = xgb.Booster()
        modelo.load_model(f'{nome}.json')
    else:
        with open(f'{nome}.json', encoding='utf-8') as f:
            modelo = [{k: np.array(v) for k, v in arvore.items()} for arvore in json.load(f)['arvores']]
    return modelo, meta


def prever_probabilidade(modelo, meta, linha):
    """Probabilidade de diabetes para um dict {feature: valor}"""
    x = np.array([linha[f] for f in meta['features']], dtype=float)
    if meta['tipo'] == 'xgboost':
        nomes = meta['features'] if modelo.feature_names else None
        return float(modelo.predict(xgb.DMatrix(x[None, :], feature_names=nomes))[0])

    # Random Forest exportado por otimizar_modelo_clinico.py: mesma sequência do pipeline de treino
    for i, f in enumerate(meta['features']):
        if f in meta['zero_e_ausente'] and x[i] == 0:
            x[i] = np.nan  # no dataset Pima, zero nesses exames significa "não medido"
    x = np.where(np.isnan(x), meta['imputer_medianas'], x)
    # O scikit-learn arredonda a entrada para float32 e compara com os limites em float64
    z = ((x - meta['scaler_media']) / meta['scaler_escala']).astype(np.float32).astype(np.float64)
    total = 0.0
    for a in modelo:
        no = 0
        while a['esquerda'][no] != -1:
            no = a['esquerda'][no] if z[a['variavel'][no]] <= a['limite'][no] else a['direita'][no]
        total += a['prob'][no]
    return float(total / len(modelo))

# Inicializar session_state
if 'resultado_mostrado' not in st.session_state:
    st.session_state.resultado_mostrado = False

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Dados do perfil
if 'perfil_nome' not in st.session_state:
    st.session_state.perfil_nome = ""
if 'perfil_idade' not in st.session_state:
    st.session_state.perfil_idade = 30
if 'perfil_sexo' not in st.session_state:
    st.session_state.perfil_sexo = "Masculino"
if 'perfil_altura' not in st.session_state:
    st.session_state.perfil_altura = 170.0
if 'perfil_peso' not in st.session_state:
    st.session_state.perfil_peso = 70.0
if 'perfil_imc' not in st.session_state:
    st.session_state.perfil_imc = 0.0
if 'perfil_alergias' not in st.session_state:
    st.session_state.perfil_alergias = ""
if 'perfil_hist_familiar' not in st.session_state:
    st.session_state.perfil_hist_familiar = "Não"
if 'perfil_medicacoes' not in st.session_state:
    st.session_state.perfil_medicacoes = ""
if 'perfil_preenchido' not in st.session_state:
    st.session_state.perfil_preenchido = False

# ============================================================================
# VERIFICAÇÃO DE AUTENTICAÇÃO
# ============================================================================

auth.restaurar_sessao()
auth.sincronizar_cookie()

if not auth.is_logged_in():
    # Container centralizado melhorado
    # ~430px de largura dentro do block-container de 1150px
    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:
        # Logo/Ícone
        st.markdown("""
            <div class="logo-icon">
                ⚕️
            </div>
        """, unsafe_allow_html=True)

        # Cabeçalho
        st.markdown('<div class="main-header">Sistema de Predição de Diabetes</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Análise inteligente de risco com IA</div>', unsafe_allow_html=True)

        if st.session_state.pop('conta_excluida', False):
            st.success("Sua conta e todos os seus dados foram excluídos.")

        # Tabs de Login/Cadastro
        tab1, tab2 = st.tabs(["Login", "Cadastrar"])

        with tab1:
            with st.form("login_form"):
                st.markdown("#### Acesse sua conta")
                st.text_input("Email", key="email_login", placeholder="seu@email.com")
                st.text_input("Senha", type="password", key="senha_login", placeholder="Sua senha")

                submit_login = st.form_submit_button("ENTRAR", width="stretch", type="primary")

                if submit_login:
                    email_login = st.session_state.email_login
                    senha_login = st.session_state.senha_login

                    if not email_login or not senha_login:
                        st.error("Por favor, preencha email e senha")
                    else:
                        with st.spinner("Fazendo login..."):
                            sucesso, mensagem = auth.login(email_login, senha_login)
                            if sucesso:
                                st.success(mensagem)
                                auth.carregar_perfil()
                                st.rerun()
                            else:
                                st.error(mensagem)

            # Só aparece com o SMTP configurado no Supabase; sem ele o email com o código não chega
            if os.getenv("RECUPERACAO_SENHA_ATIVA") == "1":
                with st.expander("Esqueci minha senha"):
                    if 'reset_email' not in st.session_state:
                        with st.form("reset_solicitar_form"):
                            st.text_input("Email da conta", key="reset_email_input", placeholder="seu@email.com")
                            enviar_codigo = st.form_submit_button("ENVIAR CÓDIGO", width="stretch", type="primary")

                            if enviar_codigo:
                                email_reset = st.session_state.reset_email_input.strip()
                                if not email_reset:
                                    st.error("Informe o email da conta")
                                else:
                                    with st.spinner("Enviando código..."):
                                        sucesso, mensagem = auth.solicitar_reset_senha(email_reset)
                                    if sucesso:
                                        st.session_state.reset_email = email_reset
                                        st.rerun()
                                    else:
                                        st.error(mensagem)
                    else:
                        st.info(f"Se {st.session_state.reset_email} estiver cadastrado, você receberá um código por email.")
                        with st.form("reset_confirmar_form"):
                            st.text_input("Código recebido", key="reset_codigo", max_chars=10, placeholder="Código do email")
                            st.text_input("Nova senha", type="password", key="reset_nova_senha", placeholder=f"Mínimo {SENHA_MINIMA} caracteres")
                            st.text_input("Confirmar nova senha", type="password", key="reset_confirma_senha", placeholder="Repita a senha")
                            redefinir = st.form_submit_button("REDEFINIR SENHA", width="stretch", type="primary")

                            if redefinir:
                                codigo = st.session_state.reset_codigo.strip()
                                nova_senha = st.session_state.reset_nova_senha
                                if not codigo or not nova_senha:
                                    st.error("Preencha o código e a nova senha")
                                elif nova_senha != st.session_state.reset_confirma_senha:
                                    st.error("As senhas não coincidem")
                                elif len(nova_senha) < SENHA_MINIMA:
                                    st.error(f"Senha deve ter no mínimo {SENHA_MINIMA} caracteres")
                                else:
                                    with st.spinner("Redefinindo senha..."):
                                        sucesso, mensagem = auth.redefinir_senha(st.session_state.reset_email, codigo, nova_senha)
                                    if sucesso:
                                        del st.session_state.reset_email
                                        auth.carregar_perfil()
                                        st.rerun()
                                    else:
                                        st.error(mensagem)

                        if st.button("Usar outro email / reenviar código", width="stretch"):
                            del st.session_state.reset_email
                            st.rerun()

        with tab2:
            with st.expander("Ler Termo de Consentimento e Política de Privacidade"):
                st.markdown(TERMO_PRIVACIDADE)

            with st.form("signup_form"):
                st.markdown("#### Crie sua conta gratuitamente")
                st.text_input("Nome Completo", key="nome_cadastro", placeholder="Seu nome completo")
                st.text_input("Email", key="email_cadastro", placeholder="seu@email.com")

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.text_input("Senha", type="password", key="senha_cadastro",
                                help=f"Mínimo {SENHA_MINIMA} caracteres", placeholder="Sua senha")
                with col_s2:
                    st.text_input("Confirmar Senha", type="password", key="senha_confirma",
                                placeholder="Repita a senha")

                st.checkbox(
                    "Li e aceito o Termo de Consentimento e a Política de Privacidade, "
                    "incluindo o tratamento dos meus dados de saúde",
                    key="aceite_termo_cadastro",
                )

                submit_cadastro = st.form_submit_button("CRIAR CONTA", width="stretch", type="primary")

                if submit_cadastro:
                    nome_cadastro = st.session_state.nome_cadastro
                    email_cadastro = st.session_state.email_cadastro
                    senha_cadastro = st.session_state.senha_cadastro
                    senha_confirma = st.session_state.senha_confirma

                    if not email_cadastro or not senha_cadastro or not nome_cadastro:
                        st.error("Por favor, preencha todos os campos")
                    elif not st.session_state.aceite_termo_cadastro:
                        st.error("Para criar a conta, é preciso aceitar o Termo de Consentimento")
                    elif senha_cadastro != senha_confirma:
                        st.error("As senhas não coincidem")
                    elif len(senha_cadastro) < SENHA_MINIMA:
                        st.error(f"Senha deve ter no mínimo {SENHA_MINIMA} caracteres")
                    else:
                        with st.spinner("Criando conta..."):
                            sucesso, mensagem = auth.signup(email_cadastro, senha_cadastro, nome_cadastro)
                            if sucesso:
                                st.success(mensagem)
                                st.rerun()
                            else:
                                st.error(mensagem)


        # Footer
        st.markdown("""
            <div class="auth-footer">
                <p style='font-size: 12px; color: var(--muted);'>
                    Desenvolvido por <strong style='color: var(--accent);'>David Reis</strong> | 2026
                </p>
                <p style='font-size: 11px; color: var(--muted); margin-top: 8px;'>
                    Validação cross-cultural Brasil e USA
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.stop()

# ============================================================================
# CONSENTIMENTO (contas criadas antes do termo, ou termo atualizado)
# ============================================================================

def secao_excluir_conta():
    """Exclusão da conta e de todos os dados (LGPD art. 18, VI)"""
    with st.expander("Excluir minha conta"):
        st.warning("Isto apaga **definitivamente** sua conta, seu perfil e todas as suas avaliações. "
                   "Não é possível desfazer.")
        confirmacao = st.text_input("Para confirmar, digite EXCLUIR", key="confirma_exclusao")
        if st.button("Excluir minha conta e meus dados", type="primary", width="stretch",
                     disabled=confirmacao.strip().upper() != "EXCLUIR"):
            with st.spinner("Excluindo..."):
                sucesso, mensagem = auth.excluir_conta()
            if sucesso:
                st.session_state.conta_excluida = True
                st.rerun()
            else:
                st.error(mensagem)

if not auth.consentimento_em_dia():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="main-header">Privacidade dos seus dados</div>', unsafe_allow_html=True)
        st.info("Para continuar usando o sistema, leia e aceite o termo abaixo.")
        with st.container(height=400):
            st.markdown(TERMO_PRIVACIDADE)

        aceite = st.checkbox(
            "Li e aceito o Termo de Consentimento e a Política de Privacidade, "
            "incluindo o tratamento dos meus dados de saúde"
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("CONTINUAR", type="primary", width="stretch", disabled=not aceite):
                sucesso, mensagem = auth.registrar_consentimento()
                if sucesso:
                    st.rerun()
                else:
                    st.error(mensagem)
        with col_b:
            if st.button("Não aceito (sair)", width="stretch"):
                auth.logout()
                st.rerun()
        st.markdown("---")
        secao_excluir_conta()
    st.stop()

# ============================================================================
# USUÁRIO LOGADO - SIDEBAR
# ============================================================================

st.sidebar.markdown("# Predição de Diabetes")
st.sidebar.markdown("---")

# Info do usuário
if st.session_state.perfil_nome:
    nome_exibir = st.session_state.perfil_nome
else:
    nome_exibir = auth.get_user_email().split('@')[0]

st.sidebar.success(f"**{nome_exibir}**")

if st.sidebar.button("Sair", width="stretch"):
    auth.logout()
    st.rerun()

st.sidebar.markdown("---")

# Navegação
pagina = st.sidebar.radio(
    "Navegação",
    ["Início", "Meu Perfil", "Modelo Clínico", "Modelo Comportamental", "Comparação", "Histórico & Estatísticas"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Sobre o sistema
**Autor:** David Reis

**Datasets:**
- Pima Indians (clínico)
- BRFSS 2015 (comportamental)

**Modelos:**
- Random Forest (clínico)
- XGBoost (comportamental)
""")

# ============================================================================
# PÁGINA INÍCIO
# ============================================================================

if pagina == "Início":
    st.markdown('<div class="main-header">Sistema de Predição de Diabetes</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Análise inteligente de risco de diabetes tipo 2 com IA</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h3>Modelo Clínico</h3>
            <p><strong>F1-Score: 0.69</strong></p>
            <p>Dataset: Pima Indians<br>
            Features: Exames laboratoriais<br>
            <strong>Melhor precisão</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h3>Modelo Comportamental</h3>
            <p><strong>F1-Score: 0.46</strong></p>
            <p>Dataset: BRFSS (253k registros)<br>
            Features: Hábitos de vida<br>
            <strong>Triagem inicial</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <h3>Descoberta</h3>
            <p><strong>Clínico +51% melhor!</strong></p>
            <p>Features clínicas são superiores<br>
            Qualidade > Quantidade<br>
            <strong>768 vs 253k registros</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("Use o menu lateral para testar os modelos de predição!")


# ============================================================================
# PÁGINA MEU PERFIL
# ============================================================================

elif pagina == "Meu Perfil":
    st.markdown('<div class="main-header">Meu Perfil</div>', unsafe_allow_html=True)
    
    st.markdown("### Informações Pessoais")
    
    col1, col2 = st.columns(2)
    
    with col1:
        nome = st.text_input("Nome Completo", value=st.session_state.perfil_nome)
        idade = st.number_input("Idade", min_value=1, max_value=120, value=st.session_state.perfil_idade)
        sexo = st.selectbox("Sexo", ["Masculino", "Feminino"], index=0 if st.session_state.perfil_sexo == "Masculino" else 1)
        altura = st.number_input("Altura (cm)", min_value=50.0, max_value=250.0, value=st.session_state.perfil_altura, step=0.1)
    
    with col2:
        peso = st.number_input("Peso (kg)", min_value=20.0, max_value=300.0, value=st.session_state.perfil_peso, step=0.1)
        
        if altura > 0 and peso > 0:
            imc = peso / ((altura/100) ** 2)
            st.metric("IMC Calculado", f"{imc:.1f}")
        else:
            imc = 0
        
        hist_familiar = st.selectbox("Histórico Familiar de Diabetes", ["Não", "Sim"], 
                                     index=0 if st.session_state.perfil_hist_familiar == "Não" else 1)
    
    st.markdown("### Histórico Médico")
    alergias = st.text_area("Alergias", value=st.session_state.perfil_alergias, 
                            placeholder="Ex: Penicilina, Frutos do mar...")
    medicacoes = st.text_area("Medicações Atuais", value=st.session_state.perfil_medicacoes,
                              placeholder="Ex: Metformina 500mg...")
    
    if st.button("Salvar Perfil", type="primary", width="stretch"):
        st.session_state.perfil_nome = nome
        st.session_state.perfil_idade = idade
        st.session_state.perfil_sexo = sexo
        st.session_state.perfil_altura = altura
        st.session_state.perfil_peso = peso
        st.session_state.perfil_imc = imc
        st.session_state.perfil_alergias = alergias
        st.session_state.perfil_hist_familiar = hist_familiar
        st.session_state.perfil_medicacoes = medicacoes
        st.session_state.perfil_preenchido = True
        
        sucesso, msg_bd = auth.salvar_perfil(nome, idade, sexo, altura, peso, imc, alergias, hist_familiar, medicacoes)
        
        if sucesso:
            st.success("Perfil salvo com sucesso!")
        else:
            st.warning(f"Perfil salvo localmente. Erro no banco: {msg_bd}")

    st.markdown("---")
    st.markdown("### Privacidade e Dados")
    with st.expander("Termo de Consentimento e Política de Privacidade"):
        st.markdown(TERMO_PRIVACIDADE)
    secao_excluir_conta()

# ============================================================================
# PÁGINA MODELO CLÍNICO
# ============================================================================

elif pagina == "Modelo Clínico":
    st.markdown('<div class="main-header">Modelo Clínico (Pima Indians)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predição baseada em exames laboratoriais</div>', unsafe_allow_html=True)
    
    if st.session_state.perfil_preenchido:
        st.success(f"Usando dados do perfil: {st.session_state.perfil_nome} | Idade: {st.session_state.perfil_idade} anos | IMC: {st.session_state.perfil_imc:.1f}")
    
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        glucose = st.number_input("Glicemia TOTG 2h (mg/dL)", min_value=0, max_value=250, value=120,
                                  help="Glicemia 2 horas após ingerir glicose (teste oral de tolerância à glicose). "
                                       "Não é a glicemia de jejum.")
        insulin = st.number_input("Insulina Sérica (µU/ml)", min_value=0, max_value=900, value=80,
                                  help="Deixe 0 se não tiver esse exame.")

    with col2:
        blood_pressure = st.number_input("Pressão Diastólica (mm Hg)", min_value=0, max_value=150, value=80,
                                         help="O número de baixo da pressão. Ex.: em 120/80, informe 80.")
        skin_thickness = st.number_input("Espessura da Pele (mm)", min_value=0, max_value=100, value=20,
                                         help="Dobra cutânea do tríceps. Deixe 0 se não tiver essa medida.")

    with col3:
        bmi = st.number_input("IMC", min_value=10.0, max_value=70.0,
                             value=st.session_state.perfil_imc if st.session_state.perfil_imc > 0 else 25.0, step=0.1)
        age = st.number_input("Idade (anos)", min_value=18, max_value=120,
                             value=max(18, int(st.session_state.perfil_idade)))

    with col4:
        pregnancies = st.number_input("Número de Gestações", min_value=0, max_value=20, value=0)
        diabetes_pedigree = st.number_input("Função Pedigree de Diabetes", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    
    if st.button("Analisar Risco", key="btn_clinico", type="primary", width="stretch"):
        try:
            modelo, meta_clinico = carregar_modelo('modelo_clinico')

            probabilidade = prever_probabilidade(modelo, meta_clinico, {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': diabetes_pedigree,
                'Age': age,
            })
            predicao = 1 if probabilidade >= meta_clinico['threshold'] else 0

        except Exception as e:
            st.error(f"Não foi possível executar o modelo clínico: {e}")
            st.stop()
        
        # Salvar no banco
        try:
            supabase_db.salvar_predicao(
                user_id=auth.get_user_id(),
                tipo_modelo='clinico',
                probabilidade=float(probabilidade),
                resultado=int(predicao),
                dados_input={
                    'glucose': glucose, 'bmi': bmi, 'age': age,
                    'insulin': insulin, 'blood_pressure': blood_pressure
                },
                session_id=st.session_state.session_id,
            )
        except Exception as e:
            print(f"Erro ao salvar: {e}")
        
        st.markdown("---")
        st.markdown("## Resultado da Análise")
        col_res, col_gauge = st.columns([3, 2], vertical_alignment="center")
        with col_res:
            if predicao == 1:
                st.markdown(f"""
                <div class="result-high">
                    <h3>RISCO ELEVADO DE DIABETES</h3>
                    <p class='prob'><strong>Probabilidade: {probabilidade*100:.1f}%</strong></p>
                    <p>Baseado nos exames, o modelo identifica risco elevado.</p>
                    <p><strong>Recomendações:</strong></p>
                    <ul>
                        <li>Consulte endocrinologista</li>
                        <li>Repita exames de glicemia e HbA1c</li>
                        <li>Inicie mudanças no estilo de vida</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                    <h3>RISCO BAIXO DE DIABETES</h3>
                    <p class='prob'><strong>Probabilidade: {probabilidade*100:.1f}%</strong></p>
                    <p>Baseado nos exames, o modelo não identifica risco elevado.</p>
                    <p><strong>Recomendações:</strong></p>
                    <ul>
                        <li>Mantenha hábitos saudáveis</li>
                        <li>Faça check-ups regulares</li>
                        <li>Monitore glicemia anualmente</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        with col_gauge:
            st.image(criar_gauge_risco(probabilidade, threshold=meta_clinico['threshold']), width="stretch")

        # Fatores de Risco Detalhados
        st.markdown("---")
        st.markdown("### Fatores de Risco Identificados")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Principais Contribuições:**")
            # Limites do TOTG 2h: < 140 normal, 140-199 tolerância diminuída, >= 200 faixa de diabetes
            if glucose >= 200:
                st.error(f"Glicemia na faixa de diabetes ({glucose} mg/dL no TOTG)")
            elif glucose >= 140:
                st.warning(f"Tolerância à glicose diminuída ({glucose} mg/dL no TOTG)")
            else:
                st.success(f"Glicemia normal ({glucose} mg/dL no TOTG)")

            if bmi >= 30:
                st.error(f"Obesidade (IMC {bmi:.1f}) - Fator de risco #1")
            elif bmi >= 25:
                st.warning(f"Sobrepeso (IMC {bmi:.1f})")
            else:
                st.success(f"Peso adequado (IMC {bmi:.1f})")

        with col2:
            st.markdown("**Outros Fatores:**")
            if insulin > 200:
                st.error(f"Insulina elevada ({insulin} µU/ml)")
            elif insulin > 150:
                st.warning(f"Insulina moderada ({insulin} µU/ml)")
            else:
                st.success(f"Insulina normal ({insulin} µU/ml)")

            if age > 45:
                st.warning(f"Idade: {age} anos (fator de risco)")
            else:
                st.info(f"Idade: {age} anos")

        # Recomendações Personalizadas
        st.markdown("---")
        st.markdown("### Recomendações Personalizadas")

        recomendacoes = []

        if predicao == 1:
            st.markdown("**URGENTE - Alto Risco:**")

            if glucose >= 200:
                recomendacoes.append("**Glicemia na faixa de diabetes:** Procure um endocrinologista o quanto antes para confirmar o diagnóstico")
            elif glucose >= 140:
                recomendacoes.append("**Tolerância à glicose diminuída:** Repita o exame e peça HbA1c ao seu médico")

            if insulin > 200:
                recomendacoes.append("**Insulina elevada:** Pode indicar resistência à insulina - avalie com um endocrinologista")

            if bmi >= 30:
                recomendacoes.append("**Obesidade:** Perder 5-10% do peso reduz risco em 58% - consulte nutricionista")
            elif bmi >= 25:
                recomendacoes.append("**Sobrepeso:** Objetivo IMC < 25 - dieta + exercícios")

            if st.session_state.perfil_hist_familiar == "Sim":
                recomendacoes.append("**Histórico familiar:** Com exames alterados, faça acompanhamento médico mais frequente")

            if blood_pressure >= 90:
                recomendacoes.append("**Pressão diastólica alta:** Diabetes + hipertensão aumentam o risco cardiovascular")

            recomendacoes.append("**Protocolo Completo:** Glicemia jejum + HbA1c + Curva glicêmica + Insulina + Peptídeo C")
            recomendacoes.append("**Exercício Obrigatório:** 30 min/dia de caminhada MELHORA sensibilidade à insulina")

        else:
            st.markdown("**Risco Baixo - Mantenha a Vigilância:**")

            if glucose >= 140:
                recomendacoes.append("**Glicemia acima do normal no TOTG:** Reduza açúcar e carboidratos refinados e repita o exame")

            if insulin > 150:
                recomendacoes.append("**Insulina moderada:** Evite resistência insulínica com exercício")

            if bmi >= 25:
                recomendacoes.append("**Sobrepeso:** Objetivo IMC < 25 - dieta + exercícios")
            elif bmi > 23:
                recomendacoes.append("**Peso no limite:** IMC saudável mas perto do sobrepeso - não ganhe mais peso")
            else:
                recomendacoes.append("**Peso ideal:** Continue assim! Peso saudável é sua melhor proteção")

            if st.session_state.perfil_hist_familiar == "Sim":
                recomendacoes.append("**Histórico Familiar:** Mesmo com exames normais, monitore glicemia ANUALMENTE")

            if age < 40:
                recomendacoes.append("**Idade Jovem:** Ótimo momento para criar hábitos saudáveis que duram a vida toda")

            recomendacoes.append("**Mantenha exercício:** 150 min/semana mantém sensibilidade à insulina")
            recomendacoes.append("**Dieta preventiva:** Reduza açúcar, refrigerantes e carboidratos refinados")

        st.markdown("\n".join(f"- {rec}" for rec in recomendacoes))

        if st.session_state.perfil_alergias:
            st.warning(f"**ATENÇÃO - Alergias:** {st.session_state.perfil_alergias}\n\nInforme ao médico antes de QUALQUER medicação!")

        if predicao == 0 and glucose < 140 and bmi < 25:
            st.success("**Parabéns!** Seus exames estão ótimos. Continue com hábitos saudáveis!")

        # Geração de PDF
        st.markdown("---")
        st.markdown("### Exportar Relatório")

        try:
            pdf_path = gerar_relatorio_predicao(
                nome_arquivo=os.path.join(tempfile.gettempdir(), f"relatorio_{uuid.uuid4().hex}.pdf"),
                nome_paciente=st.session_state.perfil_nome or "Paciente",
                tipo_modelo="Clínico",
                resultado=predicao,
                probabilidade=probabilidade,
                dados_paciente={
                    'Idade': age,
                    'Sexo': st.session_state.perfil_sexo,
                    'Glicemia (mg/dL)': glucose,
                    'Insulina (µU/ml)': insulin,
                    'Pressão Arterial (mm Hg)': blood_pressure,
                    'Espessura da Pele (mm)': skin_thickness,
                    'IMC': bmi,
                    'Gestações': pregnancies,
                    'Função Pedigree': diabetes_pedigree,
                },
                # reportlab usa <b>, não markdown
                recomendacoes=[re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", r) for r in recomendacoes],
            )

            with open(pdf_path, 'rb') as f:
                st.download_button(
                    label="Baixar Relatório PDF",
                    data=f,
                    file_name=f"relatorio_diabetes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    width="stretch"
                )

            st.success("Relatório gerado! Leve ao seu médico.")

        except Exception as e:
            st.info(f"Geração de PDF: {e}")


# ============================================================================
# PÁGINA MODELO COMPORTAMENTAL  
# ============================================================================

elif pagina == "Modelo Comportamental":
    st.markdown('<div class="main-header">Modelo Comportamental (BRFSS)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predição baseada em hábitos de vida (SEM exames)</div>', unsafe_allow_html=True)
    st.caption("Estima o risco de **pré-diabetes ou diabetes**: na base usada (BRFSS 2015) os dois são agrupados, "
               "e o diagnóstico é autorreferido pelos entrevistados.")
    
    
    opcoes_saude = ["Excelente", "Muito boa", "Boa", "Razoável", "Ruim"]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        idade_brfss = st.number_input("Idade (anos)", min_value=18, max_value=120,
                                      value=int(st.session_state.perfil_idade) if st.session_state.perfil_idade >= 18 else 30)
        sexo_brfss = st.selectbox("Sexo", ["Masculino", "Feminino"],
                                  index=0 if st.session_state.perfil_sexo == "Masculino" else 1)

    with col2:
        bmi_brfss = st.number_input("IMC", min_value=10.0, max_value=70.0,
                                    value=st.session_state.perfil_imc if st.session_state.perfil_imc > 0 else 25.0)
        gen_health = st.selectbox("Saúde geral?", opcoes_saude, index=2)

    with col3:
        high_bp = st.selectbox("Pressão alta?", ["Não", "Sim"])
        high_chol = st.selectbox("Colesterol alto?", ["Não", "Sim"])

    with col4:
        phys_activity = st.selectbox("Pratica atividade física?", ["Não", "Sim"])
        smoker = st.selectbox("Fumante?", ["Não", "Sim"])

    if st.button("Analisar Risco", key="btn_comportamental", type="primary", width="stretch"):
        modelo_comp, meta_comp = carregar_modelo('modelo_comportamental')
        # BRFSS codifica idade em faixas: 1 = 18-24, 2 = 25-29, ..., 13 = 80+
        faixa_idade = 1 if idade_brfss < 25 else min(13, (idade_brfss - 25) // 5 + 2)
        entrada = {
            'HighBP': int(high_bp == "Sim"),
            'HighChol': int(high_chol == "Sim"),
            'BMI': bmi_brfss,
            'Smoker': int(smoker == "Sim"),
            'PhysActivity': int(phys_activity == "Sim"),
            'GenHlth': opcoes_saude.index(gen_health) + 1,
            'Age': faixa_idade,
            'Sex': int(sexo_brfss == "Masculino"),
        }

        probabilidade = prever_probabilidade(modelo_comp, meta_comp, entrada)
        predicao = 1 if probabilidade >= meta_comp['threshold'] else 0
        
        # Salvar
        try:
            supabase_db.salvar_predicao(
                user_id=auth.get_user_id(),
                tipo_modelo='comportamental',
                probabilidade=float(probabilidade),
                resultado=int(predicao),
                dados_input={
                    'age': idade_brfss, 'sex': sexo_brfss, 'bmi': bmi_brfss, 'gen_health': gen_health,
                    'high_bp': high_bp, 'high_chol': high_chol,
                    'phys_activity': phys_activity, 'smoker': smoker,
                },
                session_id=st.session_state.session_id,
            )
        except Exception as e:
            print(f"Erro: {e}")
        
        st.markdown("---")
        st.markdown("## Resultado")
        col_res, col_gauge = st.columns([3, 2], vertical_alignment="center")
        with col_res:
            if predicao == 1:
                st.markdown(f"""
                <div class="result-high">
                    <h3>RISCO ELEVADO</h3>
                    <p class='prob'><strong>{probabilidade*100:.1f}%</strong></p>
                    <p>Baseado nos hábitos, risco elevado identificado.</p>
                    <ul>
                        <li>Procure médico para exames</li>
                        <li>Aumente atividade física</li>
                        <li>Melhore alimentação</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                    <h3>RISCO BAIXO</h3>
                    <p class='prob'><strong>{probabilidade*100:.1f}%</strong></p>
                    <p>Baseado nos hábitos, risco baixo.</p>
                    <ul>
                        <li>Mantenha bons hábitos!</li>
                        <li>Continue exercícios</li>
                        <li>Check-ups anuais</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        with col_gauge:
            st.image(criar_gauge_risco(probabilidade, threshold=meta_comp['threshold']), width="stretch")

        # Fatores de Risco
        st.markdown("---")
        st.markdown("### Fatores de Risco Identificados")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Condições:**")
            if high_bp == "Sim":
                st.error("Pressão alta")
            else:
                st.success("Pressão normal")

            if high_chol == "Sim":
                st.error("Colesterol alto")
            else:
                st.success("Colesterol normal")

        with col2:
            st.markdown("**Hábitos:**")
            if phys_activity == "Sim":
                st.success("Ativo fisicamente")
            else:
                st.error("Sedentário")

            if smoker == "Sim":
                st.error("Fumante")
            else:
                st.success("Não fumante")

# ============================================================================
# PÁGINA COMPARAÇÃO
# ============================================================================

elif pagina == "Comparação":
    st.markdown('<div class="main-header">Comparação dos Modelos</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h3>Modelo Clínico</h3>
            <p><strong>Dataset:</strong> Pima Indians (768 registros)</p>
            <p><strong>F1-Score:</strong> 0.69</p>
            <p><strong>Precisão:</strong> 59.9%</p>
            <p><strong>Recall:</strong> 82.2%</p>
            <hr>
            <p><strong>Vantagens:</strong></p>
            <ul>
                <li>Alta precisão</li>
                <li>Features preditivas</li>
            </ul>
            <p><strong>Desvantagens:</strong></p>
            <ul>
                <li>Requer exames (R$ 50-100)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h3>Modelo Comportamental</h3>
            <p><strong>Dataset:</strong> BRFSS (253k registros)</p>
            <p><strong>F1-Score:</strong> 0.46</p>
            <p><strong>Precisão:</strong> 36.7%</p>
            <p><strong>Recall:</strong> 61.1%</p>
            <hr>
            <p><strong>Vantagens:</strong></p>
            <ul>
                <li>Sem custo (R$ 0)</li>
                <li>Triagem rápida</li>
            </ul>
            <p><strong>Desvantagens:</strong></p>
            <ul>
                <li>Menor precisão</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.success("""
    ### Conclusão
    
    **Sistema Híbrido de 2 Fases:**
    
    1. **Triagem Inicial:** Modelo Comportamental (grátis)
    2. **Diagnóstico Preciso:** Modelo Clínico (com exames)
    
    **Resultado:** Melhor custo-benefício!
    """)


# ============================================================================
# PÁGINA HISTÓRICO & ESTATÍSTICAS
# ============================================================================

elif pagina == "Histórico & Estatísticas":
    st.markdown('<div class="main-header">Histórico & Estatísticas</div>', unsafe_allow_html=True)
    
    # Obter predições do usuário
    try:
        predicoes_usuario = supabase_db.obter_ultimas_predicoes(limite=50, user_id=auth.get_user_id())
    except:
        predicoes_usuario = []
    
    if predicoes_usuario:
        st.success(f"Total de {len(predicoes_usuario)} predições realizadas")
        
        # Estatísticas básicas
        col1, col2, col3 = st.columns(3)
        
        clinico_count = len([p for p in predicoes_usuario if p.get('tipo_modelo') == 'clinico'])
        comportamental_count = len([p for p in predicoes_usuario if p.get('tipo_modelo') == 'comportamental'])
        alto_risco = len([p for p in predicoes_usuario if p.get('resultado') == 1])
        
        with col1:
            st.metric("Modelo Clínico", clinico_count)
        with col2:
            st.metric("Modelo Comportamental", comportamental_count)
        with col3:
            st.metric("Alto Risco", alto_risco)
        
        st.markdown("---")
        
        # Tabela de histórico
        st.markdown("### Últimas Predições")
        
        dados_tabela = []
        for pred in predicoes_usuario[:10]:
            dados_tabela.append({
                'Data': datetime.fromisoformat(pred['created_at'].replace('Z', '+00:00')).strftime('%d/%m/%Y %H:%M'),
                'Tipo': 'Clínico' if pred['tipo_modelo'] == 'clinico' else 'Comportamental',
                'Resultado': 'Alto Risco' if pred['resultado'] == 1 else 'Baixo Risco',
                'Probabilidade': f"{pred['probabilidade']*100:.1f}%"
            })
        
        if dados_tabela:
            df = pd.DataFrame(dados_tabela)
            st.dataframe(df, width="stretch", hide_index=True)
        
        # Análise Temporal
        if len(predicoes_usuario) >= 2:
            st.markdown("---")
            st.markdown("### Evolução do Risco")
            
            try:
                stats = calcular_estatisticas_evolucao(predicoes_usuario)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Primeira Avaliação", f"{stats['primeira_prob']*100:.1f}%")
                with col2:
                    st.metric("Última Avaliação", f"{stats['ultima_prob']*100:.1f}%")
                with col3:
                    delta = stats['diferenca_absoluta']
                    st.metric("Mudança", f"{abs(delta):.1f}%", 
                             delta=f"{delta:.1f}%")
                
                if stats['melhorou']:
                    st.success("Parabéns! Seu risco diminuiu!")
                elif stats['piorou']:
                    st.warning("Atenção! Seu risco aumentou. Consulte um médico.")
                else:
                    st.info("Seu risco está estável. Continue monitorando!")
                
                # Gráficos de evolução
                try:
                    img_evolucao = criar_grafico_evolucao(predicoes_usuario)
                    st.image(img_evolucao, width="stretch")
                except Exception as e:
                    st.warning(f"Gráfico de evolução: {e}")

                try:
                    img_progresso = criar_grafico_progresso(predicoes_usuario)
                    st.markdown("### Comparação: Antes vs Agora")
                    st.image(img_progresso, width="stretch")
                except Exception as e:
                    st.warning(f"Gráfico de progresso: {e}")

            except Exception as e:
                st.warning(f"Análise temporal indisponível: {e}")

        # Distribuição de probabilidades
        st.markdown("---")
        st.markdown("### Distribuição de Probabilidades")

        try:
            probs = [p['probabilidade'] * 100 for p in predicoes_usuario]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(probs, bins=20, color='#4fb9c4', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Probabilidade de Diabetes (%)', fontsize=12)
            ax.set_ylabel('Frequência', fontsize=12)
            ax.set_title('Distribuição das Suas Predições', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.set_facecolor('#1a1a2e')
            fig.patch.set_facecolor('#0f0f0f')
            ax.tick_params(colors='#e0e0e0')
            ax.xaxis.label.set_color('#e0e0e0')
            ax.yaxis.label.set_color('#e0e0e0')
            ax.title.set_color('#e0e0e0')

            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.info(f"Gráfico de distribuição: {e}")

        # Comparação Clínico vs Comportamental
        if clinico_count > 0 and comportamental_count > 0:
            st.markdown("---")
            st.markdown("### Comparação: Clínico vs Comportamental")

            try:
                clinico_probs = [p['probabilidade'] * 100 for p in predicoes_usuario if p.get('tipo_modelo') == 'clinico']
                comportamental_probs = [p['probabilidade'] * 100 for p in predicoes_usuario if p.get('tipo_modelo') == 'comportamental']

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Prob. Média Clínico",
                        f"{np.mean(clinico_probs):.1f}%"
                    )

                with col2:
                    st.metric(
                        "Prob. Média Comportamental",
                        f"{np.mean(comportamental_probs):.1f}%"
                    )

                with col3:
                    diff = np.mean(clinico_probs) - np.mean(comportamental_probs)
                    st.metric(
                        "Diferença",
                        f"{abs(diff):.1f}%",
                        delta=f"{diff:.1f}%"
                    )

            except Exception as e:
                st.info(f"Comparação: {e}")

    else:
        st.info("Nenhuma predição realizada ainda. Teste os modelos!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p>Autor: David Reis | 2026</p>
    <p><em>Este sistema é para fins educacionais e não substitui consulta médica.</em></p>
</div>
""", unsafe_allow_html=True)

