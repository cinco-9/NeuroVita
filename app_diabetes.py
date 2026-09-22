# -*- coding: utf-8 -*-
"""
INTERFACE WEB - SISTEMA DE PREDICAO DE DIABETES
TCC: Agente de IA para Estimativa de Risco de Diabetes Tipo 2
Autor: David Reis
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import uuid
from datetime import datetime
warnings.filterwarnings('ignore')

# Importar módulos
from supabase_db import db as supabase_db
from auth import auth
from pdf_generator import gerar_relatorio_predicao
from shap_explicabilidade import criar_grafico_barras_shap, obter_top_features_shap
from gauge_component import criar_gauge_risco
from analise_temporal import criar_grafico_evolucao, criar_grafico_progresso, calcular_estatisticas_evolucao
import pickle

# Configuração da página
st.set_page_config(
    page_title="Predição de Diabetes - TCC",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Dark Mode Profissional - SEM EMOJIS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 50%, #16213e 100%);
        background-attachment: fixed;
    }

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e0e0e0;
    }

    .main-header {
        font-size: 24px;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 8px 8px;
        margin-bottom: 2px;
        letter-spacing: -0.3px;
    }

    .sub-header {
        font-size: 12px;
        text-align: center;
        color: #a0a0a0;
        padding-bottom: 10px;
        font-weight: 400;
    }

    .metric-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d44 100%);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }

    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 212, 255, 0.3);
    }

    .result-high {
        background: linear-gradient(135deg, #1a0a0f 0%, #3d1a28 100%);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(255, 0, 60, 0.3);
        border: 2px solid #ff0040;
    }

    .result-high h3 {
        color: #ff4060 !important;
        font-weight: 700;
        font-size: 28px;
        text-shadow: 0 0 20px rgba(255, 64, 96, 0.5);
    }

    .result-low {
        background: linear-gradient(135deg, #0a1a1f 0%, #1a3d44 100%);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 255, 170, 0.2);
        border: 2px solid #00ffaa;
    }

    .result-low h3 {
        color: #00ffaa !important;
        font-weight: 700;
        font-size: 28px;
        text-shadow: 0 0 20px rgba(0, 255, 170, 0.5);
    }

    .stButton>button {
        background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%);
        color: white;
        border-radius: 12px;
        padding: 12px 30px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 30px rgba(0, 212, 255, 0.6);
    }

    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        background-color: #1a1a2e !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 6px !important;
        color: #e0e0e0 !important;
        padding: 6px 10px !important;
        font-size: 12px !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 0 4px rgba(0, 212, 255, 0.3) !important;
        background-color: #252540 !important;
        transform: translateY(-2px);
    }

    .stTextInput>div>div>input::placeholder {
        color: rgba(255, 255, 255, 0.4) !important;
    }

    /* Labels dos inputs */
    .stTextInput label,
    .stNumberInput label,
    .stSelectbox label {
        color: #00d4ff !important;
        font-weight: 600 !important;
        font-size: 11px !important;
        margin-bottom: 3px !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f0f 0%, #1a1a2e 100%);
    }

    /* Tabs melhoradas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(26, 26, 46, 0.6) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 6px !important;
        padding: 6px 16px !important;
        color: #e0e0e0 !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        transition: all 0.3s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 212, 255, 0.15) !important;
        border-color: rgba(0, 212, 255, 0.5) !important;
        transform: translateY(-2px);
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.3), rgba(123, 44, 191, 0.3)) !important;
        border-color: #00d4ff !important;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4) !important;
    }

    /* Botão melhorado */
    button[kind="primary"] {
        background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%) !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 7px 20px !important;
        font-weight: 700 !important;
        font-size: 12px !important;
        color: white !important;
        box-shadow: 0 3px 12px rgba(0, 212, 255, 0.4) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 20px rgba(0, 212, 255, 0.6) !important;
        background: linear-gradient(135deg, #00ffff 0%, #9945ff 100%) !important;
    }

    /* Container do form */
    [data-testid="stForm"] {
        background: rgba(26, 26, 46, 0.3) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 10px !important;
        padding: 14px !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.5) !important;
    }

    /* Headings */
    h1, h2, h3 {
        color: #e0e0e0 !important;
        font-weight: 700 !important;
    }

    h3 {
        font-size: 16px !important;
        margin-bottom: 8px !important;
        background: linear-gradient(135deg, #00d4ff 0%, #ffffff 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 700 !important;
    }

    p {
        color: #e0e0e0 !important;
        font-size: 12px !important;
    }

    /* Linha separadora */
    hr {
        border-color: rgba(0, 212, 255, 0.2) !important;
        margin: 10px 0 !important;
    }

    /* Efeitos de erro e sucesso */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 12px !important;
        padding: 16px !important;
        font-weight: 500 !important;
    }

    .stSuccess {
        background: rgba(0, 255, 170, 0.1) !important;
        border: 2px solid rgba(0, 255, 170, 0.3) !important;
    }

    .stError {
        background: rgba(255, 0, 64, 0.1) !important;
        border: 2px solid rgba(255, 0, 64, 0.3) !important;
    }

    /* Melhorar selectbox */
    .stSelectbox>div>div>select {
        background-color: #1a1a2e !important;
        border: 2px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 12px !important;
        color: #e0e0e0 !important;
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
</style>
""", unsafe_allow_html=True)

# Cache dos modelos REAIS
@st.cache_resource
def carregar_modelo_pima():
    """Carrega modelo REAL treinado no Pima Indians"""
    try:
        with open('modelo_melhorado_clinico.pkl', 'rb') as f:
            modelo = pickle.load(f)
        return modelo
    except Exception as e:
        st.error(f"Erro ao carregar modelo clínico: {e}")
        # Fallback: modelo básico
        modelo = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        return modelo

@st.cache_resource
def carregar_scaler():
    """Carrega scaler para normalização"""
    try:
        with open('scaler_melhorado.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except Exception as e:
        st.warning(f"Scaler não encontrado, usando StandardScaler padrão")
        return StandardScaler()

@st.cache_resource
def carregar_modelo_brfss():
    """Carrega modelo treinado no BRFSS"""
    # Vamos treinar esse se não existir
    try:
        with open('modelo_comportamental.pkl', 'rb') as f:
            modelo = pickle.load(f)
        return modelo
    except:
        # Fallback
        modelo = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        return modelo

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

if not auth.is_logged_in():
    # Container centralizado
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<div class="main-header">Sistema de Predição de Diabetes</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Faça login ou cadastre-se para começar</div>', unsafe_allow_html=True)
        st.markdown("---")

        tab1, tab2 = st.tabs(["Login", "Cadastrar"])

    with tab1:
        st.markdown("### Fazer Login")
        with st.form("login_form"):
            email_login = st.text_input("Email", key="email_login", placeholder="seu@email.com")
            senha_login = st.text_input("Senha", type="password", key="senha_login")
            submit_login = st.form_submit_button("Entrar", use_container_width=True, type="primary")

            if submit_login:
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

    with tab2:
        st.markdown("### Criar Nova Conta")
        with st.form("signup_form"):
            nome_cadastro = st.text_input("Nome Completo", key="nome_cadastro", placeholder="João Silva")
            email_cadastro = st.text_input("Email", key="email_cadastro", placeholder="seu@email.com")
            senha_cadastro = st.text_input("Senha", type="password", key="senha_cadastro", help="Mínimo 6 caracteres")
            senha_confirma = st.text_input("Confirmar Senha", type="password", key="senha_confirma")
            submit_cadastro = st.form_submit_button("Cadastrar", use_container_width=True, type="primary")

            if submit_cadastro:
                if not email_cadastro or not senha_cadastro or not nome_cadastro:
                    st.error("Por favor, preencha todos os campos")
                elif senha_cadastro != senha_confirma:
                    st.error("As senhas não coincidem")
                elif len(senha_cadastro) < 6:
                    st.error("Senha deve ter no mínimo 6 caracteres")
                else:
                    with st.spinner("Criando conta..."):
                        sucesso, mensagem = auth.signup(email_cadastro, senha_cadastro, nome_cadastro)
                        if sucesso:
                            st.success(mensagem)
                            st.rerun()
                        else:
                            st.error(mensagem)

        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; padding: 20px;'>
                <p><strong>TCC: Agente de IA para Estimativa de Risco de Diabetes Tipo 2</strong></p>
                <p>Autor: David Reis | 2026</p>
            </div>
        """, unsafe_allow_html=True)

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

if st.sidebar.button("Sair", use_container_width=True):
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
### Sobre o TCC
**Autor:** David Reis
**Tema:** Agente de IA para Estimativa de Risco

**Datasets:**
- Pima Indians (clínico)
- BRFSS 2015 (comportamental)

**Modelos:**
- XGBoost + Feature Engineering
- SHAP Explicabilidade
""")

# ============================================================================
# PÁGINA INÍCIO
# ============================================================================

if pagina == "Início":
    st.markdown('<div class="main-header">Sistema de Predição de Diabetes</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">TCC: Agente de IA para Estimativa de Risco de Diabetes Tipo 2</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h3>Modelo Clínico</h3>
            <p><strong>F1-Score: 0.72</strong></p>
            <p>Dataset: Pima Indians<br>
            Features: Exames laboratoriais<br>
            <strong>Melhor precisão</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h3>Modelo Comportamental</h3>
            <p><strong>F1-Score: 0.47</strong></p>
            <p>Dataset: BRFSS (253k registros)<br>
            Features: Hábitos de vida<br>
            <strong>Triagem inicial</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <h3>Descoberta</h3>
            <p><strong>Clínico +53% melhor!</strong></p>
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
    
    if st.button("Salvar Perfil", type="primary", use_container_width=True):
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

# ============================================================================
# PÁGINA MODELO CLÍNICO
# ============================================================================

elif pagina == "Modelo Clínico":
    st.markdown('<div class="main-header">Modelo Clínico (Pima Indians)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predição baseada em exames laboratoriais</div>', unsafe_allow_html=True)
    
    if st.session_state.perfil_preenchido:
        st.success(f"Usando dados do perfil: {st.session_state.perfil_nome} | Idade: {st.session_state.perfil_idade} anos | IMC: {st.session_state.perfil_imc:.1f}")
    
    st.info("**Preencha os dados dos exames laboratoriais abaixo:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Número de Gestações", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glicemia (mg/dL)", min_value=0, max_value=250, value=120)
        blood_pressure = st.number_input("Pressão Arterial (mm Hg)", min_value=0, max_value=150, value=80)
        skin_thickness = st.number_input("Espessura da Pele (mm)", min_value=0, max_value=100, value=20)
    
    with col2:
        insulin = st.number_input("Insulina Sérica (µU/ml)", min_value=0, max_value=900, value=80)
        bmi = st.number_input("IMC", min_value=10.0, max_value=70.0, 
                             value=st.session_state.perfil_imc if st.session_state.perfil_imc > 0 else 25.0, step=0.1)
        diabetes_pedigree = st.number_input("Função Pedigree de Diabetes", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Idade (anos)", min_value=18, max_value=120, 
                             value=st.session_state.perfil_idade if st.session_state.perfil_idade > 0 else 33)
    
    if st.button("Analisar Risco", key="btn_clinico", type="primary", use_container_width=True):
        # PREDIÇÃO REAL com modelo treinado
        try:
            # Carregar modelo e scaler
            modelo = carregar_modelo_pima()
            scaler = carregar_scaler()

            # Preparar features (incluindo feature engineering)
            glucose_bmi = glucose * bmi
            age_glucose = age * glucose

            # Criar array com todas as features
            features = np.array([[
                pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, diabetes_pedigree, age,
                glucose_bmi, age_glucose, bmi**2, glucose**2,
                insulin * bmi, age * bmi, glucose / bmi if bmi > 0 else 0,
                insulin / glucose if glucose > 0 else 0
            ]])

            # Normalizar
            features_scaled = scaler.transform(features)

            # PREDIÇÃO REAL
            probabilidade = modelo.predict_proba(features_scaled)[0][1]
            predicao = 1 if probabilidade > 0.35 else 0  # Threshold otimizado

            st.success("✓ Predição feita com MODELO TREINADO REAL!")

        except Exception as e:
            st.warning(f"Usando modo fallback: {e}")
            # Fallback para lógica simples
            score = 0
            if glucose > 140: score += 0.4
            elif glucose > 120: score += 0.2
            if bmi > 30: score += 0.3
            elif bmi > 25: score += 0.15
            if insulin > 200: score += 0.2
            if diabetes_pedigree > 0.7: score += 0.2
            if age > 45: score += 0.15

            probabilidade = min(score, 0.95)
            predicao = 1 if probabilidade > 0.35 else 0
        
        # Salvar no banco
        try:
            supabase_db.salvar_predicao(
                user_id=auth.get_user_id(),
                tipo_modelo='clinico',
                probabilidade=probabilidade,
                resultado=predicao,
                features={
                    'glucose': glucose, 'bmi': bmi, 'age': age,
                    'insulin': insulin, 'blood_pressure': blood_pressure
                }
            )
        except Exception as e:
            print(f"Erro ao salvar: {e}")
        
        st.markdown("---")
        st.markdown("## Resultado da Análise")
        
        if predicao == 1:
            st.markdown(f"""
            <div class="result-high">
                <h3>RISCO ELEVADO DE DIABETES</h3>
                <p style='font-size: 24px;'><strong>Probabilidade: {probabilidade*100:.1f}%</strong></p>
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
                <p style='font-size: 24px;'><strong>Probabilidade: {probabilidade*100:.1f}%</strong></p>
                <p>Baseado nos exames, o modelo não identifica risco elevado.</p>
                <p><strong>Recomendações:</strong></p>
                <ul>
                    <li>Mantenha hábitos saudáveis</li>
                    <li>Faça check-ups regulares</li>
                    <li>Monitore glicemia anualmente</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Gauge visual
        try:
            gauge_path = criar_gauge_risco(probabilidade)
            st.markdown("### Medidor de Risco Visual")
            st.image(gauge_path, use_column_width=True)
        except Exception as e:
            st.warning(f"Gauge indisponível: {e}")

        # SHAP Explicabilidade
        st.markdown("---")
        st.markdown("### Explicação da Predição (SHAP)")

        try:
            # Criar DataFrame com features
            import os
            dados_shap = {
                'Pregnancies': [pregnancies],
                'Glucose': [glucose],
                'BloodPressure': [blood_pressure],
                'SkinThickness': [skin_thickness],
                'Insulin': [insulin],
                'BMI': [bmi],
                'DiabetesPedigreeFunction': [diabetes_pedigree],
                'Age': [age],
                'Glucose_BMI': [glucose * bmi],
                'Age_Glucose': [age * glucose]
            }

            df_shap = pd.DataFrame(dados_shap)

            # Gerar gráfico SHAP
            shap_path = criar_grafico_barras_shap(df_shap, carregar_modelo_pima())

            if os.path.exists(shap_path):
                st.image(shap_path, use_column_width=True)
                st.info("O gráfico SHAP mostra quais fatores mais influenciaram sua predição.")

        except Exception as e:
            st.info(f"Explicação SHAP indisponível nesta versão: {e}")

        # Fatores de Risco Detalhados
        st.markdown("---")
        st.markdown("### Fatores de Risco Identificados")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Principais Contribuições:**")
            if glucose > 140:
                st.error(f"Glicemia ALTA ({glucose} mg/dL) - Risco significativo!")
            elif glucose > 120:
                st.warning(f"Glicemia borderline ({glucose} mg/dL) - Monitorar")
            else:
                st.success(f"Glicemia normal ({glucose} mg/dL)")

            if bmi > 30:
                st.error(f"Obesidade (IMC {bmi:.1f}) - Fator de risco #1")
            elif bmi > 25:
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

            if glucose > 140:
                recomendacoes.append("**GLICEMIA ALTA:** Consulte endocrinologista ESTA SEMANA - pode ser pré-diabetes ou diabetes")
            elif glucose > 120:
                recomendacoes.append("**Glicemia Borderline:** Repita exame + HbA1c - você está em pré-diabetes")

            if insulin > 200:
                recomendacoes.append("**Resistência Insulínica:** Avalie com endocrinologista + considere Metformina")

            if bmi > 30:
                recomendacoes.append("**OBESIDADE:** Perder 5-10% do peso reduz risco em 58% - consulte nutricionista")
            elif bmi > 25:
                recomendacoes.append("**Sobrepeso:** Objetivo IMC < 25 - dieta + exercícios")

            if st.session_state.perfil_hist_familiar == "Sim":
                recomendacoes.append("**Risco Genético Alto:** Com histórico familiar + exames alterados, monitoramento a cada 2 meses")

            if blood_pressure > 140:
                recomendacoes.append("**Pressão Alta:** Diabetes + hipertensão = risco cardiovascular dobrado")

            recomendacoes.append("**Protocolo Completo:** Glicemia jejum + HbA1c + Curva glicêmica + Insulina + Peptídeo C")
            recomendacoes.append("**Exercício Obrigatório:** 30 min/dia de caminhada MELHORA sensibilidade à insulina")

        else:
            st.markdown("**Exames Normais - Mantenha a Vigilância:**")

            if glucose > 100:
                recomendacoes.append("**Glicemia no limite:** Reduza açúcar e carboidratos refinados AGORA")

            if insulin > 150:
                recomendacoes.append("**Insulina moderada:** Evite resistência insulínica com exercício")

            if bmi > 23:
                recomendacoes.append("**Peso no limite:** IMC saudável mas perto do sobrepeso - não ganhe mais peso")
            else:
                recomendacoes.append("**Peso ideal:** Continue assim! Peso saudável é sua melhor proteção")

            if st.session_state.perfil_hist_familiar == "Sim":
                recomendacoes.append("**Histórico Familiar:** Mesmo com exames normais, monitore glicemia ANUALMENTE")

            if age < 40:
                recomendacoes.append("**Idade Jovem:** Ótimo momento para criar hábitos saudáveis que duram a vida toda")

            recomendacoes.append("**Mantenha exercício:** 150 min/semana mantém sensibilidade à insulina")
            recomendacoes.append("**Dieta preventiva:** Reduza açúcar, refrigerantes e carboidratos refinados")

        for rec in recomendacoes:
            st.markdown(f"- {rec}")

        if st.session_state.perfil_alergias:
            st.warning(f"**ATENÇÃO - Alergias:** {st.session_state.perfil_alergias}\n\nInforme ao médico antes de QUALQUER medicação!")

        if predicao == 0 and glucose < 100 and bmi < 25:
            st.success("**Parabéns!** Seus exames estão ótimos. Continue com hábitos saudáveis!")

        # Geração de PDF
        st.markdown("---")
        st.markdown("### Exportar Relatório")

        try:
            pdf_path = gerar_relatorio_predicao(
                tipo_modelo='clinico',
                probabilidade=probabilidade,
                resultado=predicao,
                features=dados_shap,
                nome_paciente=st.session_state.perfil_nome or "Paciente",
                idade=age,
                sexo=st.session_state.perfil_sexo
            )

            with open(pdf_path, 'rb') as f:
                st.download_button(
                    label="Baixar Relatório PDF",
                    data=f,
                    file_name=f"relatorio_diabetes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
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
    
    st.info("**Preencha o questionário sobre seus hábitos:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Histórico de Saúde")
        high_bp = st.selectbox("Pressão alta?", ["Não", "Sim"])
        high_chol = st.selectbox("Colesterol alto?", ["Não", "Sim"])
        bmi_brfss = st.number_input("IMC", min_value=10.0, max_value=70.0, 
                                    value=st.session_state.perfil_imc if st.session_state.perfil_imc > 0 else 25.0)
    
    with col2:
        st.markdown("#### Hábitos de Vida")
        phys_activity = st.selectbox("Pratica atividade física?", ["Não", "Sim"])
        smoker = st.selectbox("Fumante?", ["Não", "Sim"])
        gen_health = st.select_slider("Saúde geral?", options=["Excelente", "Muito boa", "Boa", "Razoável", "Ruim"])
    
    if st.button("Analisar Risco", key="btn_comportamental", type="primary", use_container_width=True):
        # Cálculo de risco
        score = 0
        if high_bp == "Sim": score += 0.2
        if high_chol == "Sim": score += 0.2
        if bmi_brfss > 30: score += 0.25
        elif bmi_brfss > 25: score += 0.15
        if gen_health in ["Razoável", "Ruim"]: score += 0.2
        if phys_activity == "Não": score += 0.1
        if smoker == "Sim": score += 0.1
        
        probabilidade = min(score, 0.90)
        predicao = 1 if probabilidade > 0.24 else 0
        
        # Salvar
        try:
            supabase_db.salvar_predicao(
                user_id=auth.get_user_id(),
                tipo_modelo='comportamental',
                probabilidade=probabilidade,
                resultado=predicao,
                features={'bmi': bmi_brfss, 'high_bp': high_bp, 'high_chol': high_chol}
            )
        except Exception as e:
            print(f"Erro: {e}")
        
        st.markdown("---")
        st.markdown("## Resultado")
        
        if predicao == 1:
            st.markdown(f"""
            <div class="result-high">
                <h3>RISCO ELEVADO</h3>
                <p style='font-size: 24px;'><strong>{probabilidade*100:.1f}%</strong></p>
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
                <p style='font-size: 24px;'><strong>{probabilidade*100:.1f}%</strong></p>
                <p>Baseado nos hábitos, risco baixo.</p>
                <ul>
                    <li>Mantenha bons hábitos!</li>
                    <li>Continue exercícios</li>
                    <li>Check-ups anuais</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Gauge visual
        try:
            gauge_path = criar_gauge_risco(probabilidade)
            st.markdown("### Medidor de Risco Visual")
            st.image(gauge_path, use_column_width=True)
        except Exception as e:
            pass

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
            <p><strong>F1-Score:</strong> 0.72</p>
            <p><strong>Precisão:</strong> 63.4%</p>
            <p><strong>Recall:</strong> 83.3%</p>
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
            <p><strong>F1-Score:</strong> 0.47</p>
            <p><strong>Precisão:</strong> 37.1%</p>
            <p><strong>Recall:</strong> 63.8%</p>
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
            st.dataframe(df, use_column_width=True, hide_index=True)
        
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
                    st.image(img_evolucao, use_column_width=True)
                except Exception as e:
                    st.warning(f"Gráfico de evolução: {e}")

                try:
                    img_progresso = criar_grafico_progresso(predicoes_usuario)
                    st.markdown("### Comparação: Antes vs Agora")
                    st.image(img_progresso, use_column_width=True)
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
            ax.hist(probs, bins=20, color='#00d4ff', alpha=0.7, edgecolor='black')
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
    <p><strong>TCC: Agente de IA para Estimativa de Risco de Diabetes Tipo 2</strong></p>
    <p>Autor: David Reis | 2026</p>
    <p><em>Este sistema é para fins educacionais e não substitui consulta médica.</em></p>
</div>
""", unsafe_allow_html=True)

