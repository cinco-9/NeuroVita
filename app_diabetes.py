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
import diario
import medico
from medico import UFS
from pdf_generator import gerar_relatorio_predicao
# from shap_explicabilidade import criar_grafico_barras_shap, obter_top_features_shap  # Desabilitado - muito pesado
from analise_temporal import criar_grafico_evolucao, criar_grafico_progresso, calcular_estatisticas_evolucao
import json
import html

# Configuração da página
st.set_page_config(
    page_title="Predição de Diabetes",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema escuro inspirado no Atlassian Design System: superfícies em camadas, bordas discretas,
# raio pequeno, tipografia Inter sem caixa alta, cor de destaque só em ações e estados ativos
st.markdown("""
<style>
    :root {
        --bg: #1D2125;
        --sidebar: #161A1D;
        --surface: #22272B;
        --surface-2: #2C333A;
        --border: rgba(166, 197, 226, 0.16);
        --border-input: rgba(166, 197, 226, 0.24);
        --text: #B6C2CF;
        --heading: #DEE4EA;
        --subtle: #9FADBC;
        --muted: #8C9BAB;
        --accent: #4fb9c4;
        --accent-hover: #6cc8d1;
        --accent-soft: rgba(79, 185, 196, 0.14);
        --on-accent: #0e1f22;
        --success-bg: #164B35;
        --success: #7EE2B8;
        --warning-bg: #533F04;
        --warning: #F5CD47;
        --danger-bg: #5D1F1A;
        --danger: #FD9891;
        --info-bg: #1C2B41;
        --info: #85B8FF;
    }

    .stApp { background: var(--bg); }

    .block-container {
        max-width: 1120px !important;
        padding-top: 3.5rem !important;  /* abaixo da barra do topo do Streamlit */
        padding-bottom: 1.5rem !important;
    }

    [data-testid="stVerticalBlock"] { gap: 0.75rem !important; }

    h1, h2, h3, h4, h5 { color: var(--heading) !important; font-weight: 600 !important; letter-spacing: 0 !important; }
    h2 { font-size: 20px !important; }
    h3 { font-size: 16px !important; padding: 8px 0 0 !important; }
    h4, h5 { font-size: 14px !important; }
    p, li { font-size: 14px !important; }
    p { margin-bottom: 4px !important; }

    /* O Streamlit compensa a margem de 16px dos parágrafos com -16px no container; como a margem foi reduzida, zera a compensação */
    [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
    [data-testid="stAlertContainer"] p { margin-bottom: 0 !important; }

    hr { border-color: var(--border) !important; margin: 8px 0 !important; }

    /* ---------- Cabeçalho das páginas ---------- */
    .page-header { display: flex; flex-direction: column; gap: 4px; padding-bottom: 8px; }
    .page-header .eyebrow { font-size: 12px; color: var(--muted); }
    .page-header .title { font-size: 24px; line-height: 32px; font-weight: 600; color: var(--heading); }
    .page-header .subtitle { color: var(--subtle); font-size: 14px; }

    /* ---------- Cards, etiquetas e listas ---------- */
    .card {
        background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
        padding: 20px; display: flex; flex-direction: column; gap: 10px; height: 100%; box-sizing: border-box;
    }
    .card-title { font-weight: 600; color: var(--heading); }
    .card-row { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
    .card-big { font-size: 32px; line-height: 40px; font-weight: 600; color: var(--heading); }
    .card-note { font-size: 12px; color: var(--muted); }
    .card-text { color: var(--text); }

    /* Containers com key "card_*" viram cards (aceitam botões dentro, ao contrário do HTML puro) */
    [class*="st-key-card_"] {
        background: var(--surface); border: 1px solid var(--border) !important; border-radius: 8px !important;
        padding: 20px !important;
    }
    [class*="st-key-card_"] [data-testid="stVerticalBlock"] { gap: 10px !important; }
    [class*="st-key-card_"] .stButton:last-child { margin-top: auto; }

    /* Botão terciário fora do menu funciona como link */
    [data-testid="stMain"] .stButton > button[kind="tertiary"] {
        color: var(--accent) !important; padding: 0 !important; min-height: 0; font-size: 13px;
    }
    [data-testid="stMain"] .stButton > button[kind="tertiary"] * { color: var(--accent) !important; }
    [data-testid="stMain"] .stButton > button[kind="tertiary"]:hover { text-decoration: underline; }

    /* Réguas coloridas (IMC, risco): o valor aparece como um marcador sobre as faixas */
    .escala { position: relative; height: 10px; margin: 14px 0 4px; }
    .escala .faixas { position: absolute; inset: 0; border-radius: 5px; overflow: hidden; }
    .escala .faixas div { position: absolute; top: 0; bottom: 0; opacity: 0.85; }
    .escala .marcador {
        position: absolute; top: 50%; width: 16px; height: 16px; border-radius: 50%; transform: translate(-50%, -50%);
        background: var(--heading); border: 3px solid var(--bg); box-shadow: 0 0 0 1px var(--heading);
    }
    .escala-rotulos { position: relative; height: 14px; font-size: 11px; color: var(--muted); }
    .escala-rotulos span { position: absolute; transform: translateX(-50%); }
    .escala-legenda { display: flex; justify-content: space-between; gap: 8px; font-size: 12px; color: var(--muted); }
    .escala-legenda i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
    .significado {
        background: var(--bg); border-radius: 6px; padding: 12px 14px; color: var(--text); font-size: 14px; line-height: 21px;
    }
    .significado b { color: var(--heading); }

    .estado {
        width: 24px; height: 24px; flex: none; border-radius: 50%; display: flex; align-items: center;
        justify-content: center; font-size: 13px; font-weight: 700;
    }
    .estado.success { background: var(--success-bg); color: var(--success); }
    .estado.warning { background: var(--warning-bg); color: var(--warning); }
    .estado.danger { background: var(--danger-bg); color: var(--danger); }
    .estado.info { background: var(--info-bg); color: var(--info); }
    .estado.neutral { background: var(--surface-2); color: var(--muted); }

    /* Card do próximo passo: a ação recomendada fica em destaque */
    .st-key-card_passo { background: var(--accent-soft) !important; border-color: rgba(79, 185, 196, 0.45) !important; }

    .progresso { height: 4px; border-radius: 2px; background: var(--surface-2); overflow: hidden; margin: 2px 0 4px; }
    .progresso div { height: 100%; background: var(--success); }

    .tecnico { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
    .tecnico div { background: var(--bg); border-radius: 6px; padding: 10px 12px; display: flex; flex-direction: column; gap: 2px; }
    .tecnico span { font-size: 11px; color: var(--muted); }
    .tecnico b { color: var(--heading); font-size: 15px; font-variant-numeric: tabular-nums; }
    .tabela code { font-size: 12px; color: var(--subtle); background: transparent; }

    .lozenge {
        display: inline-block; font-size: 11px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.04em; padding: 2px 6px; border-radius: 3px; white-space: nowrap;
    }
    .lozenge.success { background: var(--success-bg); color: var(--success); }
    .lozenge.warning { background: var(--warning-bg); color: var(--warning); }
    .lozenge.danger { background: var(--danger-bg); color: var(--danger); }
    .lozenge.info { background: var(--info-bg); color: var(--info); }
    .lozenge.neutral { background: var(--surface-2); color: var(--text); }

    .fatores { display: flex; flex-direction: column; }
    .fator {
        display: flex; align-items: center; gap: 12px;
        padding: 10px 0; border-bottom: 1px solid var(--border);
    }
    .fator:last-child { border-bottom: none; }
    .fator .nome { color: var(--heading); }
    .fator .detalhe { font-size: 12px; color: var(--muted); }

    .tabela { width: 100%; border-collapse: collapse; font-size: 14px; }
    .tabela th {
        text-align: left; font-size: 12px; font-weight: 600; color: var(--subtle);
        padding: 10px 16px; border-bottom: 1px solid var(--border);
    }
    .tabela td { padding: 12px 16px; border-bottom: 1px solid var(--border); color: var(--text); }
    .tabela tr:last-child td { border-bottom: none; }
    /* O Streamlit desenha bordas em todas as células de tabelas no markdown */
    .tabela th, .tabela td { border-left: none !important; border-right: none !important; border-top: none !important; }
    .tabela-card table { margin: 0 !important; }

    /* Ícone de link que o Streamlit põe ao lado dos títulos */
    [data-testid="stHeaderActionElements"] { display: none !important; }
    .tabela-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow-x: auto; }

    /* ---------- Resultado da avaliação ---------- */
    .risco-prob { font-size: 40px; line-height: 48px; font-weight: 600; color: var(--heading); }
    .risco-barra { position: relative; height: 8px; border-radius: 4px; background: var(--surface-2); margin: 18px 0 6px; }
    .risco-barra .preenchido { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; }
    .risco-barra .limite { position: absolute; top: -6px; bottom: -6px; width: 2px; background: var(--heading); }
    .risco-legenda { position: relative; height: 16px; font-size: 12px; color: var(--muted); }
    .risco-legenda span { position: absolute; transform: translateX(-50%); white-space: nowrap; }

    /* ---------- Botões ---------- */
    .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
        border-radius: 6px !important; font-weight: 500 !important; min-height: 38px;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--accent); color: var(--heading); }

    button[kind="primary"], button[kind="primaryFormSubmit"] {
        background: var(--accent) !important; border: none !important;
        color: var(--on-accent) !important; font-weight: 600 !important;
    }
    button[kind="primary"] *, button[kind="primaryFormSubmit"] * { color: var(--on-accent) !important; }
    button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover { background: var(--accent-hover) !important; }

    /* ---------- Campos ---------- */
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stTextArea label,
    .stDateInput label, .stRadio > label, .stMultiSelect label {
        color: var(--subtle) !important; font-weight: 600 !important; font-size: 12px !important;
    }
    input::placeholder, textarea::placeholder { color: var(--muted) !important; opacity: 1 !important; }

    [data-testid="stForm"] {
        background: var(--surface) !important; border: 1px solid var(--border) !important;
        border-radius: 8px !important; padding: 20px !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) { border-radius: 8px; }

    /* ---------- Abas ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] { padding: 10px 12px !important; color: var(--subtle) !important; font-weight: 500 !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: var(--accent) !important; font-weight: 600 !important; }

    /* ---------- Avisos ---------- */
    [data-testid="stAlertContainer"] { border-radius: 6px !important; }

    /* ---------- Menu lateral ---------- */
    [data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--border); }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 2px !important; }

    .marca { display: flex; align-items: center; gap: 10px; padding: 0 8px 16px; }
    .marca .logo {
        width: 34px; height: 34px; border-radius: 10px; background: rgba(90, 169, 255, 0.14);
        display: flex; align-items: center; justify-content: center; flex: none;
    }
    .marca .nome { font-weight: 600; color: var(--heading); font-size: 14px; line-height: 18px; }
    .marca .desc { font-size: 12px; color: var(--muted); }

    .grupo-menu {
        font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
        color: var(--muted); padding: 14px 12px 4px;
    }

    /* Itens do menu: botões terciários alinhados à esquerda; o ativo é primário com fundo suave */
    [data-testid="stSidebar"] .stButton > button {
        justify-content: flex-start; min-height: 36px; padding: 6px 12px; border: none !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="tertiary"] { color: var(--text) !important; }
    [data-testid="stSidebar"] .stButton > button[kind="tertiary"]:hover { background: var(--surface-2) !important; color: var(--heading) !important; }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: var(--accent-soft) !important; color: var(--accent) !important;
        box-shadow: inset 3px 0 0 var(--accent); border-radius: 4px 8px 8px 4px !important; font-weight: 600 !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] * { color: var(--accent) !important; }
    [data-testid="stSidebar"] .stButton > button > div { justify-content: flex-start; }

    .perfil-card { display: flex; flex-direction: column; gap: 12px; padding: 4px 4px 8px; }
    .perfil-topo { display: flex; align-items: center; gap: 10px; }
    .avatar {
        width: 40px; height: 40px; flex: none; border-radius: 50%; background: var(--accent-soft);
        color: var(--accent); display: flex; align-items: center; justify-content: center; font-weight: 600;
    }
    .perfil-nome { color: var(--heading); font-weight: 600; line-height: 18px; }
    .perfil-tipo { font-size: 12px; color: var(--muted); }
    .perfil-resumo { font-size: 12px; color: var(--subtle); }
    .perfil-imc { background: var(--bg); border-radius: 6px; padding: 10px 12px 8px; }
    .perfil-imc .rotulo { font-size: 12px; font-weight: 600; color: var(--subtle); }
    .perfil-imc .valor { color: var(--heading); font-weight: 600; font-size: 14px; }
    .perfil-frase { font-size: 12px; color: var(--text); line-height: 17px; padding-top: 4px; }

    /* Botões do card de perfil voltam a ter borda */
    [data-testid="stSidebar"] .st-key-perfil_editar button,
    [data-testid="stSidebar"] .st-key-perfil_sair button {
        justify-content: center; border: 1px solid var(--border) !important; min-height: 32px;
    }
    [data-testid="stSidebar"] .st-key-perfil_editar button > div,
    [data-testid="stSidebar"] .st-key-perfil_sair button > div { justify-content: center; }

    .aviso-rodape { font-size: 12px; color: var(--muted); padding: 12px 8px 0; }

    /* ---------- Login ---------- */
    .login-marca { display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 8px 0 16px; text-align: center; }
    .login-marca .logo {
        width: 52px; height: 52px; border-radius: 14px; background: rgba(90, 169, 255, 0.14);
        display: flex; align-items: center; justify-content: center;
    }
    .login-marca .title { font-size: 22px; font-weight: 600; color: var(--heading); }
    .login-marca .subtitle { font-size: 14px; color: var(--subtle); }
    .auth-footer { text-align: center; margin-top: 12px; font-size: 12px; color: var(--muted); }
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


# ============================================================================
# COMPONENTES VISUAIS (HTML com as classes do CSS acima)
# ============================================================================

# Círculo azul: símbolo mundial da diabetes (International Diabetes Federation)
ICONE_LOGO = ('<svg width="{t}" height="{t}" viewBox="0 0 24 24" fill="none" aria-hidden="true">'
              '<circle cx="12" cy="12" r="7.5" stroke="#5AA9FF" stroke-width="3.2"></circle></svg>')

# Ícone redondo ao lado de cada fator, para o estado ser entendido sem ler a etiqueta
SIMBOLO_ESTADO = {"success": "✓", "warning": "!", "danger": "!", "info": "i", "neutral": "–"}


def cabecalho(titulo, subtitulo="", eyebrow=""):
    partes = [f'<span class="eyebrow">{eyebrow}</span>' if eyebrow else "",
              f'<span class="title">{titulo}</span>',
              f'<span class="subtitle">{subtitulo}</span>' if subtitulo else ""]
    st.markdown(f'<div class="page-header">{"".join(partes)}</div>', unsafe_allow_html=True)


def lozenge(texto, tipo="neutral"):
    return f'<span class="lozenge {tipo}">{texto}</span>'


def num(valor, casas=1):
    return f"{valor:.{casas}f}".replace(".", ",")


def formatar_pct(valor):
    return f"{num(valor * 100)}%"


def faixa_imc(imc):
    """(rótulo, tipo do lozenge) pelas faixas da OMS"""
    if imc <= 0:
        return "Não informado", "neutral"
    if imc < 18.5:
        return "Abaixo do peso", "warning"
    if imc < 25:
        return "Normal", "success"
    if imc < 30:
        return "Sobrepeso", "warning"
    return "Obesidade", "danger"


FRASE_IMC = {
    "Abaixo do peso": "Seu peso está abaixo do recomendado para sua altura.",
    "Normal": "Seu peso está adequado para sua altura.",
    "Sobrepeso": "Seu peso está um pouco acima do recomendado para sua altura.",
    "Obesidade": "Seu peso está bem acima do recomendado e aumenta o risco de diabetes.",
}


def escala(valor, faixas, minimo, maximo, marcas):
    """Régua colorida com um marcador na posição do valor.
    faixas: [(fim da faixa, cor)]; marcas: valores escritos embaixo da régua."""
    def pos(v):
        return (min(max(v, minimo), maximo) - minimo) / (maximo - minimo) * 100
    inicio, segmentos = minimo, ""
    for fim, cor in faixas:
        segmentos += f'<div style="left: {pos(inicio):.1f}%; width: {pos(fim) - pos(inicio):.1f}%; background: {cor}"></div>'
        inicio = fim
    rotulos = "".join(f'<span style="left: {pos(m):.1f}%">{num(m, 1 if m % 1 else 0)}</span>' for m in marcas)
    return (f'<div class="escala"><div class="faixas">{segmentos}</div>'
            f'<div class="marcador" style="left: {pos(valor):.1f}%"></div></div>'
            + (f'<div class="escala-rotulos">{rotulos}</div>' if marcas else ''))


def escala_imc(imc):
    return escala(imc, [(18.5, "var(--warning)"), (25, "var(--success)"), (30, "var(--warning)"), (40, "var(--danger)")],
                  15, 40, [18.5, 25, 30])


def escala_risco(probabilidade, threshold):
    """Verde até o limite em que o modelo passa a indicar risco elevado, vermelho depois"""
    return escala(probabilidade * 100, [(threshold * 100, "var(--success)"), (100, "var(--danger)")], 0, 100, [])


def cartao_resultado(probabilidade, threshold, predicao, significado):
    etiqueta = lozenge("Risco elevado", "danger") if predicao else lozenge("Risco baixo", "success")
    st.markdown(f"""<div class="card">
<div class="card-row"><span class="card-title">Seu resultado</span>{etiqueta}</div>
<span class="risco-prob">{formatar_pct(probabilidade)}</span>
<span class="card-note">de probabilidade estimada pelo modelo</span>
{escala_risco(probabilidade, threshold)}
<div class="escala-legenda"><span><i style="background: var(--success)"></i>Risco baixo</span>
<span>a partir de {threshold * 100:.0f}%</span><span><i style="background: var(--danger)"></i>Risco elevado</span></div>
<div class="significado"><b>O que isso significa?</b> {significado}</div>
</div>""", unsafe_allow_html=True)


def lista_fatores(fatores, titulo="Seus valores comparados ao normal"):
    """fatores: lista de (nome, explicação, rótulo do lozenge, tipo do lozenge)"""
    linhas = "".join(
        f'<div class="fator"><span class="estado {tipo}">{SIMBOLO_ESTADO[tipo]}</span>'
        f'<div style="flex: 1"><div class="nome">{nome}</div><div class="detalhe">{detalhe}</div></div>'
        f'{lozenge(rotulo, tipo)}</div>'
        for nome, detalhe, rotulo, tipo in fatores
    )
    st.markdown(f'<div class="card"><span class="card-title">{titulo}</span>'
                f'<div class="fatores">{linhas}</div></div>', unsafe_allow_html=True)


# Visão técnica, mostrada só para contas de médico
NOMES_VARIAVEIS = {
    'Pregnancies': 'Gestações', 'Glucose': 'Glicemia TOTG 2h (mg/dL)', 'BloodPressure': 'Pressão diastólica (mm Hg)',
    'SkinThickness': 'Dobra cutânea do tríceps (mm)', 'Insulin': 'Insulina sérica (µU/ml)', 'BMI': 'IMC (kg/m²)',
    'DiabetesPedigreeFunction': 'Função pedigree', 'Age': 'Idade', 'HighBP': 'Hipertensão (0/1)',
    'HighChol': 'Colesterol alto (0/1)', 'Smoker': 'Fumante, 100+ cigarros (0/1)',
    'PhysActivity': 'Atividade física 30 dias (0/1)', 'GenHlth': 'Saúde autoavaliada (1 = excelente, 5 = ruim)',
    'Sex': 'Sexo (1 = masculino)',
}
MODELOS_DESCRICAO = {
    'modelo_clinico': ("Random Forest", "Pima Indians Diabetes (768 mulheres, 268 com diabetes)",
                       "validação cruzada aninhada 5 × 3"),
    'modelo_comportamental': ("XGBoost", "BRFSS 2015, versão Kaggle (253.680 entrevistas; pré-diabetes e diabetes agrupados)",
                              "conjunto de teste separado (20%)"),
}


def metricas_modelo(meta):
    """(AUC, F1, precisão, recall, Brier) dos dois formatos de meta"""
    if 'metricas_cv_aninhada' in meta:
        m = meta['metricas_cv_aninhada']
        return m['auc_media'], m['f1_media'], m['precisao_media'], m['recall_media'], m['brier_media']
    m = meta['metricas']
    return m['roc_auc'], m['f1_score'], m['precision'], m['recall'], m['brier']


def detalhes_tecnicos(nome_modelo, meta, probabilidade, entrada, extra=None):
    """Card técnico: modelo, validação, limiar e o vetor de entrada como o modelo o recebeu"""
    algoritmo, base, validacao = MODELOS_DESCRICAO[nome_modelo]
    auc, f1, precisao, recall, brier = metricas_modelo(meta)
    threshold = meta['threshold']
    linhas = ""
    for i, f in enumerate(meta['features']):
        valor = entrada[f]
        usado = valor
        observacao = ""
        if f in meta.get('zero_e_ausente', []) and valor == 0:
            usado = meta['imputer_medianas'][i]
            observacao = "não medido: mediana do treino"
        if f == 'Age' and nome_modelo == 'modelo_comportamental':
            observacao = "faixa de 5 anos do BRFSS (1 = 18-24, 13 = 80+)"
        linhas += (f"<tr><td>{NOMES_VARIAVEIS.get(f, f)}</td><td><code>{f}</code></td>"
                   f"<td>{extra.get(f, valor) if extra else valor}</td><td>{usado}</td>"
                   f"<td style='color: var(--muted)'>{observacao}</td></tr>")
    st.markdown(f"""<div class="card">
<div class="card-row"><span class="card-title">Detalhes técnicos</span>{lozenge("Visão do médico", "info")}</div>
<div class="tecnico">
<div><span>Modelo</span><b>{algoritmo}</b></div>
<div><span>Probabilidade</span><b>{probabilidade:.3f}</b></div>
<div><span>Limiar de decisão</span><b>{threshold:.2f}</b></div>
<div><span>Distância ao limiar</span><b>{probabilidade - threshold:+.3f}</b></div>
<div><span>AUC</span><b>{num(auc, 3)}</b></div>
<div><span>F1</span><b>{num(f1, 3)}</b></div>
<div><span>Precisão / recall</span><b>{num(precisao * 100, 0)}% / {num(recall * 100, 0)}%</b></div>
<div><span>Brier</span><b>{num(brier, 3)}</b></div>
</div>
<div class="card-note">Base de treino: {base}. Métricas em {validacao}. O limiar foi escolhido para maximizar o F1
(sem usar os dados de avaliação); a probabilidade não é calibrada como risco absoluto.</div>
<div class="tabela-card"><table class="tabela">
<thead><tr><th>Variável</th><th>Nome no modelo</th><th>Informado</th><th>Usado</th><th></th></tr></thead>
<tbody>{linhas}</tbody></table></div>
</div>""", unsafe_allow_html=True)


def ir_para(pagina_destino):
    st.session_state.pagina = pagina_destino


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
if 'perfil_tipo_conta' not in st.session_state:
    st.session_state.perfil_tipo_conta = "paciente"
if 'perfil_crm' not in st.session_state:
    st.session_state.perfil_crm = ""
if 'perfil_crm_uf' not in st.session_state:
    st.session_state.perfil_crm_uf = ""

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
        st.markdown(f"""
            <div class="login-marca">
                <div class="logo">{ICONE_LOGO.format(t=26)}</div>
                <span class="title">Predição de Diabetes</span>
                <span class="subtitle">Estimativa de risco de diabetes tipo 2 com Machine Learning</span>
            </div>
        """, unsafe_allow_html=True)

        if st.session_state.pop('conta_excluida', False):
            st.success("Sua conta e todos os seus dados foram excluídos.")

        # Tabs de Login/Cadastro
        tab1, tab2 = st.tabs(["Entrar", "Criar conta"])

        with tab1:
            with st.form("login_form"):
                st.markdown("#### Acesse sua conta")
                st.text_input("Email", key="email_login", placeholder="seu@email.com")
                st.text_input("Senha", type="password", key="senha_login", placeholder="Sua senha")

                submit_login = st.form_submit_button("Entrar", width="stretch", type="primary")

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
                            enviar_codigo = st.form_submit_button("Enviar código", width="stretch", type="primary")

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
                            redefinir = st.form_submit_button("Redefinir senha", width="stretch", type="primary")

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

            if 'cadastro_msg' in st.session_state:
                st.success(st.session_state.pop('cadastro_msg'))

            st.markdown("#### Crie sua conta gratuitamente")
            # Fora do form para os campos de CRM aparecerem assim que "Médico" for escolhido
            tipo_conta_cadastro = st.radio("Tipo de conta", ["Paciente", "Médico"], horizontal=True,
                                           key="tipo_conta_cadastro")

            with st.form("signup_form"):
                st.text_input("Nome Completo", key="nome_cadastro", placeholder="Seu nome completo")
                st.text_input("Email", key="email_cadastro", placeholder="seu@email.com")

                if tipo_conta_cadastro == "Médico":
                    col_crm, col_uf = st.columns([2, 1])
                    col_crm.text_input("CRM", key="crm_cadastro", max_chars=10, placeholder="Somente números")
                    col_uf.selectbox("UF", UFS, key="crm_uf_cadastro", index=UFS.index("SP"))

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

                submit_cadastro = st.form_submit_button("Criar conta", width="stretch", type="primary")

                if submit_cadastro:
                    nome_cadastro = st.session_state.nome_cadastro
                    email_cadastro = st.session_state.email_cadastro
                    senha_cadastro = st.session_state.senha_cadastro
                    senha_confirma = st.session_state.senha_confirma

                    e_medico = tipo_conta_cadastro == "Médico"
                    crm_cadastro = st.session_state.get('crm_cadastro', '').strip() if e_medico else ''

                    if not email_cadastro or not senha_cadastro or not nome_cadastro:
                        st.error("Por favor, preencha todos os campos")
                    elif e_medico and not crm_cadastro.isdigit():
                        st.error("Informe o número do CRM (somente números)")
                    elif not st.session_state.aceite_termo_cadastro:
                        st.error("Para criar a conta, é preciso aceitar o Termo de Consentimento")
                    elif senha_cadastro != senha_confirma:
                        st.error("As senhas não coincidem")
                    elif len(senha_cadastro) < SENHA_MINIMA:
                        st.error(f"Senha deve ter no mínimo {SENHA_MINIMA} caracteres")
                    else:
                        with st.spinner("Criando conta..."):
                            sucesso, mensagem = auth.signup(
                                email_cadastro, senha_cadastro, nome_cadastro,
                                tipo_conta='medico' if e_medico else 'paciente',
                                crm=crm_cadastro,
                                crm_uf=st.session_state.get('crm_uf_cadastro', '') if e_medico else '',
                            )
                        if sucesso:
                            # Com confirmação de email não há login ainda: a mensagem precisa
                            # sobreviver ao rerun para a pessoa saber que deve abrir o email
                            if not auth.is_logged_in():
                                st.session_state.cadastro_msg = mensagem
                            st.rerun()
                        else:
                            st.error(mensagem)


        # Footer
        st.markdown('<div class="auth-footer">David Reis · 2026 · Uso educacional, não substitui consulta médica.</div>',
                    unsafe_allow_html=True)

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
        cabecalho("Privacidade dos seus dados")
        st.info("Para continuar usando o sistema, leia e aceite o termo abaixo.")
        with st.container(height=400):
            st.markdown(TERMO_PRIVACIDADE)

        aceite = st.checkbox(
            "Li e aceito o Termo de Consentimento e a Política de Privacidade, "
            "incluindo o tratamento dos meus dados de saúde"
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Continuar", type="primary", width="stretch", disabled=not aceite):
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
# USUÁRIO LOGADO - MENU LATERAL
# ============================================================================

e_medico = st.session_state.perfil_tipo_conta == "medico"

# (chave da página, rótulo no menu, ícone)
MENU = [
    ("Principal", [
        ("Início", "Início", ":material/home:"),
        ("Meus Pacientes", "Meus pacientes", ":material/groups:") if e_medico
        else ("Meu Diário", "Meu diário", ":material/menu_book:"),
        ("Histórico & Estatísticas", "Histórico", ":material/history:"),
    ]),
    ("Avaliações", [
        ("Modelo Comportamental", "Questionário rápido", ":material/assignment:"),
        ("Modelo Clínico", "Exames laboratoriais", ":material/biotech:"),
    ]),
    ("Sobre", [
        ("Comparação", "Sobre os modelos", ":material/info:"),
    ]),
]
PAGINAS = {chave for _, itens in MENU for chave, _, _ in itens} | {"Meu Perfil"}
if st.session_state.get('pagina') not in PAGINAS:
    st.session_state.pagina = "Início"
pagina = st.session_state.pagina

nome_exibir = st.session_state.perfil_nome or auth.get_user_email().split('@')[0]
partes_nome = nome_exibir.split()
iniciais = (partes_nome[0][0] + (partes_nome[-1][0] if len(partes_nome) > 1 else "")).upper()

with st.sidebar:
    st.markdown(f"""
        <div class="marca">
            <div class="logo">{ICONE_LOGO.format(t=18)}</div>
            <div><div class="nome">Predição de Diabetes</div><div class="desc">Risco de diabetes tipo 2</div></div>
        </div>
    """, unsafe_allow_html=True)

    for grupo, itens in MENU:
        st.markdown(f'<div class="grupo-menu">{grupo}</div>', unsafe_allow_html=True)
        for chave, rotulo, icone in itens:
            st.button(rotulo, icon=icone, key=f"nav_{chave}", width="stretch",
                      type="primary" if pagina == chave else "tertiary",
                      on_click=ir_para, args=(chave,))

    # Perfil fica separado da navegação, num card próprio
    st.markdown('<div style="height: 20px"></div>', unsafe_allow_html=True)
    with st.container(border=True, key="card_perfil"):
        if e_medico:
            tipo_conta_txt = "Médico"
            if st.session_state.perfil_crm:
                tipo_conta_txt += f" · CRM {st.session_state.perfil_crm}/{st.session_state.perfil_crm_uf}"
        else:
            tipo_conta_txt = "Paciente"

        if st.session_state.perfil_preenchido and not e_medico:
            imc_txt, imc_tipo = faixa_imc(st.session_state.perfil_imc)
            imc_perfil = st.session_state.perfil_imc
            resumo = (f"{st.session_state.perfil_idade} anos · {st.session_state.perfil_sexo} · "
                      f"{num(st.session_state.perfil_altura / 100, 2)} m · {num(st.session_state.perfil_peso)} kg")
            familia = ("Tem diabetes na família" if st.session_state.perfil_hist_familiar == "Sim"
                       else "Sem diabetes na família")
            # Numa linha só: com indentação o markdown trataria o HTML como bloco de código
            dados_html = (f'<div class="perfil-resumo">{resumo}</div>'
                          f'<div class="perfil-imc"><div class="card-row"><span class="rotulo">IMC</span>'
                          f'<span class="valor">{num(imc_perfil)} {lozenge(imc_txt, imc_tipo)}</span></div>'
                          f'{escala_imc(imc_perfil)}'
                          f'<div class="perfil-frase">{FRASE_IMC.get(imc_txt, "")}</div></div>'
                          f'<div class="perfil-resumo">{familia}</div>')
        elif e_medico:
            dados_html = ""
        else:
            dados_html = '<div class="perfil-tipo">Complete seu perfil: os dados preenchem as avaliações.</div>'

        st.markdown(f"""
            <div class="perfil-card">
                <div class="perfil-topo">
                    <div class="avatar">{html.escape(iniciais)}</div>
                    <div><div class="perfil-nome">{html.escape(nome_exibir)}</div><div class="perfil-tipo">{tipo_conta_txt}</div></div>
                </div>
                {dados_html}
            </div>
        """, unsafe_allow_html=True)

        st.button("Editar perfil", key="perfil_editar", icon=":material/edit:", width="stretch",
                  type="primary" if pagina == "Meu Perfil" else "secondary",
                  on_click=ir_para, args=("Meu Perfil",))
        if st.button("Sair", key="perfil_sair", icon=":material/logout:", width="stretch"):
            auth.logout()
            st.rerun()

    st.markdown('<div class="aviso-rodape">Uso educacional. Não substitui consulta médica.</div>',
                unsafe_allow_html=True)

# Ao trocar de página, o Streamlit mantém os elementos da anterior na tela (marcados com
# data-stale) até a nova terminar de carregar. Só nessas execuções esses elementos são escondidos;
# nas demais (cliques dentro da mesma página) o espaço recebe um texto vazio, que substitui o estilo
# logo no começo da execução e evita que a tela pisque.
estilo_troca = st.empty()
if st.session_state.get('pagina_desenhada') != pagina:
    estilo_troca.markdown("""<style>
        [data-testid="stMain"] [data-stale="true"],
        [data-testid="stMain"] [data-testid="stLayoutWrapper"]:has([data-stale="true"]):not(:has([data-stale="false"])) {
            display: none !important;
        }
    </style>""", unsafe_allow_html=True)
    st.session_state.pagina_desenhada = pagina
else:
    estilo_troca.markdown("")

# ============================================================================
# PÁGINA INÍCIO
# ============================================================================

if pagina == "Início":
    try:
        predicoes_recentes = supabase_db.obter_ultimas_predicoes(limite=5, user_id=auth.get_user_id())
    except Exception:
        predicoes_recentes = []

    col_titulo, col_acoes = st.columns([3, 2], vertical_alignment="bottom")
    with col_titulo:
        cabecalho(f"Olá, {html.escape(nome_exibir.split()[0])}",
                  "Acompanhe seu risco de diabetes tipo 2 e sua rotina.", eyebrow="Início")
    with col_acoes:
        col_a, col_b = st.columns(2)
        if e_medico:
            col_a.button("Meus pacientes", width="stretch", on_click=ir_para, args=("Meus Pacientes",))
        else:
            col_a.button("Registrar no diário", width="stretch", on_click=ir_para, args=("Meu Diário",))
        col_b.button("Nova avaliação", icon=":material/add:", type="primary", width="stretch",
                     on_click=ir_para, args=("Modelo Comportamental",))

    # ---- Resumo ----
    col1, col2, col3 = st.columns(3)

    with col1, st.container(border=True, key="card_ultima", height="stretch"):
        if predicoes_recentes:
            ultima = predicoes_recentes[0]
            elevado = ultima['resultado'] == 1
            etiqueta = lozenge("Risco elevado", "danger") if elevado else lozenge("Risco baixo", "success")
            data_ultima = datetime.fromisoformat(ultima['created_at'].replace('Z', '+00:00')).strftime('%d/%m/%Y')
            tipo_ultima = "Exames laboratoriais" if ultima['tipo_modelo'] == 'clinico' else "Questionário rápido"
            limite = carregar_modelo('modelo_clinico' if ultima['tipo_modelo'] == 'clinico'
                                     else 'modelo_comportamental')[1]['threshold']
            frase = ("Acima do limite: vale conversar com um médico." if elevado
                     else "Na faixa verde: nenhum sinal de risco elevado.")
            st.markdown(
                f'<div class="card-row"><span class="card-title">Última avaliação</span>{etiqueta}</div>'
                f'<div class="card-big">{formatar_pct(ultima["probabilidade"])}</div>'
                f'{escala_risco(ultima["probabilidade"], limite)}'
                f'<div class="card-text">{frase}</div>'
                f'<div class="card-note">{tipo_ultima} · {data_ultima}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card-title">Última avaliação</div>'
                        '<div class="card-text">Você ainda não fez nenhuma avaliação.</div>', unsafe_allow_html=True)
        st.button("Ver histórico", key="ir_historico", type="tertiary",
                  on_click=ir_para, args=("Histórico & Estatísticas",))

    with col2, st.container(border=True, key="card_diario", height="stretch"):
        if e_medico:
            try:
                n_pacientes = len(medico._rpc('meus_pacientes'))
            except Exception:
                n_pacientes = None
            st.markdown(
                f'<div class="card-title">Meus pacientes</div>'
                f'<div class="card-big">{n_pacientes if n_pacientes is not None else "—"}</div>'
                f'<div class="card-note">compartilham os dados com você</div>', unsafe_allow_html=True)
            st.button("Abrir pacientes", key="ir_pacientes", type="tertiary", on_click=ir_para, args=("Meus Pacientes",))
        else:
            registros_hoje = diario.carregar_registros(auth.get_user_id(), diario.hoje(), diario.hoje())
            n_refeicoes = len([r for r in registros_hoje if r['tipo'] == 'refeicao'])
            minutos = sum(r['dados'].get('minutos', 0) for r in registros_hoje if r['tipo'] == 'atividade')
            glicemias = [r['dados'] for r in registros_hoje if r['tipo'] == 'glicemia']
            itens_dia = [
                ("Refeições", f"{n_refeicoes} registrada{'s' if n_refeicoes > 1 else ''}" if n_refeicoes else ""),
                ("Atividade física", f"{minutos} min" if minutos else ""),
                ("Glicemia", f"{glicemias[-1]['valor']} mg/dL" if glicemias else ""),
            ]
            feitos = sum(1 for _, valor in itens_dia if valor)

            linhas = "".join(
                f'<div class="card-row" style="justify-content: flex-start">'
                f'<span class="estado {"success" if valor else "neutral"}">{"✓" if valor else "–"}</span>'
                f'<span style="flex: 1; color: var(--subtle)">{rotulo}</span>'
                f'<span style="color: {"var(--heading)" if valor else "var(--muted)"}; font-weight: 500">'
                f'{valor or "Falta registrar"}</span></div>'
                for rotulo, valor in itens_dia)
            st.markdown(
                f'<div class="card-row"><span class="card-title">Diário de hoje</span>'
                f'<span class="card-note">{feitos} de {len(itens_dia)}</span></div>'
                f'<div class="progresso"><div style="width: {feitos / len(itens_dia) * 100:.0f}%"></div></div>'
                f'{linhas}', unsafe_allow_html=True)
            st.button("Abrir diário", key="ir_diario", type="tertiary", on_click=ir_para, args=("Meu Diário",))

    with col3, st.container(border=True, key="card_passo", height="stretch"):
        imc_perfil = st.session_state.perfil_imc
        if predicoes_recentes and predicoes_recentes[0]['resultado'] == 1:
            texto_passo = ("Sua última avaliação indicou risco elevado. Procure um médico e leve o "
                           "relatório em PDF gerado na avaliação.")
            destino, rotulo_passo = "Modelo Clínico", "Avaliar com exames"
        elif not predicoes_recentes:
            texto_passo = "Faça sua primeira avaliação. O questionário rápido leva um minuto e não precisa de exames."
            destino, rotulo_passo = "Modelo Comportamental", "Começar questionário"
        elif not st.session_state.perfil_preenchido:
            texto_passo = "Complete seu perfil: idade, sexo, altura e peso preenchem as avaliações sozinhos."
            destino, rotulo_passo = "Meu Perfil", "Completar perfil"
        elif imc_perfil >= 25 and not e_medico:
            texto_passo = (f"Seu IMC ({num(imc_perfil)}) está na faixa de {faixa_imc(imc_perfil)[0].lower()}. "
                           "Chegar abaixo de 25 com alimentação e exercícios reduz o risco.")
            destino, rotulo_passo = "Meu Diário", "Registrar atividade"
        elif not e_medico and feitos < len(itens_dia):
            texto_passo = ("Registrar refeições, atividade e glicemia todo dia mostra como sua rotina "
                           "afeta sua saúde ao longo do tempo.")
            destino, rotulo_passo = "Meu Diário", "Preencher o diário de hoje"
        else:
            texto_passo = "Continue com hábitos saudáveis e refaça a avaliação uma vez por ano."
            destino, rotulo_passo = "Modelo Comportamental", "Nova avaliação"
        st.markdown(f'<div class="card-title">Próximo passo</div>'
                    f'<div class="card-text">{texto_passo}</div>', unsafe_allow_html=True)
        st.button(rotulo_passo, key="ir_passo", type="primary", width="stretch", on_click=ir_para, args=(destino,))

    # ---- Nova avaliação ----
    st.markdown("### Nova avaliação")
    col1, col2 = st.columns(2)
    with col1, st.container(border=True, key="card_questionario", height="stretch"):
        st.markdown(
            '<div class="card-row"><span class="card-title">Questionário rápido</span>' + lozenge("Sem exames", "info") + '</div>'
            '<div class="card-text">8 perguntas simples sobre você e seus hábitos, como peso, pressão e atividade física.</div>'
            '<div class="card-note">' + ('XGBoost · BRFSS 2015 · AUC 0,821' if e_medico
                                         else 'Leva 1 minuto · bom para começar') + '</div>', unsafe_allow_html=True)
        st.button("Responder questionário", key="ir_comportamental", width="stretch",
                  on_click=ir_para, args=("Modelo Comportamental",))
    with col2, st.container(border=True, key="card_exames", height="stretch"):
        st.markdown(
            '<div class="card-row"><span class="card-title">Exames laboratoriais</span>' + lozenge("Mais preciso", "success") + '</div>'
            '<div class="card-text">Use os resultados dos seus exames de sangue, como a curva glicêmica (TOTG) e a insulina.</div>'
            
            '<div class="card-note">' + ('Random Forest · Pima Indians · AUC 0,839' if e_medico
                                         else 'Leva 2 minutos · precisa ter os exames em mãos') + '</div>',
            unsafe_allow_html=True)
        st.button("Informar exames", key="ir_clinico", width="stretch", on_click=ir_para, args=("Modelo Clínico",))

    # ---- Histórico recente ----
    st.markdown("### Histórico recente")
    if predicoes_recentes:
        linhas = ""
        for pred in predicoes_recentes:
            data_pred = datetime.fromisoformat(pred['created_at'].replace('Z', '+00:00')).strftime('%d/%m/%Y')
            tipo_pred = "Exames laboratoriais" if pred['tipo_modelo'] == 'clinico' else "Questionário rápido"
            etiqueta = lozenge("Risco elevado", "danger") if pred['resultado'] == 1 else lozenge("Risco baixo", "success")
            linhas += (f"<tr><td>{data_pred}</td><td style='color: var(--heading)'>{tipo_pred}</td>"
                       f"<td style='color: var(--heading)'>{formatar_pct(pred['probabilidade'])}</td><td>{etiqueta}</td></tr>")
        st.markdown(f"""
            <div class="tabela-card"><table class="tabela">
                <thead><tr><th>Data</th><th>Avaliação</th><th>Probabilidade</th><th>Resultado</th></tr></thead>
                <tbody>{linhas}</tbody>
            </table></div>
        """, unsafe_allow_html=True)
    else:
        st.info("Suas avaliações vão aparecer aqui.")


# ============================================================================
# PÁGINAS DO DIÁRIO (paciente) E MEUS PACIENTES (médico)
# ============================================================================

elif pagina == "Meu Diário":
    cabecalho("Meu diário", "Registre sua rotina: alimentação, atividade física, glicemia e mais.", eyebrow="Principal")
    diario.mostrar_diario(auth.get_user_id())

elif pagina == "Meus Pacientes":
    medico.pagina_meus_pacientes(auth.get_user_id())


# ============================================================================
# PÁGINA MEU PERFIL
# ============================================================================

elif pagina == "Meu Perfil":
    col_titulo, col_acoes = st.columns([3, 1], vertical_alignment="bottom")
    with col_titulo:
        cabecalho("Meu perfil", "Esses dados preenchem as avaliações automaticamente.", eyebrow="Conta")
    # O botão fica no topo, mas os campos abaixo já têm os valores editados quando ele é clicado
    salvar = col_acoes.button("Salvar perfil", type="primary", width="stretch")
    aviso_perfil = st.empty()

    abas = ["Dados pessoais", "Saúde"] + ([] if e_medico else ["Médicos com acesso"]) + ["Privacidade"]
    tabs = dict(zip(abas, st.tabs(abas)))

    with tabs["Dados pessoais"]:
        nome = st.text_input("Nome completo", value=st.session_state.perfil_nome)
        col1, col2, col3, col4 = st.columns(4)
        idade = col1.number_input("Idade", min_value=1, max_value=120, value=st.session_state.perfil_idade)
        sexo = col2.selectbox("Sexo", ["Masculino", "Feminino"],
                              index=0 if st.session_state.perfil_sexo == "Masculino" else 1)
        altura = col3.number_input("Altura (cm)", min_value=50.0, max_value=250.0,
                                   value=st.session_state.perfil_altura, step=0.1)
        peso = col4.number_input("Peso (kg)", min_value=20.0, max_value=300.0,
                                 value=st.session_state.perfil_peso, step=0.1)

        imc = peso / ((altura / 100) ** 2) if altura > 0 and peso > 0 else 0
        imc_txt, imc_tipo = faixa_imc(imc)
        st.markdown(f"""
            <div class="card" style="flex-direction: row; align-items: center; flex-wrap: wrap; gap: 12px; padding: 12px 16px">
                <span style="color: var(--subtle)">IMC calculado</span>
                <span style="font-size: 20px; font-weight: 600; color: var(--heading)">{num(imc)}</span>
                {lozenge(imc_txt, imc_tipo)}
                <span class="card-note">Atualiza sozinho ao mudar altura ou peso.</span>
            </div>
        """, unsafe_allow_html=True)

        crm, crm_uf = st.session_state.perfil_crm, st.session_state.perfil_crm_uf
        if e_medico:
            st.markdown("##### Registro profissional")
            col_crm, col_uf = st.columns([2, 1])
            crm = col_crm.text_input("CRM", value=st.session_state.perfil_crm, max_chars=10).strip()
            crm_uf = col_uf.selectbox("UF do CRM", UFS, index=UFS.index(crm_uf) if crm_uf in UFS else UFS.index("SP"))

    with tabs["Saúde"]:
        hist_familiar = st.segmented_control(
            "Algum parente de primeiro grau (pais ou irmãos) tem diabetes?", ["Não", "Sim"],
            default=st.session_state.perfil_hist_familiar) or "Não"
        col1, col2 = st.columns(2)
        alergias = col1.text_area("Alergias", value=st.session_state.perfil_alergias,
                                  placeholder="Ex.: penicilina, frutos do mar")
        medicacoes = col2.text_area("Medicações em uso", value=st.session_state.perfil_medicacoes,
                                    placeholder="Ex.: metformina 500 mg")

    if not e_medico:
        with tabs["Médicos com acesso"]:
            medico.secao_compartilhamento(auth.get_user_id())

    with tabs["Privacidade"]:
        with st.expander("Termo de Consentimento e Política de Privacidade"):
            st.markdown(TERMO_PRIVACIDADE)
        secao_excluir_conta()

    if 'perfil_msg' in st.session_state:
        aviso_perfil.success(st.session_state.pop('perfil_msg'))

    if salvar and e_medico and not crm.isdigit():
        aviso_perfil.error("Informe o número do CRM (somente números)")
    elif salvar:
        sucesso, msg_bd = auth.salvar_perfil(nome, idade, sexo, altura, peso, imc, alergias, hist_familiar,
                                             medicacoes, st.session_state.perfil_tipo_conta, crm, crm_uf)
        st.session_state.perfil_nome = nome
        st.session_state.perfil_idade = idade
        st.session_state.perfil_sexo = sexo
        st.session_state.perfil_altura = altura
        st.session_state.perfil_peso = peso
        st.session_state.perfil_imc = imc
        st.session_state.perfil_alergias = alergias
        st.session_state.perfil_hist_familiar = hist_familiar
        st.session_state.perfil_medicacoes = medicacoes
        st.session_state.perfil_crm = crm
        st.session_state.perfil_crm_uf = crm_uf
        st.session_state.perfil_preenchido = True
        if sucesso:
            # Rerun para o card de perfil do menu lateral mostrar os dados novos
            st.session_state.perfil_msg = "Perfil salvo com sucesso!"
            st.rerun()
        aviso_perfil.warning(f"Perfil salvo só nesta sessão. Erro no banco: {msg_bd}")

# ============================================================================
# PÁGINA MODELO CLÍNICO
# ============================================================================

elif pagina == "Modelo Clínico":
    cabecalho("Exames laboratoriais", "Estimativa de risco a partir de exames de sangue e medidas clínicas.",
              eyebrow="Avaliações")

    st.info("Informe a **glicemia do TOTG 2h** (2 horas após tomar glicose), não a de jejum, e a pressão "
            "**diastólica** (o número de baixo). Insulina e dobra cutânea podem ficar em 0 se você não tiver.")

    with st.container(border=True, key="card_form_clinico"):
        if st.session_state.perfil_preenchido:
            st.caption(f"Idade e IMC vêm do seu perfil ({st.session_state.perfil_idade} anos, "
                       f"IMC {num(st.session_state.perfil_imc)}).")
        col1, col2, col3, col4 = st.columns(4)
        glucose = col1.number_input("Glicemia TOTG 2h (mg/dL)", min_value=0, max_value=250, value=120,
                                    help="Glicemia 2 horas após ingerir glicose (teste oral de tolerância à glicose). "
                                         "Não é a glicemia de jejum.")
        blood_pressure = col2.number_input("Pressão diastólica (mm Hg)", min_value=0, max_value=150, value=80,
                                           help="O número de baixo da pressão. Ex.: em 120/80, informe 80.")
        bmi = col3.number_input("IMC", min_value=10.0, max_value=70.0,
                                value=st.session_state.perfil_imc if st.session_state.perfil_imc > 0 else 25.0, step=0.1)
        age = col4.number_input("Idade (anos)", min_value=18, max_value=120,
                                value=max(18, int(st.session_state.perfil_idade)))
        insulin = col1.number_input("Insulina sérica (µU/ml)", min_value=0, max_value=900, value=80,
                                    help="Deixe 0 se não tiver esse exame.")
        skin_thickness = col2.number_input("Dobra cutânea (mm)", min_value=0, max_value=100, value=20,
                                           help="Dobra cutânea do tríceps. Deixe 0 se não tiver essa medida.")
        pregnancies = col3.number_input("Número de gestações", min_value=0, max_value=20, value=0)
        diabetes_pedigree = col4.number_input("Função pedigree de diabetes", min_value=0.0, max_value=3.0,
                                              value=0.5, step=0.01,
                                              help="Resume o histórico de diabetes na família. 0,5 é um valor típico.")
        analisar = st.button("Analisar risco", key="btn_clinico", type="primary", width="stretch")

    if analisar:
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

        # Limites do TOTG 2h: < 140 normal, 140-199 tolerância diminuída, >= 200 faixa de diabetes
        if glucose >= 200:
            fator_glicose = ("Faixa de diabetes", "danger")
        elif glucose >= 140:
            fator_glicose = ("Tolerância diminuída", "warning")
        else:
            fator_glicose = ("Normal", "success")
        imc_txt, imc_tipo = faixa_imc(bmi)
        if insulin == 0:
            fator_insulina = ("Não informado", "neutral")
        elif insulin > 200:
            fator_insulina = ("Elevada", "danger")
        elif insulin > 150:
            fator_insulina = ("Moderada", "warning")
        else:
            fator_insulina = ("Normal", "success")
        fator_pressao = ("Alta", "warning") if blood_pressure >= 90 else (
            ("Não informada", "neutral") if blood_pressure == 0 else ("Normal", "success"))
        fatores = [
            ("Glicemia após 2 horas (TOTG)", f"{glucose} mg/dL · o normal é abaixo de 140", *fator_glicose),
            ("IMC", f"{num(bmi)} · o normal vai de 18,5 a 24,9", imc_txt, imc_tipo),
            ("Insulina", f"{insulin} µU/ml · acima de 150 pode indicar resistência à insulina" if insulin
             else "sem exame: o modelo usa um valor típico", *fator_insulina),
            ("Pressão diastólica", f"{blood_pressure} mm Hg · o normal é abaixo de 90" if blood_pressure
             else "sem medida", *fator_pressao),
            ("Idade", f"{age} anos · o risco aumenta depois dos 45",
             *(("Fator de risco", "warning") if age > 45 else ("Sem impacto", "info"))),
        ]

        recomendacoes = []

        if predicao == 1:
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

            recomendacoes.append("**Protocolo completo:** Glicemia de jejum + HbA1c + curva glicêmica + insulina + peptídeo C")
            recomendacoes.append("**Exercício:** 30 min/dia de caminhada melhora a sensibilidade à insulina")

        else:
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
                recomendacoes.append("**Histórico familiar:** Mesmo com exames normais, monitore a glicemia uma vez por ano")

            if age < 40:
                recomendacoes.append("**Idade jovem:** Ótimo momento para criar hábitos saudáveis que duram a vida toda")

            recomendacoes.append("**Mantenha o exercício:** 150 min/semana mantém a sensibilidade à insulina")
            recomendacoes.append("**Dieta preventiva:** Reduza açúcar, refrigerantes e carboidratos refinados")

        st.markdown("### Resultado")
        col_res, col_fat = st.columns(2)
        with col_res:
            cartao_resultado(probabilidade, meta_clinico['threshold'], predicao,
                             "Seus exames se parecem com os de pessoas que têm diabetes. Isso <b>não é um "
                             "diagnóstico</b>: procure um médico para confirmar com novos exames."
                             if predicao else
                             "Seus exames não mostram sinais de risco elevado de diabetes. Continue se cuidando "
                             "e repita os exames uma vez por ano.")
        with col_fat:
            lista_fatores(fatores)

        if e_medico:
            detalhes_tecnicos('modelo_clinico', meta_clinico, probabilidade, {
                'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi,
                'DiabetesPedigreeFunction': diabetes_pedigree, 'Age': age,
            })

        with st.container(border=True, key="card_recomendacoes_clinico"):
            st.markdown('<span class="card-title">Recomendações</span>', unsafe_allow_html=True)
            st.markdown("\n".join(f"- {rec}" for rec in recomendacoes))
            if st.session_state.perfil_alergias:
                st.warning(f"**Alergias:** {st.session_state.perfil_alergias}. Informe ao médico antes de qualquer medicação.")

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
                        label="Baixar relatório em PDF para levar ao médico",
                        data=f,
                        file_name=f"relatorio_diabetes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        icon=":material/download:",
                    )

            except Exception as e:
                st.info(f"Geração de PDF: {e}")


# ============================================================================
# PÁGINA MODELO COMPORTAMENTAL  
# ============================================================================

elif pagina == "Modelo Comportamental":
    cabecalho("Questionário rápido", "Estimativa de risco a partir de hábitos e condições de saúde, sem exames.",
              eyebrow="Avaliações")
    st.caption("Estima o risco de **pré-diabetes ou diabetes**: na base usada (BRFSS 2015) os dois são agrupados, "
               "e o diagnóstico é autorreferido pelos entrevistados.")

    opcoes_saude = ["Excelente", "Muito boa", "Boa", "Razoável", "Ruim"]
    with st.container(border=True, key="card_form_comportamental"):
        col1, col2, col3, col4 = st.columns(4)
        idade_brfss = col1.number_input("Idade (anos)", min_value=18, max_value=120,
                                        value=int(st.session_state.perfil_idade) if st.session_state.perfil_idade >= 18 else 30)
        sexo_brfss = col2.selectbox("Sexo", ["Masculino", "Feminino"],
                                    index=0 if st.session_state.perfil_sexo == "Masculino" else 1)
        bmi_brfss = col3.number_input("IMC", min_value=10.0, max_value=70.0,
                                      value=st.session_state.perfil_imc if st.session_state.perfil_imc > 0 else 25.0)
        gen_health = col4.selectbox("Como você avalia sua saúde?", opcoes_saude, index=2)
        high_bp = col1.selectbox("Pressão alta?", ["Não", "Sim"])
        high_chol = col2.selectbox("Colesterol alto?", ["Não", "Sim"])
        phys_activity = col3.selectbox("Atividade física no último mês?", ["Não", "Sim"])
        smoker = col4.selectbox("Fumou 100 cigarros na vida?", ["Não", "Sim"],
                                help="Pergunta do BRFSS para identificar quem é ou já foi fumante.")
        analisar = st.button("Analisar risco", key="btn_comportamental", type="primary", width="stretch")

    if analisar:
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

        imc_txt, imc_tipo = faixa_imc(bmi_brfss)
        fatores = [
            ("Pressão alta", "aumenta o risco de diabetes" if high_bp == "Sim" else "você informou que não tem",
             *(("Fator de risco", "danger") if high_bp == "Sim" else ("Ok", "success"))),
            ("Colesterol alto", "aumenta o risco de diabetes" if high_chol == "Sim" else "você informou que não tem",
             *(("Fator de risco", "danger") if high_chol == "Sim" else ("Ok", "success"))),
            ("IMC", f"{num(bmi_brfss)} · o normal vai de 18,5 a 24,9", imc_txt, imc_tipo),
            ("Atividade física", "ajuda a controlar o açúcar no sangue" if phys_activity == "Sim"
             else "ficar parado aumenta o risco",
             *(("Ativo", "success") if phys_activity == "Sim" else ("Sedentário", "warning"))),
            ("Cigarro", "fumar aumenta o risco" if smoker == "Sim" else "você informou que não fuma",
             *(("Fumante ou ex", "warning") if smoker == "Sim" else ("Ok", "success"))),
            ("Como você avalia sua saúde", gen_health,
             *(("Fator de risco", "warning") if gen_health in ("Razoável", "Ruim") else ("Ok", "success"))),
        ]

        if predicao == 1:
            recomendacoes = ["**Procure um médico** e peça glicemia de jejum e HbA1c",
                             "**Atividade física:** 150 min por semana de caminhada ou equivalente",
                             "**Alimentação:** menos açúcar, refrigerantes e carboidratos refinados",
                             "Se já tiver exames, faça também a **avaliação com exames laboratoriais**"]
        else:
            recomendacoes = ["**Mantenha os bons hábitos** de alimentação e atividade física",
                             "**Check-up anual** com glicemia de jejum",
                             "Registre sua rotina no **diário** para acompanhar a evolução"]

        st.markdown("### Resultado")
        col_res, col_fat = st.columns(2)
        with col_res:
            cartao_resultado(probabilidade, meta_comp['threshold'], predicao,
                             "Suas respostas se parecem com as de pessoas que têm pré-diabetes ou diabetes. "
                             "Isso <b>não é um diagnóstico</b>: peça ao médico um exame de glicemia."
                             if predicao else
                             "Suas respostas não mostram sinais de risco elevado. Mantenha os bons hábitos "
                             "e faça um check-up uma vez por ano.")
            with st.container(border=True, key="card_recomendacoes_comportamental"):
                st.markdown('<span class="card-title">Recomendações</span>', unsafe_allow_html=True)
                st.markdown("\n".join(f"- {rec}" for rec in recomendacoes))
        with col_fat:
            lista_fatores(fatores)

        if e_medico:
            detalhes_tecnicos('modelo_comportamental', meta_comp, probabilidade, entrada,
                              extra={'Age': f"{idade_brfss} anos → {faixa_idade}"})

# ============================================================================
# PÁGINA COMPARAÇÃO
# ============================================================================

elif pagina == "Comparação":
    cabecalho("Sobre os modelos", "Como cada avaliação funciona e quando usar cada uma.", eyebrow="Sobre")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Modelo Clínico</h3>
            <p><strong>Dataset:</strong> Pima Indians (768 registros)</p>
            <p><strong>F1-Score:</strong> 0,69</p>
            <p><strong>Precisão:</strong> 59,9%</p>
            <p><strong>Recall:</strong> 82,2%</p>
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
        <div class="card">
            <h3>Modelo Comportamental</h3>
            <p><strong>Dataset:</strong> BRFSS (253k registros)</p>
            <p><strong>F1-Score:</strong> 0,46</p>
            <p><strong>Precisão:</strong> 36,7%</p>
            <p><strong>Recall:</strong> 61,1%</p>
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
    cabecalho("Histórico", "Suas avaliações anteriores e a evolução do risco.", eyebrow="Principal")
    
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
            ax.set_facecolor('#22272B')
            fig.patch.set_facecolor('#1D2125')
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
