# -*- coding: utf-8 -*-
"""
DIÁRIO DE SAÚDE
Registro da rotina do paciente (alimentação, atividade física, glicemia, peso,
pressão, sono, sintomas e hábitos), resumo do dia, alertas e evolução.
A mesma tela é usada pelo médico, só para leitura (mostrar_diario(..., somente_leitura=True)).
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

import alimentos
from supabase_db import db as supabase_db

COR_DESTAQUE = "#4fb9c4"   # primaryColor do tema (.streamlit/config.toml)
COR_REFERENCIA = "#8a8a8a"


def hoje() -> date:
    """Data de hoje no Brasil (o servidor do Streamlit Cloud roda em UTC)"""
    return datetime.now(ZoneInfo("America/Sao_Paulo")).date()

REFEICOES = ["Café da manhã", "Lanche da manhã", "Almoço", "Lanche da tarde", "Jantar", "Ceia"]

ATIVIDADES = ["Caminhada", "Corrida", "Bicicleta", "Musculação", "Natação", "Hidroginástica",
              "Dança", "Futebol / esportes", "Alongamento / ioga", "Outra"]
INTENSIDADES = ["Leve", "Moderada", "Intensa"]

MOMENTOS_GLICEMIA = ["Jejum", "Antes de refeição", "2 horas após refeição", "Ao acaso", "Antes de dormir"]

# Sintomas clássicos de diabetes (SBD / ADA) primeiro, depois sintomas associados
SINTOMAS_CLASSICOS = ["Muita sede", "Urinar muitas vezes", "Muita fome", "Perda de peso sem motivo"]
SINTOMAS_OUTROS = ["Cansaço / fraqueza", "Visão embaçada", "Feridas que demoram a cicatrizar",
                   "Formigamento nas mãos ou pés", "Infecções frequentes (urina, pele, candidíase)"]

CATEGORIAS = {
    "refeicao": "Alimentação",
    "atividade": "Atividade física",
    "glicemia": "Glicemia",
    "peso": "Peso",
    "pressao": "Pressão arterial",
    "sono": "Sono",
    "sintomas": "Sintomas",
    "habitos": "Hábitos (álcool, cigarro, refrigerante)",
}


# ============================================================================
# BANCO DE DADOS
# ============================================================================

def _client():
    if not supabase_db.conectado:
        return None
    return supabase_db.client


def salvar_registro(user_id: str, tipo: str, dia: date, dados: Dict) -> bool:
    client = _client()
    if not client or not user_id:
        return False
    try:
        client.table('diario_registros').insert({
            'user_id': user_id, 'tipo': tipo, 'data': dia.isoformat(), 'dados': dados,
        }).execute()
        return True
    except Exception as e:
        print(f"Erro ao salvar registro do diário: {e}")
        return False


def excluir_registro(registro_id: int) -> bool:
    client = _client()
    if not client:
        return False
    try:
        client.table('diario_registros').delete().eq('id', registro_id).execute()
        return True
    except Exception as e:
        print(f"Erro ao excluir registro do diário: {e}")
        return False


def carregar_registros(user_id: str, inicio: date, fim: date) -> List[Dict]:
    client = _client()
    if not client or not user_id:
        return []
    try:
        resposta = (client.table('diario_registros').select('*')
                    .eq('user_id', user_id)
                    .gte('data', inicio.isoformat()).lte('data', fim.isoformat())
                    .order('data').order('created_at')
                    .execute())
        return resposta.data or []
    except Exception as e:
        print(f"Erro ao carregar diário: {e}")
        return []


def buscar_produto_usuario(user_id: str, codigo: str) -> Optional[Dict]:
    client = _client()
    if not client:
        return None
    try:
        resposta = (client.table('produtos_usuario').select('*')
                    .eq('user_id', user_id).eq('codigo_barras', codigo).execute())
    except Exception as e:
        print(f"Erro ao buscar produto: {e}")
        return None
    if not resposta.data:
        return None
    p = resposta.data[0]
    return {
        'nome': p['nome'], 'porcao_g': None, 'porcao_texto': None, 'imagem': None,
        'por_100g': {chave: p.get(chave) for chave in alimentos.NUTRIENTES},
        'fonte': 'cadastro próprio',
    }


def salvar_produto_usuario(user_id: str, codigo: str, nome: str, por_100g: Dict) -> bool:
    client = _client()
    if not client:
        return False
    try:
        client.table('produtos_usuario').upsert(
            {'user_id': user_id, 'codigo_barras': codigo, 'nome': nome, **por_100g},
            on_conflict='user_id,codigo_barras',
        ).execute()
        return True
    except Exception as e:
        print(f"Erro ao salvar produto: {e}")
        return False


# ============================================================================
# ALERTAS
# Limites da Sociedade Brasileira de Diabetes (2023) e da American Diabetes
# Association; diretriz brasileira de hipertensão para pressão arterial.
# Uma medida isolada não fecha diagnóstico: os alertas orientam procurar o médico.
# ============================================================================

def alertas_glicemia(valor: float, momento: str) -> Optional[tuple]:
    """(nível, mensagem) ou None; nível: 'erro' | 'aviso'"""
    if valor < 70:
        return ('erro', f"Glicemia de {valor:.0f} mg/dL está **baixa** (hipoglicemia, abaixo de 70). "
                        "Se tiver tremor, suor frio ou confusão, coma algo com açúcar e procure atendimento.")
    if momento == "Jejum":
        if valor >= 126:
            return ('erro', f"Glicemia de jejum de {valor:.0f} mg/dL está na faixa **compatível com diabetes** "
                            "(126 ou mais). Procure um médico para confirmar com exames.")
        if valor >= 100:
            return ('aviso', f"Glicemia de jejum de {valor:.0f} mg/dL está **alterada** (100 a 125, faixa de "
                             "pré-diabetes). Vale conversar com um médico.")
    elif momento == "2 horas após refeição":
        if valor >= 200:
            return ('erro', f"Glicemia de {valor:.0f} mg/dL duas horas após comer está **muito alta** "
                            "(200 ou mais). Procure um médico.")
        if valor >= 140:
            return ('aviso', f"Glicemia de {valor:.0f} mg/dL duas horas após comer está **acima do ideal** "
                             "(140 a 199).")
    elif valor >= 200:
        return ('erro', f"Glicemia de {valor:.0f} mg/dL está **muito alta** (200 ou mais). Com sintomas como "
                        "sede e urina em excesso, é critério de diabetes: procure um médico.")
    return None


def alertas_pressao(sistolica: int, diastolica: int) -> Optional[tuple]:
    if sistolica >= 180 or diastolica >= 120:
        return ('erro', f"Pressão de {sistolica}/{diastolica} mmHg está **muito alta**. "
                        "Se tiver dor de cabeça forte, dor no peito ou falta de ar, procure atendimento de urgência.")
    if sistolica >= 140 or diastolica >= 90:
        return ('aviso', f"Pressão de {sistolica}/{diastolica} mmHg está **elevada** (140/90 ou mais). "
                         "Se repetir, procure um médico.")
    return None


def alertas_sintomas(sintomas: List[str]) -> Optional[tuple]:
    classicos = [s for s in sintomas if s in SINTOMAS_CLASSICOS]
    if len(classicos) >= 2:
        return ('erro', f"Você marcou {len(classicos)} sintomas clássicos de diabetes "
                        f"({', '.join(s.lower() for s in classicos)}). Procure um médico e peça exame de glicemia.")
    if sintomas:
        return ('aviso', "Os sintomas marcados podem ter várias causas. Se persistirem, converse com um médico.")
    return None


def _mostrar_alerta(alerta: Optional[tuple]):
    if not alerta:
        return
    nivel, mensagem = alerta
    (st.error if nivel == 'erro' else st.warning)(mensagem)


def alertas_do_dia(registros: List[Dict]) -> List[tuple]:
    alertas = []
    for r in registros:
        d = r['dados']
        if r['tipo'] == 'glicemia':
            alertas.append(alertas_glicemia(d['valor'], d['momento']))
        elif r['tipo'] == 'pressao':
            alertas.append(alertas_pressao(d['sistolica'], d['diastolica']))
        elif r['tipo'] == 'sintomas':
            alertas.append(alertas_sintomas(d['sintomas']))
    return [a for a in alertas if a]


# ============================================================================
# FORMULÁRIOS DE REGISTRO (paciente)
# ============================================================================

def _registrar(user_id: str, tipo: str, dia: date, dados: Dict, alerta: Optional[tuple] = None):
    if salvar_registro(user_id, tipo, dia, dados):
        st.session_state.diario_msg = ("Registro salvo!", alerta)
        # Nova versão dos campos de alimento: limpa busca, foto e código para o próximo item
        st.session_state.diario_versao = st.session_state.get('diario_versao', 0) + 1
        st.rerun()
    else:
        st.error("Não foi possível salvar. Tente novamente.")


def _form_atividade(user_id: str, dia: date):
    with st.form("form_atividade", clear_on_submit=True):
        tipo = st.selectbox("Atividade", ATIVIDADES)
        col1, col2 = st.columns(2)
        minutos = col1.number_input("Duração (minutos)", min_value=1, max_value=600, value=30, step=5)
        intensidade = col2.selectbox("Intensidade", INTENSIDADES, index=1,
                                     help="Moderada: dá para conversar, mas não cantar. Intensa: difícil falar.")
        if st.form_submit_button("Salvar atividade", type="primary", width="stretch"):
            _registrar(user_id, 'atividade', dia,
                       {'atividade': tipo, 'minutos': int(minutos), 'intensidade': intensidade})


def _form_glicemia(user_id: str, dia: date):
    with st.form("form_glicemia", clear_on_submit=True):
        col1, col2 = st.columns(2)
        valor = col1.number_input("Glicemia (mg/dL)", min_value=20, max_value=600, value=100)
        momento = col2.selectbox("Momento da medida", MOMENTOS_GLICEMIA)
        if st.form_submit_button("Salvar glicemia", type="primary", width="stretch"):
            _registrar(user_id, 'glicemia', dia, {'valor': int(valor), 'momento': momento},
                       alertas_glicemia(valor, momento))


def _form_peso(user_id: str, dia: date):
    altura_cm = st.session_state.get('perfil_altura') or 0
    with st.form("form_peso", clear_on_submit=True):
        peso = st.number_input("Peso (kg)", min_value=20.0, max_value=300.0,
                               value=float(st.session_state.get('perfil_peso') or 70.0), step=0.1)
        if st.form_submit_button("Salvar peso", type="primary", width="stretch"):
            dados = {'peso_kg': round(peso, 1)}
            if altura_cm:
                dados['imc'] = round(peso / ((altura_cm / 100) ** 2), 1)
            _registrar(user_id, 'peso', dia, dados)
    if not altura_cm:
        st.caption("Preencha sua altura em *Meu Perfil* para o IMC ser calculado.")


def _form_pressao(user_id: str, dia: date):
    with st.form("form_pressao", clear_on_submit=True):
        col1, col2 = st.columns(2)
        sistolica = col1.number_input("Sistólica (máxima)", min_value=60, max_value=260, value=120)
        diastolica = col2.number_input("Diastólica (mínima)", min_value=30, max_value=160, value=80)
        if st.form_submit_button("Salvar pressão", type="primary", width="stretch"):
            if diastolica >= sistolica:
                st.error("A diastólica (mínima) deve ser menor que a sistólica (máxima).")
            else:
                _registrar(user_id, 'pressao', dia, {'sistolica': int(sistolica), 'diastolica': int(diastolica)},
                           alertas_pressao(sistolica, diastolica))


def _form_sono(user_id: str, dia: date):
    with st.form("form_sono", clear_on_submit=True):
        horas = st.number_input("Horas dormidas na última noite", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        if st.form_submit_button("Salvar sono", type="primary", width="stretch"):
            _registrar(user_id, 'sono', dia, {'horas': float(horas)})


def _form_sintomas(user_id: str, dia: date):
    with st.form("form_sintomas", clear_on_submit=True):
        sintomas = st.multiselect("O que você sentiu hoje?", SINTOMAS_CLASSICOS + SINTOMAS_OUTROS,
                                  placeholder="Escolha um ou mais sintomas")
        if st.form_submit_button("Salvar sintomas", type="primary", width="stretch"):
            if not sintomas:
                st.error("Escolha pelo menos um sintoma.")
            else:
                _registrar(user_id, 'sintomas', dia, {'sintomas': sintomas}, alertas_sintomas(sintomas))


def _form_habitos(user_id: str, dia: date):
    with st.form("form_habitos", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        alcool = col1.number_input("Doses de álcool", min_value=0, max_value=50, value=0,
                                   help="1 dose = 1 lata de cerveja, 1 taça de vinho ou 1 dose de destilado")
        cigarros = col2.number_input("Cigarros", min_value=0, max_value=100, value=0)
        refrigerante = col3.number_input("Copos de refrigerante / suco adoçado", min_value=0, max_value=30, value=0)
        if st.form_submit_button("Salvar hábitos", type="primary", width="stretch"):
            _registrar(user_id, 'habitos', dia, {'alcool_doses': int(alcool), 'cigarros': int(cigarros),
                                                  'bebidas_acucaradas_copos': int(refrigerante)})


# ---------------------------------------------------------------------------
# Alimentação
# ---------------------------------------------------------------------------

def _tabela_nutrientes(valores: Dict, titulo: str):
    linhas = [
        ("Calorias", valores.get('kcal'), "kcal"),
        ("Carboidratos", valores.get('carboidratos_g'), "g"),
        ("Açúcares", valores.get('acucares_g'), "g"),
        ("Proteínas", valores.get('proteinas_g'), "g"),
        ("Gorduras", valores.get('gorduras_g'), "g"),
        ("Fibras", valores.get('fibra_g'), "g"),
        ("Sódio", valores.get('sodio_mg'), "mg"),
    ]
    df = pd.DataFrame(
        [(nome, "não informado" if v is None else f"{v:.1f} {unidade}") for nome, v, unidade in linhas],
        columns=["Nutriente", titulo],
    )
    st.dataframe(df, hide_index=True, width="stretch")


def _adicionar_alimento(user_id: str, dia: date, refeicao: str, nome: str, fonte: str,
                        por_100g: Dict, gramas: float, medida: str, codigo: Optional[str] = None):
    dados = {
        'refeicao': refeicao, 'alimento': nome, 'fonte': fonte, 'quantidade_g': round(gramas, 1),
        'medida': medida, **alimentos.calcular_porcao(por_100g, gramas),
        'incompleto': any(por_100g.get(k) is None for k in ('kcal', 'carboidratos_g')),
    }
    if codigo:
        dados['codigo_barras'] = codigo
    _registrar(user_id, 'refeicao', dia, dados)


def _quantidade_por_medida(chave: str) -> tuple:
    """Escolha de medida caseira + quantidade; devolve (gramas, descrição da medida)"""
    col1, col2 = st.columns([2, 1])
    medida = col1.selectbox("Medida", list(alimentos.MEDIDAS_CASEIRAS), index=3, key=f"{chave}_medida")
    peso_medida = alimentos.MEDIDAS_CASEIRAS[medida]
    if peso_medida == 1:
        gramas = col2.number_input("Quantidade (g ou ml)", min_value=1.0, max_value=3000.0, value=100.0,
                                   step=10.0, key=f"{chave}_gramas")
        return gramas, f"{gramas:.0f} g"
    qtd = col2.number_input("Quantidade", min_value=0.5, max_value=20.0, value=1.0, step=0.5, key=f"{chave}_qtd")
    gramas = qtd * peso_medida
    st.caption(f"≈ {gramas:.0f} g (peso típico da medida, segundo o IBGE; para mais precisão, use gramas)")
    return gramas, f"{qtd:g} × {medida.lower()}"


def _alimento_taco(user_id: str, dia: date, refeicao: str):
    v = st.session_state.get('diario_versao', 0)
    termo = st.text_input("Buscar alimento", placeholder="Ex.: arroz cozido, feijão, banana, pão francês",
                          key=f"taco_busca_{v}")
    if not termo:
        st.caption("Tabela Brasileira de Composição de Alimentos (TACO/Unicamp), com 597 alimentos.")
        return
    resultados = alimentos.buscar_taco(termo)
    if resultados.empty:
        st.info("Nenhum alimento encontrado. Tente uma palavra só (ex.: \"feijão\") ou sem acento.")
        return

    opcoes = {int(r['id']): f"{r['descricao']} ({r['kcal']:.0f} kcal/100 g)" if pd.notna(r['kcal'])
              else r['descricao'] for _, r in resultados.iterrows()}
    escolhido = st.selectbox("Alimento", list(opcoes), format_func=opcoes.get, key="taco_alimento")
    linha = resultados[resultados['id'] == escolhido].iloc[0]
    por_100g = alimentos.nutrientes_taco(linha)

    gramas, medida = _quantidade_por_medida("taco")
    porcao = alimentos.calcular_porcao(por_100g, gramas)
    st.markdown(f"**{porcao['kcal']:.0f} kcal** · **{porcao['carboidratos_g']:.1f} g de carboidratos** · "
                f"{porcao['proteinas_g']:.1f} g de proteínas · {porcao['gorduras_g']:.1f} g de gorduras")
    if por_100g['carboidratos_g'] is None or por_100g['kcal'] is None:
        st.caption("A TACO não tem todos os valores deste alimento; os que faltam contam como zero.")

    if st.button("Adicionar ao diário", type="primary", width="stretch", key="taco_adicionar"):
        _adicionar_alimento(user_id, dia, refeicao, linha['descricao'], 'TACO', por_100g, gramas, medida)


def _alimento_codigo_barras(user_id: str, dia: date, refeicao: str):
    st.caption("No celular, toque em *Browse files* e escolha **Câmera**. Enquadre só o código de barras, "
               "com boa luz e sem reflexo.")
    v = st.session_state.get('diario_versao', 0)
    foto = st.file_uploader("Foto do código de barras", type=["jpg", "jpeg", "png", "webp"],
                            key=f"cb_foto_{v}", label_visibility="collapsed")
    digitado = st.text_input("Ou digite o número do código de barras", max_chars=14, key=f"cb_digitado_{v}",
                             placeholder="Ex.: 7894900011517")

    codigo = None
    if foto is not None:
        try:
            codigo = alimentos.ler_codigo_barras(foto.getvalue())
        except Exception as e:
            print(f"Erro ao ler código de barras: {e}")
        if not codigo:
            st.warning("Não consegui ler o código nesta foto. Tente outra foto mais próxima e nítida, "
                       "ou digite o número que aparece embaixo das barras.")
            return
        st.success(f"Código lido: **{codigo}**")
    elif digitado:
        codigo = digitado.strip()
        if not alimentos.codigo_valido(codigo):
            st.error("O código de barras deve ter 8, 12, 13 ou 14 números.")
            return
    else:
        return

    produto = buscar_produto_usuario(user_id, codigo)
    if produto is None:
        try:
            with st.spinner("Buscando produto..."):
                produto = alimentos.buscar_open_food_facts(codigo)
            if produto:
                produto['fonte'] = 'Open Food Facts'
        except Exception as e:
            print(f"Erro no Open Food Facts: {e}")
            st.warning("O serviço de produtos está indisponível agora. Você pode digitar os valores do rótulo abaixo.")
            produto = None

    if produto is None:
        _cadastrar_produto(user_id, dia, refeicao, codigo)
        return

    col_img, col_info = st.columns([1, 3])
    if produto.get('imagem'):
        col_img.image(produto['imagem'], width=90)
    with col_info:
        st.markdown(f"**{produto['nome']}**")
        st.caption(f"Fonte: {produto['fonte']}. Confira com o rótulo antes de salvar.")
    _tabela_nutrientes(produto['por_100g'], "Por 100 g / 100 ml")

    porcao_g = produto.get('porcao_g')
    padrao = float(porcao_g) if porcao_g else 100.0
    ajuda = f"Porção do rótulo: {produto['porcao_texto']}" if produto.get('porcao_texto') else None
    gramas = st.number_input("Quanto você consumiu (g ou ml)?", min_value=1.0, max_value=5000.0,
                             value=padrao, step=10.0, help=ajuda, key=f"cb_gramas_{codigo}")
    porcao = alimentos.calcular_porcao(produto['por_100g'], gramas)
    st.markdown(f"**{porcao['kcal']:.0f} kcal** · **{porcao['carboidratos_g']:.1f} g de carboidratos** · "
                f"{porcao['acucares_g']:.1f} g de açúcares")

    if st.button("Adicionar ao diário", type="primary", width="stretch", key="cb_adicionar"):
        _adicionar_alimento(user_id, dia, refeicao, produto['nome'], produto['fonte'], produto['por_100g'],
                            gramas, f"{gramas:.0f} g/ml", codigo)


def _cadastrar_produto(user_id: str, dia: date, refeicao: str, codigo: str):
    st.info(f"Produto **{codigo}** não encontrado. Digite os valores da tabela nutricional do rótulo, "
            "na coluna **100 g** (ou 100 ml). Ele fica salvo para a próxima vez.")
    with st.form(f"form_produto_{codigo}"):
        nome = st.text_input("Nome do produto", placeholder="Ex.: Leite integral Marca X")
        col1, col2 = st.columns(2)
        kcal = col1.number_input("Valor energético (kcal)", min_value=0.0, max_value=900.0, step=1.0)
        carboidratos = col2.number_input("Carboidratos (g)", min_value=0.0, max_value=100.0, step=0.1)
        acucares = col1.number_input("Açúcares totais (g)", min_value=0.0, max_value=100.0, step=0.1)
        proteinas = col2.number_input("Proteínas (g)", min_value=0.0, max_value=100.0, step=0.1)
        gorduras = col1.number_input("Gorduras totais (g)", min_value=0.0, max_value=100.0, step=0.1)
        fibras = col2.number_input("Fibra alimentar (g)", min_value=0.0, max_value=100.0, step=0.1)
        sodio = col1.number_input("Sódio (mg)", min_value=0.0, max_value=40000.0, step=1.0)
        gramas = col2.number_input("Quanto você consumiu (g ou ml)?", min_value=1.0, max_value=5000.0,
                                   value=100.0, step=10.0)
        if st.form_submit_button("Salvar produto e adicionar ao diário", type="primary", width="stretch"):
            if not nome.strip():
                st.error("Informe o nome do produto.")
                return
            if carboidratos + proteinas + gorduras > 100:
                st.error("Carboidratos + proteínas + gorduras não podem passar de 100 g em 100 g. "
                         "Confira se usou a coluna de 100 g, e não a da porção.")
                return
            por_100g = {'kcal': kcal, 'carboidratos_g': carboidratos, 'acucares_g': acucares,
                        'proteinas_g': proteinas, 'gorduras_g': gorduras, 'fibra_g': fibras, 'sodio_mg': sodio}
            if not salvar_produto_usuario(user_id, codigo, nome.strip(), por_100g):
                st.error("Não foi possível salvar o produto. Tente novamente.")
                return
            _adicionar_alimento(user_id, dia, refeicao, nome.strip(), 'cadastro próprio', por_100g,
                                gramas, f"{gramas:.0f} g/ml", codigo)


def _form_refeicao(user_id: str, dia: date):
    refeicao = st.selectbox("Refeição", REFEICOES, key="refeicao_escolhida")
    modo = st.radio("Como adicionar", ["Buscar alimento", "Código de barras"], horizontal=True,
                    key="refeicao_modo",
                    captions=["Arroz, feijão, frutas, carnes…", "Produtos industrializados"])
    if modo == "Buscar alimento":
        _alimento_taco(user_id, dia, refeicao)
    else:
        _alimento_codigo_barras(user_id, dia, refeicao)


FORMULARIOS = {
    'refeicao': _form_refeicao,
    'atividade': _form_atividade,
    'glicemia': _form_glicemia,
    'peso': _form_peso,
    'pressao': _form_pressao,
    'sono': _form_sono,
    'sintomas': _form_sintomas,
    'habitos': _form_habitos,
}


# ============================================================================
# RESUMO DO DIA
# ============================================================================

def _descrever(registro: Dict) -> str:
    d, tipo = registro['dados'], registro['tipo']
    if tipo == 'refeicao':
        texto = (f"{d['alimento']} ({d['medida']}): {d['kcal']:.0f} kcal, "
                 f"{d['carboidratos_g']:.1f} g carboidratos")
        return texto + (r" \*" if d.get('incompleto') else "")
    if tipo == 'atividade':
        return f"{d['atividade']}: {d['minutos']} min, intensidade {d['intensidade'].lower()}"
    if tipo == 'glicemia':
        return f"Glicemia {d['valor']} mg/dL ({d['momento'].lower()})"
    if tipo == 'peso':
        return f"Peso {d['peso_kg']:.1f} kg" + (f" (IMC {d['imc']:.1f})" if d.get('imc') else "")
    if tipo == 'pressao':
        return f"Pressão {d['sistolica']}/{d['diastolica']} mmHg"
    if tipo == 'sono':
        return f"Sono: {d['horas']:g} h"
    if tipo == 'sintomas':
        return "Sintomas: " + ", ".join(s.lower() for s in d['sintomas'])
    if tipo == 'habitos':
        partes = []
        if d.get('alcool_doses'):
            partes.append(f"{d['alcool_doses']} dose(s) de álcool")
        if d.get('cigarros'):
            partes.append(f"{d['cigarros']} cigarro(s)")
        if d.get('bebidas_acucaradas_copos'):
            partes.append(f"{d['bebidas_acucaradas_copos']} copo(s) de bebida adoçada")
        return "Hábitos: " + (", ".join(partes) if partes else "nenhum álcool, cigarro ou bebida adoçada")
    return tipo


def _resumo_dia(registros: List[Dict], somente_leitura: bool):
    if not registros:
        st.info("Nenhum registro neste dia.")
        return

    refeicoes = [r['dados'] for r in registros if r['tipo'] == 'refeicao']
    atividade_min = sum(r['dados']['minutos'] for r in registros if r['tipo'] == 'atividade')
    glicemias = [r['dados'] for r in registros if r['tipo'] == 'glicemia']
    sono = [r['dados']['horas'] for r in registros if r['tipo'] == 'sono']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Calorias", f"{sum(d['kcal'] for d in refeicoes):.0f} kcal" if refeicoes else "—")
    col2.metric("Carboidratos", f"{sum(d['carboidratos_g'] for d in refeicoes):.0f} g" if refeicoes else "—")
    col3.metric("Atividade física", f"{atividade_min} min" if atividade_min else "—")
    col4.metric("Última glicemia", f"{glicemias[-1]['valor']} mg/dL" if glicemias else "—",
                help=glicemias[-1]['momento'] if glicemias else None)
    if sono:
        st.caption(f"Sono: {sono[-1]:g} h")

    for alerta in alertas_do_dia(registros):
        _mostrar_alerta(alerta)

    if refeicoes:
        st.markdown("##### Alimentação")
        for refeicao in REFEICOES:
            itens = [r for r in registros if r['tipo'] == 'refeicao' and r['dados']['refeicao'] == refeicao]
            if not itens:
                continue
            total = sum(r['dados']['kcal'] for r in itens)
            carbo = sum(r['dados']['carboidratos_g'] for r in itens)
            st.markdown(f"**{refeicao}**: {total:.0f} kcal, {carbo:.0f} g de carboidratos")
            _lista_registros(itens, somente_leitura)
        if any(d.get('incompleto') for d in refeicoes):
            st.caption("\\* Alimento sem todos os valores na tabela; os que faltam contam como zero.")

    outros = [r for r in registros if r['tipo'] != 'refeicao']
    if outros:
        st.markdown("##### Outros registros")
        _lista_registros(outros, somente_leitura)


def _lista_registros(registros: List[Dict], somente_leitura: bool):
    for r in registros:
        if somente_leitura:
            st.markdown(f"- {_descrever(r)}")
            continue
        col_texto, col_botao = st.columns([6, 1], vertical_alignment="center")
        col_texto.markdown(f"- {_descrever(r)}")
        if col_botao.button("✕", key=f"excluir_{r['id']}", help="Excluir este registro"):
            if excluir_registro(r['id']):
                st.rerun()
            else:
                st.error("Não foi possível excluir.")


# ============================================================================
# EVOLUÇÃO (gráficos)
# ============================================================================

def _grafico(df: pd.DataFrame, y: str, titulo_y: str, tipo: str = 'barra',
             referencia: Optional[float] = None, texto_referencia: str = "", tooltip_extra: List = None):
    base = alt.Chart(df).encode(
        x=alt.X('data:T', title=None, axis=alt.Axis(format='%d/%m', labelAngle=0, tickCount=7)),
        y=alt.Y(f'{y}:Q', title=titulo_y),
        tooltip=[alt.Tooltip('data:T', title='Data', format='%d/%m/%Y'),
                 alt.Tooltip(f'{y}:Q', title=titulo_y, format='.1f')] + (tooltip_extra or []),
    )
    if tipo == 'barra':
        marca = base.mark_bar(color=COR_DESTAQUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=14)
    elif tipo == 'pontos':
        marca = base.mark_circle(color=COR_DESTAQUE, size=90, opacity=1)
    else:
        marca = base.mark_line(color=COR_DESTAQUE, strokeWidth=2, point=alt.OverlayMarkDef(size=60))
    camadas = [marca]
    if referencia is not None:
        regra = alt.Chart(pd.DataFrame({'ref': [referencia]})).mark_rule(
            color=COR_REFERENCIA, strokeDash=[4, 4]).encode(y='ref:Q')
        rotulo = alt.Chart(pd.DataFrame({'ref': [referencia], 'texto': [texto_referencia]})).mark_text(
            align='left', dx=4, dy=-6, color=COR_REFERENCIA, fontSize=11).encode(
            y='ref:Q', x=alt.value(0), text='texto:N')
        camadas += [regra, rotulo]
    st.altair_chart(alt.layer(*camadas).properties(height=220), width="stretch")


def _evolucao(registros: List[Dict], dias: int):
    if not registros:
        st.info(f"Nenhum registro nos últimos {dias} dias.")
        return

    def tabela(tipo):
        linhas = [{'data': pd.to_datetime(r['data']), **r['dados']} for r in registros if r['tipo'] == tipo]
        return pd.DataFrame(linhas)

    dia_atual = hoje()
    atividade = tabela('atividade')
    st.markdown("##### Atividade física")
    if atividade.empty:
        st.caption("Sem registros de atividade.")
    else:
        semana = atividade[atividade['data'] >= pd.Timestamp(dia_atual - timedelta(days=6))]['minutos'].sum()
        st.metric("Últimos 7 dias", f"{semana:.0f} de 150 min",
                  help="A OMS recomenda pelo menos 150 minutos de atividade moderada por semana.")
        st.progress(min(semana / 150, 1.0))
        por_dia = atividade.groupby('data', as_index=False)['minutos'].sum()
        _grafico(por_dia, 'minutos', 'Minutos por dia')

    refeicoes = tabela('refeicao')
    if not refeicoes.empty:
        por_dia = refeicoes.groupby('data', as_index=False)[['kcal', 'carboidratos_g']].sum()
        st.markdown("##### Carboidratos por dia")
        _grafico(por_dia, 'carboidratos_g', 'Carboidratos (g)')
        st.markdown("##### Calorias por dia")
        _grafico(por_dia, 'kcal', 'Calorias (kcal)')
        st.caption("Só aparecem os dias com alimentação registrada.")

    glicemia = tabela('glicemia')
    if not glicemia.empty:
        st.markdown("##### Glicemia de jejum")
        jejum = glicemia[glicemia['momento'] == 'Jejum']
        if jejum.empty:
            st.caption("Sem medidas em jejum.")
        else:
            _grafico(jejum, 'valor', 'mg/dL', tipo='pontos', referencia=126,
                     texto_referencia="126: faixa de diabetes")
        outras = glicemia[glicemia['momento'] != 'Jejum']
        if not outras.empty:
            with st.expander("Outras medidas de glicemia"):
                st.dataframe(outras.assign(data=outras['data'].dt.strftime('%d/%m/%Y'))
                             .rename(columns={'data': 'Data', 'valor': 'mg/dL', 'momento': 'Momento'}),
                             hide_index=True, width="stretch")

    peso = tabela('peso')
    if not peso.empty:
        st.markdown("##### Peso")
        _grafico(peso.groupby('data', as_index=False)['peso_kg'].last(), 'peso_kg', 'kg', tipo='linha')

    sono = tabela('sono')
    if not sono.empty:
        st.markdown("##### Sono")
        _grafico(sono.groupby('data', as_index=False)['horas'].last(), 'horas', 'Horas',
                 referencia=7, texto_referencia="7 h: mínimo recomendado")

    pressao = tabela('pressao')
    if not pressao.empty:
        st.markdown("##### Pressão arterial")
        st.dataframe(pressao.assign(data=pressao['data'].dt.strftime('%d/%m/%Y'),
                                    pressao=pressao['sistolica'].astype(str) + '/' + pressao['diastolica'].astype(str))
                     [['data', 'pressao']].rename(columns={'data': 'Data', 'pressao': 'mmHg'}),
                     hide_index=True, width="stretch")

    sintomas = tabela('sintomas')
    if not sintomas.empty:
        st.markdown("##### Sintomas mais frequentes")
        contagem = sintomas.explode('sintomas')['sintomas'].value_counts()
        st.dataframe(contagem.rename_axis('Sintoma').reset_index(name='Dias'), hide_index=True, width="stretch")


# ============================================================================
# TELA PRINCIPAL
# ============================================================================

def mostrar_diario(user_id: str, somente_leitura: bool = False):
    """Diário completo; com somente_leitura=True (visão do médico) não há registro nem exclusão"""
    if 'diario_msg' in st.session_state:
        mensagem, alerta = st.session_state.pop('diario_msg')
        st.success(mensagem)
        _mostrar_alerta(alerta)

    abas = ["Resumo do dia", "Evolução"] if somente_leitura else ["Registrar", "Resumo do dia", "Evolução"]
    tabs = dict(zip(abas, st.tabs(abas)))
    prefixo = f"diario_{'med' if somente_leitura else 'pac'}"

    if not somente_leitura:
        with tabs["Registrar"]:
            dia = st.date_input("Dia", value=hoje(), max_value=hoje(), format="DD/MM/YYYY",
                                key=f"{prefixo}_dia_registro")
            categoria = st.selectbox("O que você quer registrar?", list(CATEGORIAS),
                                     format_func=CATEGORIAS.get, key=f"{prefixo}_categoria")
            FORMULARIOS[categoria](user_id, dia)

    with tabs["Resumo do dia"]:
        dia_resumo = st.date_input("Dia", value=hoje(), max_value=hoje(), format="DD/MM/YYYY",
                                   key=f"{prefixo}_dia_resumo")
        _resumo_dia(carregar_registros(user_id, dia_resumo, dia_resumo), somente_leitura)

    with tabs["Evolução"]:
        dias = st.segmented_control("Período", [7, 30, 90], default=30, format_func=lambda d: f"{d} dias",
                                    key=f"{prefixo}_periodo") or 30
        _evolucao(carregar_registros(user_id, hoje() - timedelta(days=dias - 1), hoje()), dias)
