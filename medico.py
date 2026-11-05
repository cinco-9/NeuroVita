# -*- coding: utf-8 -*-
"""
ACESSO DO MÉDICO
- Paciente: compartilha seus dados com um médico (pelo email da conta de médico) e revoga quando quiser
- Médico: vê, só para leitura, uma lista de triagem dos pacientes que compartilharam e a ficha clínica
  de cada um (indicadores, achados fora da faixa, evolução, avaliações de risco e registros)
As regras de acesso ficam no banco (diario_medico.sql); o app só chama as funções.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st

import diario
from supabase_db import db as supabase_db

UFS = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA", "PB", "PE",
       "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"]


def _rpc(funcao: str, params: Dict = None) -> List[Dict]:
    resposta = supabase_db.client.rpc(funcao, params or {}).execute()
    return resposta.data or []


def _remover_compartilhamento(paciente_id: str, medico_id: str) -> bool:
    try:
        (supabase_db.client.table('compartilhamentos').delete()
         .eq('paciente_id', paciente_id).eq('medico_id', medico_id).execute())
        return True
    except Exception as e:
        print(f"Erro ao remover compartilhamento: {e}")
        return False


def _data_br(valor: str) -> str:
    return datetime.fromisoformat(valor.replace('Z', '+00:00')).strftime('%d/%m/%Y')


def _descrever_crm(crm: str, uf: str) -> str:
    return f"CRM {crm}/{uf}" if crm and uf else (f"CRM {crm}" if crm else "CRM não informado")


# ============================================================================
# PACIENTE
# ============================================================================

def secao_compartilhamento(user_id: str):
    st.markdown("### Compartilhar com meu médico")
    st.caption("O médico verá, **só para leitura**, seu perfil, suas avaliações de risco e seu diário. "
               "Você pode parar de compartilhar a qualquer momento.")

    if not supabase_db.conectado or not supabase_db.client:
        st.warning("Banco de dados indisponível.")
        return

    try:
        medicos = _rpc('meus_medicos')
    except Exception as e:
        print(f"Erro ao listar médicos: {e}")
        st.warning("Não foi possível carregar seus médicos agora.")
        return

    for m in medicos:
        col_info, col_botao = st.columns([4, 1], vertical_alignment="center")
        col_info.markdown(f"**{m.get('nome') or m['email']}** · {_descrever_crm(m.get('crm'), m.get('crm_uf'))}  \n"
                          f"{m['email']} · desde {_data_br(m['desde'])}")
        if col_botao.button("Parar", key=f"revogar_{m['medico_id']}", help="Parar de compartilhar com este médico"):
            if _remover_compartilhamento(user_id, m['medico_id']):
                st.rerun()
            else:
                st.error("Não foi possível remover. Tente novamente.")

    with st.form("form_compartilhar", clear_on_submit=True):
        email = st.text_input("Email da conta do médico no app", placeholder="medico@exemplo.com")
        st.caption("O médico precisa ter criado uma conta do tipo **Médico** no app. "
                   "O app não verifica o CRM: compartilhe só com médicos que você conhece.")
        if st.form_submit_button("Compartilhar", type="primary", width="stretch"):
            if not email.strip():
                st.error("Informe o email do médico.")
            else:
                try:
                    nome = supabase_db.client.rpc('compartilhar_com_medico', {'email_medico': email.strip()}).execute().data
                    st.session_state.compartilhamento_msg = f"Dados compartilhados com {nome}."
                    st.rerun()
                except Exception as e:
                    if 'medico_nao_encontrado' in str(e):
                        st.error("Nenhuma conta de médico encontrada com este email.")
                    else:
                        print(f"Erro ao compartilhar: {e}")
                        st.error("Não foi possível compartilhar. Tente novamente.")

    if 'compartilhamento_msg' in st.session_state:
        st.success(st.session_state.pop('compartilhamento_msg'))


# ============================================================================
# MÉDICO
# Visão própria, diferente da tela do paciente: uma lista de triagem com todos os
# pacientes e, para cada um, uma ficha clínica com os indicadores do período, os
# achados que pedem atenção, a evolução das medidas e as avaliações de risco.
# ============================================================================

# Duas séries na mesma escala (pressão sistólica e diastólica); par validado contra a
# superfície escura (#22272B) pelo validador de paleta: CVD ΔE 10,2, visão normal 18,1
COR_SERIE_1 = "#22A6B3"
COR_SERIE_2 = "#8C7AE6"
COR_REFERENCIA = "#8C9BAB"
TEXTO_EIXO = "#9FADBC"


def _lozenge(texto: str, tipo: str) -> str:
    return f'<span class="lozenge {tipo}">{texto}</span>'


def _num(valor, casas: int = 1) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def _carregar_perfil(paciente_id: str) -> Dict:
    try:
        resposta = supabase_db.client.table('user_profiles').select('*').eq('id', paciente_id).execute()
        return resposta.data[0] if resposta.data else {}
    except Exception as e:
        print(f"Erro ao carregar perfil do paciente: {e}")
        return {}


def _achados(registros: List[Dict]) -> List[Dict]:
    """Medidas fora da faixa no período, com a classificação em linguagem clínica"""
    achados = []
    for r in registros:
        d, tipo, dia = r['dados'], r['tipo'], r['data']
        if tipo == 'glicemia':
            valor, momento = d['valor'], d['momento']
            if valor < 70:
                achados.append({'data': dia, 'grave': True, 'texto': f"Hipoglicemia: {valor} mg/dL ({momento.lower()})"})
            elif momento == "Jejum" and valor >= 126:
                achados.append({'data': dia, 'grave': True, 'texto': f"Glicemia de jejum {valor} mg/dL (≥ 126)"})
            elif momento == "Jejum" and valor >= 100:
                achados.append({'data': dia, 'grave': False, 'texto': f"Glicemia de jejum alterada: {valor} mg/dL (100-125)"})
            elif momento == "2 horas após refeição" and valor >= 140:
                achados.append({'data': dia, 'grave': valor >= 200,
                                'texto': f"Glicemia pós-prandial 2 h: {valor} mg/dL ({'≥ 200' if valor >= 200 else '140-199'})"})
            elif momento != "Jejum" and valor >= 200:
                achados.append({'data': dia, 'grave': True, 'texto': f"Glicemia casual {valor} mg/dL (≥ 200)"})
        elif tipo == 'pressao':
            pas, pad = d['sistolica'], d['diastolica']
            if pas >= 180 or pad >= 120:
                achados.append({'data': dia, 'grave': True, 'texto': f"PA {pas}/{pad} mmHg (≥ 180/120)"})
            elif pas >= 140 or pad >= 90:
                achados.append({'data': dia, 'grave': False, 'texto': f"PA elevada: {pas}/{pad} mmHg (≥ 140/90)"})
        elif tipo == 'sintomas':
            classicos = [s for s in d['sintomas'] if s in diario.SINTOMAS_CLASSICOS]
            if classicos:
                achados.append({'data': dia, 'grave': len(classicos) >= 2,
                                'texto': "Sintomas clássicos: " + ", ".join(s.lower() for s in classicos)})
    return sorted(achados, key=lambda a: a['data'], reverse=True)


def _indicadores(registros: List[Dict]) -> Dict:
    """Resumo numérico do período a partir dos registros do diário"""
    def de(tipo):
        return [r for r in registros if r['tipo'] == tipo]

    jejum = [r['dados']['valor'] for r in de('glicemia') if r['dados']['momento'] == "Jejum"]
    pressoes = [(r['dados']['sistolica'], r['dados']['diastolica']) for r in de('pressao')]
    pesos = sorted(((r['data'], r['dados']['peso_kg']) for r in de('peso')))
    inicio_semana = (diario.hoje() - timedelta(days=6)).isoformat()
    atividade_semana = sum(r['dados']['minutos'] for r in de('atividade') if r['data'] >= inicio_semana)
    dias_com_registro = len({r['data'] for r in registros})
    return {
        'jejum_media': sum(jejum) / len(jejum) if jejum else None, 'jejum_n': len(jejum),
        'pa_media': (sum(p[0] for p in pressoes) / len(pressoes), sum(p[1] for p in pressoes) / len(pressoes))
        if pressoes else None, 'pa_n': len(pressoes),
        'peso_variacao': (pesos[-1][1] - pesos[0][1]) if len(pesos) >= 2 else None,
        'peso_atual': pesos[-1][1] if pesos else None,
        'atividade_semana': atividade_semana, 'dias_com_registro': dias_com_registro,
        'ultimo_registro': max((r['data'] for r in registros), default=None),
    }


def _ultima_avaliacao(predicoes: List[Dict]) -> Optional[Dict]:
    return predicoes[0] if predicoes else None


def _valores_informados(dados) -> str:
    """dados_input pode chegar como dict ou como a string JSON gravada no banco"""
    if isinstance(dados, str):
        try:
            dados = json.loads(dados)
        except ValueError:
            return dados
    return ", ".join(f"{k}: {v}" for k, v in (dados or {}).items())


def _avaliacoes_paciente(paciente_id: str):
    predicoes = supabase_db.obter_ultimas_predicoes(limite=50, user_id=paciente_id)
    if not predicoes:
        st.info("O paciente ainda não fez avaliações de risco.")
        return
    limiares = {}
    for tipo in ('clinico', 'comportamental'):
        with open(f'modelo_{tipo}_meta.json', encoding='utf-8') as f:
            limiares[tipo] = json.load(f)['threshold']
    tabela = pd.DataFrame([{
        'Data': _data_br(p['created_at']),
        'Modelo': 'Random Forest (Pima)' if p['tipo_modelo'] == 'clinico' else 'XGBoost (BRFSS)',
        'Probabilidade': round(p['probabilidade'], 3),
        'Limiar': limiares[p['tipo_modelo']],
        'Resultado': 'Alto risco' if p['resultado'] == 1 else 'Baixo risco',
        'Valores informados': _valores_informados(p.get('dados_input')),
    } for p in predicoes])
    st.dataframe(tabela, hide_index=True, width="stretch",
                 column_config={'Probabilidade': st.column_config.ProgressColumn(
                     'Probabilidade', min_value=0.0, max_value=1.0, format="%.3f")})
    st.caption("Random Forest: AUC 0,839 e F1 0,69 em validação cruzada aninhada (Pima Indians, 768 pacientes). "
               "XGBoost: AUC 0,821 e F1 0,46 no teste (BRFSS 2015, pré-diabetes e diabetes agrupados). "
               "O limiar de cada modelo foi escolhido para maximizar o F1; a probabilidade não é calibrada como risco absoluto.")



# ---------------------------------------------------------------- triagem ----

def _linha_triagem(paciente: Dict, inicio) -> Dict:
    pid = paciente['paciente_id']
    perfil = _carregar_perfil(pid)
    predicoes = supabase_db.obter_ultimas_predicoes(limite=1, user_id=pid)
    registros = diario.carregar_registros(pid, inicio, diario.hoje())
    ind = _indicadores(registros)
    achados = _achados(registros)
    ultima = _ultima_avaliacao(predicoes)
    graves = sum(a['grave'] for a in achados)
    # Prioridade: achados graves, depois risco elevado, depois outros achados
    prioridade = graves * 100 + (50 if ultima and ultima['resultado'] == 1 else 0) + (len(achados) - graves)
    return {
        'id': pid,
        'Paciente': paciente.get('nome') or paciente['email'],
        'Idade': perfil.get('idade'),
        'IMC': round(perfil['imc'], 1) if perfil.get('imc') else None,
        'Último risco': round(ultima['probabilidade'], 3) if ultima else None,
        'Resultado': ('Elevado' if ultima['resultado'] == 1 else 'Baixo') if ultima else '—',
        'Glicemia jejum (média)': round(ind['jejum_media']) if ind['jejum_media'] else None,
        'PA média': f"{ind['pa_media'][0]:.0f}/{ind['pa_media'][1]:.0f}" if ind['pa_media'] else '—',
        'Achados': len(achados),
        'Último registro': _data_iso_br(ind['ultimo_registro']) if ind['ultimo_registro'] else '—',
        '_prioridade': prioridade,
    }


def _data_iso_br(valor: str) -> str:
    return datetime.fromisoformat(valor[:10]).strftime('%d/%m/%Y')


def _triagem(pacientes: List[Dict], inicio, dias: int) -> Optional[str]:
    """Tabela com todos os pacientes, ordenada por prioridade; devolve o id do paciente clicado"""
    with st.spinner("Carregando pacientes..."):
        linhas = sorted((_linha_triagem(p, inicio) for p in pacientes), key=lambda l: -l['_prioridade'])
    tabela = pd.DataFrame(linhas)
    com_achado = int((tabela['Achados'] > 0).sum())
    elevado = int((tabela['Resultado'] == 'Elevado').sum())

    st.markdown(
        f'<div class="tecnico">'
        f'<div><span>Pacientes</span><b>{len(tabela)}</b></div>'
        f'<div><span>Com achados nos últimos {dias} dias</span><b>{com_achado}</b></div>'
        f'<div><span>Última avaliação com risco elevado</span><b>{elevado}</b></div>'
        f'</div>', unsafe_allow_html=True)

    st.caption("Ordenada por prioridade: achados graves, risco elevado e demais achados. "
               "Clique numa linha para abrir a ficha do paciente.")
    evento = st.dataframe(
        tabela.drop(columns=['id', '_prioridade']), hide_index=True, width="stretch",
        on_select="rerun", selection_mode="single-row", key="triagem_pacientes",
        column_config={
            'Último risco': st.column_config.ProgressColumn('Último risco', min_value=0.0, max_value=1.0, format="%.3f"),
            'IMC': st.column_config.NumberColumn('IMC', format="%.1f"),
            'Glicemia jejum (média)': st.column_config.NumberColumn('Glicemia jejum (média)', format="%d mg/dL"),
            'Achados': st.column_config.NumberColumn('Achados', help=f"Medidas fora da faixa nos últimos {dias} dias"),
        })
    selecionadas = evento.selection.rows if evento and evento.selection else []
    return linhas[selecionadas[0]]['id'] if selecionadas else None


# ------------------------------------------------------------------ ficha ----

def _cabecalho_ficha(paciente: Dict, perfil: Dict):
    imc = perfil.get('imc')
    if imc:
        faixa = ("Obesidade", "danger") if imc >= 30 else ("Sobrepeso", "warning") if imc >= 25 else \
            ("Normal", "success") if imc >= 18.5 else ("Baixo peso", "warning")
        imc_txt = f"{_num(imc)} {_lozenge(*faixa)}"
    else:
        imc_txt = "—"
    dados = [
        ("Idade", f"{perfil['idade']} anos" if perfil.get('idade') else "—"),
        ("Sexo", perfil.get('sexo') or "—"),
        ("Peso / altura", f"{_num(perfil['peso'])} kg · {_num(perfil['altura'] / 100, 2)} m"
         if perfil.get('peso') and perfil.get('altura') else "—"),
        ("IMC", imc_txt),
        ("Diabetes na família", "Sim" if perfil.get('hist_familiar_diabetes') else "Não"),
        ("Compartilha desde", _data_br(paciente['desde'])),
    ]
    celulas = "".join(f'<div><span>{rotulo}</span><b>{valor}</b></div>' for rotulo, valor in dados)
    nome = paciente.get('nome') or paciente['email']
    st.markdown(
        f'<div class="card">'
        f'<div class="card-row"><span class="card-title" style="font-size: 18px">{nome}</span>'
        f'<span class="card-note">{paciente["email"]}</span></div>'
        f'<div class="tecnico">{celulas}</div>'
        f'<div class="card-row" style="align-items: flex-start; gap: 24px">'
        f'<div style="flex: 1"><div class="card-note">Medicações em uso</div><div class="card-text">{perfil.get("medicacoes") or "Nenhuma informada"}</div></div>'
        f'<div style="flex: 1"><div class="card-note">Alergias</div><div class="card-text">{perfil.get("alergias") or "Nenhuma informada"}</div></div>'
        f'</div></div>', unsafe_allow_html=True)


def _cartoes_indicadores(ind: Dict, predicoes: List[Dict], dias: int):
    ultima = _ultima_avaliacao(predicoes)

    def status_jejum(v):
        if v is None:
            return _lozenge("Sem medidas", "neutral")
        return _lozenge("≥ 126", "danger") if v >= 126 else _lozenge("100-125", "warning") if v >= 100 \
            else _lozenge("Normal", "success")

    def status_pa(pa):
        if pa is None:
            return _lozenge("Sem medidas", "neutral")
        return _lozenge("Elevada", "warning") if pa[0] >= 140 or pa[1] >= 90 else _lozenge("Normal", "success")

    cartoes = [
        ("Última avaliação de risco",
         f"{_num(ultima['probabilidade'] * 100)}%" if ultima else "—",
         (_lozenge("Risco elevado", "danger") if ultima['resultado'] == 1 else _lozenge("Risco baixo", "success"))
         if ultima else _lozenge("Nenhuma", "neutral"),
         (("Exames · " if ultima['tipo_modelo'] == 'clinico' else "Questionário · ") + _data_br(ultima['created_at']))
         if ultima else ""),
        (f"Glicemia de jejum ({dias} dias)",
         f"{ind['jejum_media']:.0f} mg/dL" if ind['jejum_media'] else "—",
         status_jejum(ind['jejum_media']), f"média de {ind['jejum_n']} medida(s)"),
        (f"Pressão arterial ({dias} dias)",
         f"{ind['pa_media'][0]:.0f}/{ind['pa_media'][1]:.0f}" if ind['pa_media'] else "—",
         status_pa(ind['pa_media']), f"média de {ind['pa_n']} medida(s), mmHg"),
        ("Atividade física (7 dias)", f"{ind['atividade_semana']} min",
         _lozenge("Meta atingida", "success") if ind['atividade_semana'] >= 150 else _lozenge("Abaixo de 150", "warning"),
         "recomendação da OMS: 150 min/semana"),
        (f"Peso ({dias} dias)",
         f"{_num(ind['peso_atual'])} kg" if ind['peso_atual'] else "—",
         _lozenge(f"{ind['peso_variacao']:+.1f} kg".replace(".", ","), "info") if ind['peso_variacao'] is not None
         else _lozenge("Sem variação", "neutral"),
         "variação no período" if ind['peso_variacao'] is not None else "menos de 2 registros"),
        ("Adesão ao diário", f"{ind['dias_com_registro']} de {dias}",
         _lozenge("dias com registro", "neutral"),
         f"último em {_data_iso_br(ind['ultimo_registro'])}" if ind['ultimo_registro'] else "nenhum registro"),
    ]
    colunas = st.columns(3)
    for i, (titulo, valor, etiqueta, nota) in enumerate(cartoes):
        with colunas[i % 3]:
            st.markdown(f'<div class="card" style="margin-bottom: 12px">'
                        f'<div class="card-row"><span class="card-note">{titulo}</span>{etiqueta}</div>'
                        f'<div class="card-big" style="font-size: 26px; line-height: 32px">{valor}</div>'
                        f'<div class="card-note">{nota}</div></div>', unsafe_allow_html=True)


def _lista_achados(achados: List[Dict], dias: int):
    if not achados:
        st.success(f"Nenhuma medida fora da faixa nos últimos {dias} dias.")
        return
    linhas = "".join(
        f'<div class="fator"><span class="estado {"danger" if a["grave"] else "warning"}">!</span>'
        f'<div style="flex: 1"><div class="nome">{a["texto"]}</div></div>'
        f'<span class="card-note">{_data_iso_br(a["data"])}</span></div>'
        for a in achados[:12])
    resto = f'<div class="card-note">e mais {len(achados) - 12}</div>' if len(achados) > 12 else ""
    st.markdown(f'<div class="card"><div class="card-row"><span class="card-title">Achados que pedem atenção</span>'
                f'<span class="card-note">{len(achados)} no período</span></div>'
                f'<div class="fatores">{linhas}</div>{resto}</div>', unsafe_allow_html=True)


def _eixo_x():
    return alt.X('data:T', title=None, axis=alt.Axis(format='%d/%m', labelAngle=0, tickCount=7,
                                                      labelColor=TEXTO_EIXO, grid=False))


def _regra(valor: float, texto: str):
    base = pd.DataFrame({'ref': [valor], 'texto': [texto]})
    return [alt.Chart(base).mark_rule(color=COR_REFERENCIA, strokeDash=[4, 4], strokeWidth=1).encode(y='ref:Q'),
            alt.Chart(base).mark_text(align='left', dx=4, dy=-6, color=COR_REFERENCIA, fontSize=11)
            .encode(y='ref:Q', x=alt.value(0), text='texto:N')]


def _graficos(registros: List[Dict]):
    def tabela(tipo):
        return pd.DataFrame([{'data': pd.to_datetime(r['data']), **r['dados']} for r in registros if r['tipo'] == tipo])

    glicemia, pressao, peso = tabela('glicemia'), tabela('pressao'), tabela('peso')
    if glicemia.empty and pressao.empty and peso.empty:
        st.info("O paciente não registrou glicemia, pressão ou peso no período.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Glicemia")
        if glicemia.empty:
            st.caption("Sem medidas no período.")
        else:
            pontos = alt.Chart(glicemia).mark_point(size=70, filled=True, color=COR_SERIE_1, opacity=1).encode(
                x=_eixo_x(),
                y=alt.Y('valor:Q', title='mg/dL', axis=alt.Axis(labelColor=TEXTO_EIXO, titleColor=TEXTO_EIXO,
                                                                 gridColor='#2C333A')),
                shape=alt.Shape('momento:N', title='Momento', legend=alt.Legend(orient='bottom', labelColor=TEXTO_EIXO,
                                                                                titleColor=TEXTO_EIXO)),
                tooltip=[alt.Tooltip('data:T', title='Data', format='%d/%m/%Y'),
                         alt.Tooltip('valor:Q', title='mg/dL'), alt.Tooltip('momento:N', title='Momento')])
            camadas = [pontos] + _regra(126, "126 · diabetes (jejum)") + _regra(100, "100 · alterada (jejum)")
            st.altair_chart(alt.layer(*camadas).properties(height=260), width="stretch")

    with col2:
        st.markdown("##### Pressão arterial")
        if pressao.empty:
            st.caption("Sem medidas no período.")
        else:
            longa = pressao.melt(id_vars='data', value_vars=['sistolica', 'diastolica'], var_name='medida', value_name='mmHg')
            longa['medida'] = longa['medida'].map({'sistolica': 'Sistólica', 'diastolica': 'Diastólica'})
            linhas = alt.Chart(longa).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
                x=_eixo_x(),
                y=alt.Y('mmHg:Q', title='mmHg', scale=alt.Scale(zero=False),
                        axis=alt.Axis(labelColor=TEXTO_EIXO, titleColor=TEXTO_EIXO, gridColor='#2C333A')),
                color=alt.Color('medida:N', title=None, sort=['Sistólica', 'Diastólica'],
                                scale=alt.Scale(domain=['Sistólica', 'Diastólica'], range=[COR_SERIE_1, COR_SERIE_2]),
                                legend=alt.Legend(orient='bottom', labelColor=TEXTO_EIXO)),
                tooltip=[alt.Tooltip('data:T', title='Data', format='%d/%m/%Y'),
                         alt.Tooltip('medida:N', title='Medida'), alt.Tooltip('mmHg:Q')])
            camadas = [linhas] + _regra(140, "140 · sistólica elevada") + _regra(90, "90 · diastólica elevada")
            st.altair_chart(alt.layer(*camadas).properties(height=260), width="stretch")

    if not peso.empty:
        st.markdown("##### Peso")
        por_dia = peso.groupby('data', as_index=False)['peso_kg'].last()
        st.altair_chart(alt.Chart(por_dia).mark_line(color=COR_SERIE_1, strokeWidth=2,
                                                     point=alt.OverlayMarkDef(size=60, filled=True)).encode(
            x=_eixo_x(),
            y=alt.Y('peso_kg:Q', title='kg', scale=alt.Scale(zero=False),
                    axis=alt.Axis(labelColor=TEXTO_EIXO, titleColor=TEXTO_EIXO, gridColor='#2C333A')),
            tooltip=[alt.Tooltip('data:T', title='Data', format='%d/%m/%Y'), alt.Tooltip('peso_kg:Q', title='kg')],
        ).properties(height=200), width="stretch")


def _registros_completos(registros: List[Dict]):
    if not registros:
        st.caption("Nenhum registro no período.")
        return
    nomes = diario.CATEGORIAS
    tabela = pd.DataFrame([{
        'Data': _data_iso_br(r['data']),
        'Tipo': nomes.get(r['tipo'], r['tipo']),
        'Registro': diario._descrever(r).replace(r" \*", " *"),
    } for r in sorted(registros, key=lambda r: (r['data'], r.get('created_at', '')), reverse=True)])
    st.dataframe(tabela, hide_index=True, width="stretch")


def _ficha(paciente: Dict, medico_id: str, inicio, dias: int):
    pid = paciente['paciente_id']
    perfil = _carregar_perfil(pid)
    predicoes = supabase_db.obter_ultimas_predicoes(limite=50, user_id=pid)
    registros = diario.carregar_registros(pid, inicio, diario.hoje())

    _cabecalho_ficha(paciente, perfil)
    st.markdown("### Indicadores")
    _cartoes_indicadores(_indicadores(registros), predicoes, dias)
    _lista_achados(_achados(registros), dias)

    st.markdown("### Evolução")
    _graficos(registros)

    st.markdown("### Avaliações de risco")
    _avaliacoes_paciente(pid)

    with st.expander(f"Todos os registros do diário ({len(registros)} nos últimos {dias} dias)"):
        _registros_completos(registros)

    with st.expander("Encerrar acompanhamento deste paciente"):
        st.caption("Você deixa de ver os dados deste paciente. Para voltar, ele precisa compartilhar de novo.")
        if st.button("Encerrar acompanhamento", key=f"encerrar_{pid}"):
            if _remover_compartilhamento(pid, medico_id):
                st.session_state.pop('paciente_aberto', None)
                st.rerun()
            else:
                st.error("Não foi possível encerrar. Tente novamente.")


def _abrir_ficha(paciente_id: str):
    st.session_state.paciente_aberto = paciente_id
    st.session_state.pagina = "Meus Pacientes"


def _ir_para(pagina: str):
    st.session_state.pagina = pagina


def painel_medico(medico_id: str, primeiro_nome: str):
    """Início da conta de médico: resumo dos pacientes e quem precisa de atenção primeiro"""
    st.markdown(f'<div class="page-header"><span class="eyebrow">Painel</span>'
                f'<span class="title">Olá, {primeiro_nome}</span>'
                f'<span class="subtitle">Resumo dos pacientes que compartilham os dados com você '
                f'(últimos 30 dias).</span></div>', unsafe_allow_html=True)

    if not supabase_db.conectado or not supabase_db.client:
        st.warning("Banco de dados indisponível.")
        return
    try:
        pacientes = _rpc('meus_pacientes')
    except Exception as e:
        print(f"Erro ao listar pacientes: {e}")
        st.warning("Não foi possível carregar seus pacientes agora.")
        return

    if not pacientes:
        st.info("Nenhum paciente compartilhou dados com você ainda. O paciente vai em "
                "*Meu perfil → Médicos com acesso* e informa o seu email de cadastro.")
    else:
        with st.spinner("Carregando pacientes..."):
            linhas = sorted((_linha_triagem(p, diario.hoje() - timedelta(days=29)) for p in pacientes),
                            key=lambda l: -l['_prioridade'])
        com_achado = sum(1 for l in linhas if l['Achados'])
        elevado = sum(1 for l in linhas if l['Resultado'] == 'Elevado')
        sem_registro = sum(1 for l in linhas if l['Último registro'] == '—')
        st.markdown(
            f'<div class="tecnico">'
            f'<div><span>Pacientes</span><b>{len(linhas)}</b></div>'
            f'<div><span>Com achados fora da faixa</span><b>{com_achado}</b></div>'
            f'<div><span>Última avaliação com risco elevado</span><b>{elevado}</b></div>'
            f'<div><span>Sem registros no diário</span><b>{sem_registro}</b></div>'
            f'</div>', unsafe_allow_html=True)

        atencao = [l for l in linhas if l['_prioridade'] > 0][:5]
        st.markdown("### Precisam de atenção")
        if not atencao:
            st.success("Nenhum paciente com achados ou risco elevado no período.")
        for l in atencao:
            with st.container(border=True, key=f"card_atencao_{l['id']}"):
                col_info, col_botao = st.columns([4, 1], vertical_alignment="center")
                etiquetas = []
                if l['Resultado'] == 'Elevado':
                    etiquetas.append(_lozenge(f"Risco {_num(l['Último risco'] * 100)}%", "danger"))
                if l['Achados']:
                    etiquetas.append(_lozenge(f"{l['Achados']} achado{'s' if l['Achados'] > 1 else ''}", "warning"))
                detalhes = " · ".join(x for x in [
                    f"{l['Idade']} anos" if l['Idade'] else "",
                    f"IMC {_num(l['IMC'])}" if l['IMC'] else "",
                    f"jejum médio {l['Glicemia jejum (média)']} mg/dL" if l['Glicemia jejum (média)'] else "",
                    f"PA {l['PA média']}" if l['PA média'] != '—' else "",
                ] if x)
                col_info.markdown(f'<div class="card-row" style="justify-content: flex-start">'
                                  f'<span class="card-title">{l["Paciente"]}</span>{" ".join(etiquetas)}</div>'
                                  f'<div class="card-note">{detalhes or "sem dados no período"}</div>',
                                  unsafe_allow_html=True)
                col_botao.button("Abrir ficha", key=f"abrir_{l['id']}", width="stretch",
                                 on_click=_abrir_ficha, args=(l['id'],))
        st.button("Ver todos os pacientes", icon=":material/groups:", on_click=_ir_para, args=("Meus Pacientes",))

    st.markdown("### Ferramentas de consulta")
    col1, col2 = st.columns(2)
    with col1, st.container(border=True, key="card_ferramenta_exames", height="stretch"):
        st.markdown('<div class="card-title">Calculadora com exames</div>'
                    '<div class="card-text">Risco a partir de TOTG 2h, insulina, PA diastólica, IMC e idade.</div>'
                    '<div class="card-note">Random Forest · Pima Indians · AUC 0,839 · não salva o resultado</div>',
                    unsafe_allow_html=True)
        st.button("Abrir calculadora", key="ir_calc_exames", width="stretch", on_click=_ir_para, args=("Modelo Clínico",))
    with col2, st.container(border=True, key="card_ferramenta_questionario", height="stretch"):
        st.markdown('<div class="card-title">Questionário de triagem</div>'
                    '<div class="card-text">Risco sem exames, a partir de 8 perguntas sobre condições e hábitos.</div>'
                    '<div class="card-note">XGBoost · BRFSS 2015 · AUC 0,821 · não salva o resultado</div>',
                    unsafe_allow_html=True)
        st.button("Abrir questionário", key="ir_calc_questionario", width="stretch",
                  on_click=_ir_para, args=("Modelo Comportamental",))


def pagina_meus_pacientes(medico_id: str):
    st.markdown('<div class="page-header"><span class="eyebrow">Principal</span>'
                '<span class="title">Meus pacientes</span>'
                '<span class="subtitle">Dados compartilhados pelos pacientes, só para leitura. '
                'O paciente pode encerrar o acesso a qualquer momento.</span></div>', unsafe_allow_html=True)

    if not supabase_db.conectado or not supabase_db.client:
        st.warning("Banco de dados indisponível.")
        return
    try:
        pacientes = _rpc('meus_pacientes')
    except Exception as e:
        print(f"Erro ao listar pacientes: {e}")
        st.warning("Não foi possível carregar seus pacientes agora.")
        return

    if not pacientes:
        st.info("Nenhum paciente compartilhou dados com você ainda. Peça para o paciente ir em "
                "*Meu perfil → Médicos com acesso* e informar o seu email de cadastro.")
        return

    dias = st.segmented_control("Período", [30, 90], default=30, format_func=lambda d: f"Últimos {d} dias",
                                key="medico_periodo") or 30
    inicio = diario.hoje() - timedelta(days=dias - 1)
    por_id = {p['paciente_id']: p for p in pacientes}

    aberto = st.session_state.get('paciente_aberto')
    if aberto in por_id:
        if st.button("Voltar para a lista", icon=":material/arrow_back:", type="tertiary"):
            st.session_state.pop('paciente_aberto')
            st.rerun()
        _ficha(por_id[aberto], medico_id, inicio, dias)
        return

    clicado = _triagem(pacientes, inicio, dias)
    if clicado:
        st.session_state.paciente_aberto = clicado
        # A seleção da tabela não deve reabrir a ficha quando o médico voltar para a lista
        st.session_state.pop('triagem_pacientes', None)
        st.rerun()
