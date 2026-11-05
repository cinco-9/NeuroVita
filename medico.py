# -*- coding: utf-8 -*-
"""
ACESSO DO MÉDICO
- Paciente: compartilha seus dados com um médico (pelo email da conta de médico) e revoga quando quiser
- Médico: vê, só para leitura, perfil, avaliações de risco e diário dos pacientes que compartilharam
As regras de acesso ficam no banco (diario_medico.sql); o app só chama as funções.
"""

from datetime import datetime
from typing import Dict, List

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
# ============================================================================

def _perfil_paciente(paciente_id: str):
    try:
        resposta = supabase_db.client.table('user_profiles').select('*').eq('id', paciente_id).execute()
    except Exception as e:
        print(f"Erro ao carregar perfil do paciente: {e}")
        resposta = None
    perfil = resposta.data[0] if resposta and resposta.data else None
    if not perfil:
        st.caption("O paciente ainda não preencheu o perfil.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Idade", f"{perfil['idade']} anos" if perfil.get('idade') else "—")
    col2.metric("Sexo", perfil.get('sexo') or "—")
    col3.metric("Peso", f"{perfil['peso']:.1f} kg" if perfil.get('peso') else "—")
    col4.metric("IMC", f"{perfil['imc']:.1f}" if perfil.get('imc') else "—")
    st.markdown(f"**Histórico familiar de diabetes:** {'Sim' if perfil.get('hist_familiar_diabetes') else 'Não'}  \n"
                f"**Alergias:** {perfil.get('alergias') or '—'}  \n"
                f"**Medicações:** {perfil.get('medicacoes') or '—'}")


def _avaliacoes_paciente(paciente_id: str):
    predicoes = supabase_db.obter_ultimas_predicoes(limite=50, user_id=paciente_id)
    if not predicoes:
        st.info("O paciente ainda não fez avaliações de risco.")
        return
    tabela = pd.DataFrame([{
        'Data': _data_br(p['created_at']),
        'Modelo': 'Clínico' if p['tipo_modelo'] == 'clinico' else 'Comportamental',
        'Resultado': 'Alto risco' if p['resultado'] == 1 else 'Baixo risco',
        'Probabilidade': f"{p['probabilidade'] * 100:.1f}%",
    } for p in predicoes])
    st.dataframe(tabela, hide_index=True, width="stretch")


def pagina_meus_pacientes(medico_id: str):
    st.markdown('<div class="main-header">Meus Pacientes</div>', unsafe_allow_html=True)
    st.caption("Pacientes que compartilharam os dados com você. O acesso é só para leitura "
               "e o paciente pode encerrá-lo a qualquer momento.")

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
                "*Meu Perfil → Compartilhar com meu médico* e informar o seu email de cadastro.")
        return

    opcoes = {p['paciente_id']: f"{p.get('nome') or 'Sem nome'} ({p['email']})" for p in pacientes}
    paciente_id = st.selectbox("Paciente", list(opcoes), format_func=opcoes.get)
    desde = next(p['desde'] for p in pacientes if p['paciente_id'] == paciente_id)
    st.caption(f"Compartilhando com você desde {_data_br(desde)}")

    aba_perfil, aba_risco, aba_diario = st.tabs(["Perfil", "Avaliações de risco", "Diário"])
    with aba_perfil:
        _perfil_paciente(paciente_id)
    with aba_risco:
        _avaliacoes_paciente(paciente_id)
    with aba_diario:
        diario.mostrar_diario(paciente_id, somente_leitura=True)

    st.markdown("---")
    with st.expander("Encerrar acompanhamento deste paciente"):
        st.caption("Você deixa de ver os dados deste paciente. Para voltar, ele precisa compartilhar de novo.")
        if st.button("Encerrar acompanhamento", key=f"encerrar_{paciente_id}"):
            if _remover_compartilhamento(paciente_id, medico_id):
                st.rerun()
            else:
                st.error("Não foi possível encerrar. Tente novamente.")
