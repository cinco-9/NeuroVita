# -*- coding: utf-8 -*-
"""
MAPEAMENTO VIGITEL 2023 -> FORMATO DO MODELO
Variáveis conferidas no dicionário oficial (dados_vigitel/dicionario-vigitel-2006-2024.xlsx).
Autor: David Reis | 2026
"""

import json

import numpy as np
import pandas as pd

ORIGEM = 'dados_vigitel/Vigitel-2023-peso-rake.xlsx'
DESTINO = 'dados_vigitel/vigitel_2023_processado.csv'

# Códigos de "não sabe" / "não quis informar"
NAO_INFORMADO = [777, 888, 999]

df = pd.read_excel(ORIGEM)
print(f"Registros brutos: {len(df):,}")


def sim_nao(coluna):
    """1 = sim, 2 = não; demais códigos viram ausente."""
    s = df[coluna].replace(NAO_INFORMADO, np.nan)
    return s.map({1: 1, 2: 0})


out = pd.DataFrame({
    'Diabetes': sim_nao('q76'),                                   # q76: diabetes (diagnóstico médico)
    'Age': df['q6'],                                              # q6: idade (anos)
    'Sex': (df['q7'] == 1).astype(int),                           # q7: 1 = masculino
    'BMI': df['imc'],                                             # imc: calculado pelo VIGITEL
    'HighBP': sim_nao('q75'),                                     # q75: pressão alta
    'Smoker': df['fumante'],                                      # indicador VIGITEL: fumante atual
    'PhysActivity': df['ativo_livre'],                            # indicador VIGITEL: ativo no tempo livre
    'Fruits': df['frutareg'],                                     # indicador VIGITEL: fruta >= 5 dias/semana
    'GenHlth': df['q74'].replace(NAO_INFORMADO, np.nan),          # q74: 1 muito bom ... 5 muito ruim
    'peso': df['pesorake'],                                       # peso amostral (pós-estratificação)
})
# O VIGITEL não pergunta sobre diagnóstico de colesterol alto (só se fez o exame), por isso não há HighChol.

antes = len(out)
out = out.dropna()
out = out[(out['BMI'].between(10, 60)) & (out['Age'].between(18, 100))]
print(f"Removidos por dado ausente/inválido: {antes - len(out):,}")

for c in ['Diabetes', 'Age', 'Sex', 'HighBP', 'Smoker', 'PhysActivity', 'Fruits', 'GenHlth']:
    out[c] = out[c].astype(int)

prev_bruta = out['Diabetes'].mean()
prev_pond = np.average(out['Diabetes'], weights=out['peso'])
print(f"Registros finais: {len(out):,}")
print(f"Prevalência de diabetes: {prev_bruta*100:.1f}% (amostra) | {prev_pond*100:.1f}% (ponderada)")
print(f"Pressão alta: {np.average(out['HighBP'], weights=out['peso'])*100:.1f}% (ponderada)")

out.to_csv(DESTINO, index=False)
with open('dados_vigitel/metadata_processado.json', 'w', encoding='utf-8') as f:
    json.dump({
        'registros': int(len(out)),
        'prevalencia_diabetes_amostra': round(float(prev_bruta), 4),
        'prevalencia_diabetes_ponderada': round(float(prev_pond), 4),
        'colunas': out.columns.tolist(),
    }, f, indent=2, ensure_ascii=False)
print(f"Salvo: {DESTINO}")
