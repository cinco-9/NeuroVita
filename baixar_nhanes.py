# -*- coding: utf-8 -*-
"""
Baixa os arquivos do NHANES (CDC) usados no estudo de rastreamento de diabetes não diagnosticado.
Ciclos 2011-2012 (G), 2013-2014 (H), 2015-2016 (I) e 2017-2018 (J). Salva em dados_nhanes/*.xpt
"""

import os
import urllib.request

PASTA = 'dados_nhanes'
CICLOS = {'2011': 'G', '2013': 'H', '2015': 'I', '2017': 'J'}
ARQUIVOS = [
    'DEMO',    # idade, sexo, raça/etnia, escolaridade, renda, pesos amostrais
    'BMX',     # IMC, cintura, altura
    'BPX',     # pressão arterial medida
    'GHB',     # HbA1c (só para definir o desfecho)
    'GLU',     # glicemia de jejum (só para definir o desfecho)
    'INS',     # insulina de jejum
    'TCHOL',   # colesterol total
    'HDL',     # HDL
    'TRIGLY',  # triglicérides (subamostra em jejum)
    'DIQ',     # diabetes diagnosticado
    'BPQ',     # hipertensão diagnosticada
    'MCQ',     # histórico familiar de diabetes
    'PAQ',     # atividade física
    'SMQ',     # tabagismo
    'RHQ',     # diabetes gestacional
]

os.makedirs(PASTA, exist_ok=True)
for ano, sufixo in CICLOS.items():
    for nome in ARQUIVOS:
        arquivo = f'{nome}_{sufixo}.xpt'
        destino = os.path.join(PASTA, arquivo)
        if os.path.exists(destino):
            continue
        url = f'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{ano}/DataFiles/{arquivo}'
        try:
            urllib.request.urlretrieve(url, destino)
            with open(destino, 'rb') as f:
                if b'<html' in f.read(500).lower():
                    raise ValueError('página HTML em vez do arquivo')
            print(f'{arquivo}: {os.path.getsize(destino) / 1e6:.1f} MB')
        except Exception as e:
            if os.path.exists(destino):
                os.remove(destino)
            print(f'{arquivo}: NÃO DISPONÍVEL ({e})')
print('Download concluído.')
