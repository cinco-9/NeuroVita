# -*- coding: utf-8 -*-
"""
Baixa os dados originais do BRFSS 2015 (CDC) usados na validação Brasil x EUA.
Diferente da versão do Kaggle usada no modelo comportamental, os dados originais separam
diabetes de pré-diabetes (variável DIABETE3). Salva em dados_brfss/LLCP2015.XPT e as colunas usadas em dados_brfss/brfss2015_colunas.csv
"""

import os
import urllib.request
import zipfile

import pandas as pd

PASTA = 'dados_brfss'
URL = 'https://www.cdc.gov/brfss/annual_data/2015/files/LLCP2015XPT.zip'

os.makedirs(PASTA, exist_ok=True)
zip_path = os.path.join(PASTA, 'LLCP2015XPT.zip')
if not os.path.exists(zip_path):
    print('Baixando BRFSS 2015 (~99 MB)...')
    urllib.request.urlretrieve(URL, zip_path)
xpt = os.path.join(PASTA, 'LLCP2015.XPT')
with zipfile.ZipFile(zip_path) as z:
    if not os.path.exists(xpt):
        with z.open(z.namelist()[0]) as origem, open(xpt, 'wb') as saida:
            saida.write(origem.read())
print(f'{xpt}: {os.path.getsize(xpt) / 1e6:.0f} MB')

# Só as colunas usadas, para não reler o arquivo de 1,1 GB a cada análise
COLUNAS = ['DIABETE3', '_AGEG5YR', 'SEX', '_BMI5', '_RFHYPE5', 'GENHLTH', 'SMOKE100', '_TOTINDA', '_FRTLT1',
           '_LLCPWT']
partes = [bloco[COLUNAS] for bloco in pd.read_sas(xpt, format='xport', chunksize=100000)]
csv = os.path.join(PASTA, 'brfss2015_colunas.csv')
pd.concat(partes, ignore_index=True).to_csv(csv, index=False)
print(f'{csv}: {sum(len(p) for p in partes):,} linhas')
