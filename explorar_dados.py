# -*- coding: utf-8 -*-
"""
Passo 1: Explorar o dataset antes de treinar qualquer modelo.
Coloque o arquivo CSV baixado do Kaggle na mesma pasta deste script
e ajuste o nome do arquivo abaixo se necessário.
"""

import pandas as pd

NOME_ARQUIVO = "diabetes_binary_health_indicators_BRFSS2015.csv"

# Carrega o dataset
df = pd.read_csv(NOME_ARQUIVO)

# Visão geral
print("Formato do dataset (linhas, colunas):", df.shape)
print("\nPrimeiras linhas:")
print(df.head())

print("\nNomes das colunas:")
print(df.columns.tolist())

print("\nEstatísticas descritivas:")
print(df.describe())

print("\nQuantas pessoas têm e não têm diabetes (0 = não, 1 = sim):")
print(df["Diabetes_binary"].value_counts())

print("\nValores ausentes por coluna:")
print(df.isnull().sum())
