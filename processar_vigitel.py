# -*- coding: utf-8 -*-
"""
PROCESSAR DADOS VIGITEL 2023
Análise exploratória e preparação para o modelo
Autor: David Reis | 2026
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("PROCESSAMENTO VIGITEL 2023 - DADOS BRASILEIROS")
print("=" * 100)

# ============================================================================
# 1. CARREGAR DICIONÁRIO
# ============================================================================

print("\n[1/5] Lendo dicionário de variáveis...")

try:
    dicionario = pd.read_excel('dados_vigitel/dicionario-vigitel-2006-2024.xlsx')
    print(f"✓ Dicionário carregado: {len(dicionario)} variáveis documentadas")

    # Mostrar estrutura do dicionário
    print(f"\nColunas do dicionário: {dicionario.columns.tolist()}")

except Exception as e:
    print(f"❌ Erro ao ler dicionário: {e}")
    dicionario = None

# ============================================================================
# 2. CARREGAR DADOS VIGITEL
# ============================================================================

print("\n" + "=" * 100)
print("[2/5] Carregando dados VIGITEL 2023...")
print("=" * 100)

try:
    # Ler apenas primeiras linhas para explorar
    print("\nLendo amostra inicial...")
    df_sample = pd.read_excel('dados_vigitel/Vigitel-2023-peso-rake.xlsx', nrows=100)

    print(f"✓ Amostra carregada: {len(df_sample)} registros, {len(df_sample.columns)} colunas")

    # Agora carregar completo
    print("\nCarregando dataset completo...")
    df = pd.read_excel('dados_vigitel/Vigitel-2023-peso-rake.xlsx')

    print(f"✓ Dataset completo: {len(df):,} registros, {len(df.columns)} colunas")

except Exception as e:
    print(f"❌ Erro ao carregar dados: {e}")
    df = None
    exit(1)

# ============================================================================
# 3. PROCURAR VARIÁVEL DIABETES
# ============================================================================

print("\n" + "=" * 100)
print("[3/5] Identificando variável de DIABETES...")
print("=" * 100)

# Tentar várias abordagens
candidatas_diabetes = []

# Abordagem 1: Nome da coluna contém "diabet"
for col in df.columns:
    if 'diabet' in str(col).lower():
        candidatas_diabetes.append(col)
        print(f"\n✓ Encontrado por nome: {col}")
        print(f"  Valores únicos: {df[col].unique()[:10]}")
        print(f"  Total valores: {df[col].value_counts()}")

# Abordagem 2: Procurar Q60 (pergunta comum sobre diabetes em VIGITEL)
q60_cols = [col for col in df.columns if 'q60' in str(col).lower() or 'q_60' in str(col).lower()]
if q60_cols:
    print(f"\n✓ Encontrado Q60 (pergunta diabetes): {q60_cols}")
    for col in q60_cols:
        if col not in candidatas_diabetes:
            candidatas_diabetes.append(col)
        print(f"  {col}: {df[col].value_counts()}")

# Abordagem 3: Buscar no dicionário se disponível
if dicionario is not None:
    print("\n📖 Consultando dicionário...")

    # Tentar encontrar coluna com descrição
    col_descricao = None
    for col in dicionario.columns:
        if 'descri' in col.lower() or 'label' in col.lower():
            col_descricao = col
            break

    if col_descricao:
        diabetes_dict = dicionario[dicionario[col_descricao].str.contains('diabetes|diabete',
                                                                          case=False, na=False)]
        if len(diabetes_dict) > 0:
            print(f"\n✓ Variáveis diabetes no dicionário:")
            for idx, row in diabetes_dict.iterrows():
                var_name = row.iloc[0]  # Primeira coluna geralmente é o nome
                desc = row[col_descricao] if col_descricao else "Sem descrição"
                print(f"  - {var_name}: {desc[:80]}...")
                if var_name in df.columns and var_name not in candidatas_diabetes:
                    candidatas_diabetes.append(var_name)

# ============================================================================
# 4. ANÁLISE DAS VARIÁVEIS CANDIDATAS
# ============================================================================

print("\n" + "=" * 100)
print("[4/5] Analisando variáveis candidatas...")
print("=" * 100)

if candidatas_diabetes:
    print(f"\n✓ Total de candidatas: {len(candidatas_diabetes)}")

    for var in candidatas_diabetes:
        if var in df.columns:
            print(f"\n{'─' * 100}")
            print(f"Variável: {var}")
            print(f"{'─' * 100}")

            # Estatísticas
            print(f"\nEstatísticas:")
            print(f"  Tipo: {df[var].dtype}")
            print(f"  Valores únicos: {df[var].nunique()}")
            print(f"  Valores faltantes: {df[var].isna().sum()} ({df[var].isna().sum()/len(df)*100:.1f}%)")

            # Distribuição
            print(f"\nDistribuição de valores:")
            vc = df[var].value_counts().sort_index()
            for val, count in vc.items():
                pct = count/len(df)*100
                print(f"  {val}: {count:,} ({pct:.1f}%)")

            # Se parece binário (0/1 ou 1/2)
            if df[var].nunique() <= 3:
                # Calcular prevalência
                if 1 in df[var].values:
                    prev_1 = (df[var] == 1).sum() / len(df) * 100
                    print(f"\n  → Prevalência de '1': {prev_1:.1f}%")
                if 2 in df[var].values:
                    prev_2 = (df[var] == 2).sum() / len(df) * 100
                    print(f"  → Prevalência de '2': {prev_2:.1f}%")
else:
    print("\n⚠️ Nenhuma variável candidata encontrada diretamente")
    print("\nMostrando primeiras 30 colunas para inspeção manual:")
    for i, col in enumerate(df.columns[:30], 1):
        sample_vals = df[col].value_counts().head(3).to_dict()
        print(f"  {i:2d}. {col:20s} - Valores: {sample_vals}")

# ============================================================================
# 5. ESTATÍSTICAS GERAIS
# ============================================================================

print("\n" + "=" * 100)
print("[5/5] Estatísticas gerais do dataset")
print("=" * 100)

print(f"\n📊 Dimensões:")
print(f"  Registros: {len(df):,}")
print(f"  Variáveis: {len(df.columns)}")

print(f"\n📋 Tipos de dados:")
print(df.dtypes.value_counts())

print(f"\n🔍 Primeiras 5 linhas, primeiras 10 colunas:")
print(df.iloc[:5, :10])

# Salvar lista de todas as colunas
with open('dados_vigitel/lista_colunas.txt', 'w', encoding='utf-8') as f:
    f.write("LISTA COMPLETA DE COLUNAS - VIGITEL 2023\n")
    f.write("=" * 80 + "\n\n")
    for i, col in enumerate(df.columns, 1):
        f.write(f"{i:3d}. {col}\n")

print(f"\n✓ Lista de colunas salva em: dados_vigitel/lista_colunas.txt")

# ============================================================================
# RESUMO
# ============================================================================

print("\n" + "=" * 100)
print("✅ PROCESSAMENTO INICIAL CONCLUÍDO")
print("=" * 100)

print(f"""
📊 DADOS VIGITEL 2023:
  - Registros: {len(df):,}
  - Variáveis: {len(df.columns)}
  - Variáveis candidatas diabetes: {len(candidatas_diabetes) if candidatas_diabetes else 'A identificar'}

📖 PRÓXIMOS PASSOS:
  1. Confirmar qual variável é o diagnóstico de diabetes
  2. Identificar variáveis de features (IMC, idade, sexo, etc.)
  3. Mapear para formato do modelo
  4. Limpeza e pré-processamento
  5. Treinar modelo brasileiro!

💡 DICA:
  Consulte a nota metodológica para entender os códigos das variáveis
  Arquivo: dados_vigitel/nota-metodologica-vigitel-2006-2024.pdf
""")

print("=" * 100)
