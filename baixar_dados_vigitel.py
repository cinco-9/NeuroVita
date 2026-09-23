# -*- coding: utf-8 -*-
"""
BAIXAR E PROCESSAR DADOS VIGITEL (BRASIL)
Sistema de Vigilância de Fatores de Risco - Ministério da Saúde
Autor: David Reis | 2026
"""

import pandas as pd
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DOWNLOAD DE DADOS VIGITEL - BRASIL")
print("Sistema de Vigilância de Fatores de Risco (Ministério da Saúde)")
print("=" * 80)

# ============================================================================
# OPÇÃO 1: Download Manual (Mais Confiável)
# ============================================================================

print("\n📋 INSTRUÇÕES PARA DOWNLOAD MANUAL:")
print("-" * 80)
print("""
1. Acesse: https://svs.aids.gov.br/download/Vigitel/

2. Escolha um ano recente (ex: 2023, 2024)

3. Baixe o arquivo CSV ou DBF

4. Coloque na pasta do projeto: F:\\dev\\tcc-diabetes\\

5. Rode este script novamente para processar os dados
""")

# ============================================================================
# OPÇÃO 2: Verificar se já baixou
# ============================================================================

print("\n🔍 Procurando arquivos VIGITEL locais...")

import glob
import os

arquivos_vigitel = glob.glob("*vigitel*.csv") + glob.glob("*VIGITEL*.csv")

if arquivos_vigitel:
    print(f"\n✓ Encontrado: {len(arquivos_vigitel)} arquivo(s)")
    for arq in arquivos_vigitel:
        tamanho = os.path.getsize(arq) / (1024*1024)
        print(f"  - {arq} ({tamanho:.1f} MB)")

    # Processar o primeiro arquivo encontrado
    print(f"\n📊 Processando: {arquivos_vigitel[0]}...")

    try:
        df = pd.read_csv(arquivos_vigitel[0], encoding='latin-1', low_memory=False)
        print(f"✓ Carregado: {df.shape[0]:,} registros, {df.shape[1]} colunas")

        print("\n📋 Primeiras colunas:")
        print(df.columns[:20].tolist())

        print("\n📊 Informações do dataset:")
        print(df.info())

        # Procurar colunas relacionadas a diabetes
        colunas_diabetes = [col for col in df.columns if 'diabet' in col.lower()]
        if colunas_diabetes:
            print(f"\n🔍 Colunas relacionadas a diabetes encontradas:")
            for col in colunas_diabetes:
                print(f"  - {col}")
                if df[col].dtype in ['int64', 'float64']:
                    print(f"    Valores únicos: {df[col].unique()[:10]}")

        print("\n✅ PRÓXIMO PASSO:")
        print("Analise as colunas e crie um script de mapeamento")
        print("similar ao modelo comportamental (BRFSS)")

    except Exception as e:
        print(f"❌ Erro ao processar: {e}")
        print("Tente ajustar encoding ou formato do arquivo")

else:
    print("\n⚠️ Nenhum arquivo VIGITEL encontrado")
    print("\n📥 BAIXE MANUALMENTE:")
    print("1. Acesse: https://svs.aids.gov.br/download/Vigitel/")
    print("2. Baixe o CSV mais recente")
    print("3. Coloque nesta pasta")
    print("4. Rode este script novamente")

# ============================================================================
# OPÇÃO 3: Dados de Exemplo/Sintéticos
# ============================================================================

print("\n" + "=" * 80)
print("ALTERNATIVA: CRIAR DADOS SINTÉTICOS BRASILEIROS")
print("=" * 80)

print("""
Se não conseguir acessar VIGITEL, você pode:

1. Criar dataset sintético baseado em estatísticas brasileiras
2. Usar o mesmo modelo BRFSS ajustando prevalências
3. Documentar no TCC que dados reais não estavam disponíveis

Prevalências no Brasil (VIGITEL 2024):
- Diabetes: 12.9% (cresceu de 5.5% em 2006)
- Obesidade: ~22%
- Hipertensão: ~27%
- Atividade física: ~40% praticam

Isso é ACEITÁVEL para um TCC!
""")

print("\n✅ Script concluído!")
print("=" * 80)
