# -*- coding: utf-8 -*-
"""
MAPEAMENTO COMPLETO VIGITEL → MODELO
Identifica e mapeia todas as features necessárias
Autor: David Reis | 2026
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("MAPEAMENTO VIGITEL 2023 → MODELO DE DIABETES")
print("=" * 100)

# ============================================================================
# 1. CARREGAR DADOS
# ============================================================================

print("\n[1/6] Carregando dados VIGITEL 2023...")

df = pd.read_excel('dados_vigitel/Vigitel-2023-peso-rake.xlsx')
print(f"✓ Dataset carregado: {len(df):,} registros, {len(df.columns)} colunas")

# ============================================================================
# 2. IDENTIFICAR VARIÁVEIS PRINCIPAIS
# ============================================================================

print("\n" + "=" * 100)
print("[2/6] Identificando variáveis principais...")
print("=" * 100)

# Mapeamento baseado em VIGITEL padrão
# Fonte: Documentação VIGITEL e análise do dicionário

variaveis_importantes = {
    # TARGET
    'diabetes': 'q76',  # 1=sim, 2=não, 777=não sabe

    # DEMOGRÁFICAS
    'idade': 'idade',  # ou 'q6' se idade não existir
    'sexo': 'sexo',    # 1=masculino, 2=feminino

    # ANTROPOMÉTRICAS
    'peso': 'peso',    # peso em kg
    'altura': 'altura',  # altura em cm
    'imc': '_imc',     # IMC calculado (geralmente prefixo _)

    # CONDIÇÕES DE SAÚDE
    'hipertensao': 'q74',  # Pressão alta
    'colesterol': 'q75',   # Colesterol alto

    # ESTILO DE VIDA
    'fumante': 'r163',     # Fuma atualmente
    'ativ_fisica': 'r186', # Atividade física suficiente
    'frutas': 'r178',      # Consumo regular de frutas
    'verduras': 'r179',    # Consumo regular de verduras
    'refrigerante': 'r181', # Consumo regular de refrigerante

    # SAÚDE GERAL
    'saude_geral': 'q9',   # Auto-avaliação de saúde
}

print("\nVariáveis mapeadas:")
encontradas = []
nao_encontradas = []

for nome, var in variaveis_importantes.items():
    if var in df.columns:
        print(f"  ✓ {nome:20s} → {var:15s} (OK)")
        encontradas.append((nome, var))
    else:
        print(f"  ✗ {nome:20s} → {var:15s} (NÃO ENCONTRADA)")
        nao_encontradas.append((nome, var))

print(f"\nResumo: {len(encontradas)}/{len(variaveis_importantes)} variáveis encontradas")

# ============================================================================
# 3. CRIAR DATASET PROCESSADO
# ============================================================================

print("\n" + "=" * 100)
print("[3/6] Criando dataset processado...")
print("=" * 100)

# Criar novo dataframe
df_processed = pd.DataFrame()

# TARGET: Diabetes (binário)
if 'q76' in df.columns:
    df_processed['Diabetes'] = (df['q76'] == 1).astype(int)
    print(f"✓ Target 'Diabetes' criado")
    print(f"  Prevalência: {df_processed['Diabetes'].mean()*100:.1f}%")

# DEMOGRÁFICAS
# Idade
if 'idade' in df.columns:
    df_processed['Age'] = df['idade']
elif 'q6' in df.columns:
    df_processed['Age'] = df['q6']

if 'Age' in df_processed.columns:
    print(f"✓ Age: média {df_processed['Age'].mean():.1f} anos")

# Sexo (mapear para 0/1)
if 'sexo' in df.columns:
    df_processed['Gender'] = (df['sexo'] == 1).astype(int)  # 1=masc → 1, 2=fem → 0
    print(f"✓ Gender: {df_processed['Gender'].mean()*100:.1f}% masculino")

# ANTROPOMÉTRICAS
# Procurar IMC (várias possibilidades)
imc_cols = [col for col in df.columns if 'imc' in col.lower()]
if imc_cols:
    df_processed['BMI'] = df[imc_cols[0]]
    print(f"✓ BMI ({imc_cols[0]}): média {df_processed['BMI'].mean():.1f}")
elif 'peso' in df.columns and 'altura' in df.columns:
    # Calcular IMC
    df_processed['BMI'] = df['peso'] / ((df['altura']/100) ** 2)
    print(f"✓ BMI (calculado): média {df_processed['BMI'].mean():.1f}")

# CONDIÇÕES DE SAÚDE
# Hipertensão
if 'q74' in df.columns:
    df_processed['HighBP'] = (df['q74'] == 1).astype(int)
    print(f"✓ HighBP: {df_processed['HighBP'].mean()*100:.1f}%")

# Colesterol
if 'q75' in df.columns:
    df_processed['HighChol'] = (df['q75'] == 1).astype(int)
    print(f"✓ HighChol: {df_processed['HighChol'].mean()*100:.1f}%")

# ESTILO DE VIDA
# Fumante
fumante_cols = [col for col in df.columns if 'r163' in col or 'fumante' in col.lower()]
if fumante_cols:
    df_processed['Smoker'] = (df[fumante_cols[0]] == 1).astype(int)
    print(f"✓ Smoker: {df_processed['Smoker'].mean()*100:.1f}%")

# Atividade física
if 'r186' in df.columns:
    df_processed['PhysActivity'] = (df['r186'] == 1).astype(int)
elif any('ativ' in col.lower() for col in df.columns):
    ativ_col = [col for col in df.columns if 'ativ' in col.lower()][0]
    df_processed['PhysActivity'] = (df[ativ_col] == 1).astype(int)

if 'PhysActivity' in df_processed.columns:
    print(f"✓ PhysActivity: {df_processed['PhysActivity'].mean()*100:.1f}%")

# Frutas
if 'r178' in df.columns:
    df_processed['Fruits'] = (df['r178'] == 1).astype(int)
    print(f"✓ Fruits: {df_processed['Fruits'].mean()*100:.1f}%")

# Verduras
if 'r179' in df.columns:
    df_processed['Veggies'] = (df['r179'] == 1).astype(int)
    print(f"✓ Veggies: {df_processed['Veggies'].mean()*100:.1f}%")

# Saúde geral (converter escala 1-5 para 1-5, onde 1=excelente, 5=ruim)
if 'q9' in df.columns:
    # Limpar valores especiais (777, 888, 999)
    df_processed['GenHlth'] = df['q9'].replace({777: np.nan, 888: np.nan, 999: np.nan})
    # Inverter se necessário (verificar se 1=excelente ou 1=ruim)
    print(f"✓ GenHlth: média {df_processed['GenHlth'].mean():.1f}")

print(f"\n✓ Dataset processado criado: {len(df_processed)} linhas, {len(df_processed.columns)} colunas")

# ============================================================================
# 4. LIMPEZA E VALIDAÇÃO
# ============================================================================

print("\n" + "=" * 100)
print("[4/6] Limpeza e validação...")
print("=" * 100)

# Remover valores faltantes
print(f"\nValores faltantes antes:")
print(df_processed.isnull().sum())

# Remover linhas com target faltante
df_processed = df_processed.dropna(subset=['Diabetes'])
print(f"\n✓ Após remover linhas sem target: {len(df_processed)} registros")

# Verificar ranges válidos
if 'BMI' in df_processed.columns:
    # BMI válido: 10-60
    df_processed = df_processed[(df_processed['BMI'] >= 10) & (df_processed['BMI'] <= 60)]
    print(f"✓ Após filtrar BMI (10-60): {len(df_processed)} registros")

if 'Age' in df_processed.columns:
    # Idade válida: 18-100
    df_processed = df_processed[(df_processed['Age'] >= 18) & (df_processed['Age'] <= 100)]
    print(f"✓ Após filtrar idade (18-100): {len(df_processed)} registros")

print(f"\n✓ Dataset final: {len(df_processed):,} registros válidos")

# ============================================================================
# 5. ESTATÍSTICAS FINAIS
# ============================================================================

print("\n" + "=" * 100)
print("[5/6] Estatísticas do dataset processado")
print("=" * 100)

print(f"\n📊 Dimensões finais:")
print(f"  Registros: {len(df_processed):,}")
print(f"  Features: {len(df_processed.columns) - 1}")  # -1 para excluir target

print(f"\n🎯 Target (Diabetes):")
print(f"  Sem diabetes: {(df_processed['Diabetes'] == 0).sum():,} ({(df_processed['Diabetes'] == 0).mean()*100:.1f}%)")
print(f"  Com diabetes: {(df_processed['Diabetes'] == 1).sum():,} ({(df_processed['Diabetes'] == 1).mean()*100:.1f}%)")

print(f"\n📋 Features disponíveis:")
for col in df_processed.columns:
    if col != 'Diabetes':
        if df_processed[col].dtype in ['int64', 'float64']:
            print(f"  {col:20s}: média={df_processed[col].mean():.2f}, std={df_processed[col].std():.2f}")
        else:
            print(f"  {col:20s}: {df_processed[col].nunique()} valores únicos")

# ============================================================================
# 6. SALVAR DATASET PROCESSADO
# ============================================================================

print("\n" + "=" * 100)
print("[6/6] Salvando dataset processado...")
print("=" * 100)

# Salvar em CSV
output_file = 'dados_vigitel/vigitel_2023_processado.csv'
df_processed.to_csv(output_file, index=False)
print(f"✓ Salvo em: {output_file}")

# Salvar metadados
metadata = {
    'registros_total': len(df_processed),
    'features': len(df_processed.columns) - 1,
    'prevalencia_diabetes': df_processed['Diabetes'].mean(),
    'colunas': df_processed.columns.tolist(),
    'data_processamento': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

import json
with open('dados_vigitel/metadata_processado.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"✓ Metadados salvos")

# Estatísticas descritivas
df_processed.describe().to_csv('dados_vigitel/estatisticas_descritivas.csv')
print(f"✓ Estatísticas descritivas salvas")

# ============================================================================
# RESUMO FINAL
# ============================================================================

print("\n" + "=" * 100)
print("✅ PROCESSAMENTO COMPLETO!")
print("=" * 100)

print(f"""
📊 DATASET VIGITEL 2023 PROCESSADO:

  Arquivo: {output_file}
  Registros: {len(df_processed):,}
  Features: {len(df_processed.columns) - 1}
  Prevalência diabetes: {df_processed['Diabetes'].mean()*100:.1f}%

📋 FEATURES CRIADAS:
  {', '.join([col for col in df_processed.columns if col != 'Diabetes'])}

🎯 PRÓXIMOS PASSOS:
  1. Treinar modelo XGBoost em dados brasileiros
  2. Validar com cross-validation
  3. Comparar com modelo americano (BRFSS)
  4. Análise comparativa cross-cultural

💡 PARA TREINAR O MODELO:
  python treinar_modelo_vigitel.py

🎊 DADOS BRASILEIROS PRONTOS!
""")

print("=" * 100)
