# -*- coding: utf-8 -*-
"""
DOWNLOAD AUTOMÁTICO VIGITEL
Tenta baixar dados do Ministério da Saúde automaticamente
Autor: David Reis | 2026
"""

import requests
import os
from pathlib import Path

print("=" * 80)
print("DOWNLOAD AUTOMÁTICO - DADOS VIGITEL")
print("Ministério da Saúde do Brasil")
print("=" * 80)

# Criar pasta para dados
pasta_dados = Path("dados_vigitel")
pasta_dados.mkdir(exist_ok=True)
print(f"\n✓ Pasta criada: {pasta_dados}")

# URLs conhecidas (baseado em anos anteriores)
urls_possiveis = [
    # 2023
    "https://svs.aids.gov.br/download/Vigitel/Dados/VIGITEL-2023-peso-rake.csv",
    "https://svs.aids.gov.br/download/Vigitel/Dados/VIGITEL-2023.csv",
    "https://svs.aids.gov.br/download/Vigitel/Dados/vigitel_2023.csv",

    # 2022 (backup)
    "https://svs.aids.gov.br/download/Vigitel/Dados/VIGITEL-2022-peso-rake.csv",
    "https://svs.aids.gov.br/download/Vigitel/Dados/VIGITEL-2022.csv",

    # Dicionário
    "https://svs.aids.gov.br/download/Vigitel/Dicionario/Dicionario-de-dados-Vigitel-2023.xlsx",
]

print("\n🔍 Tentando baixar arquivos VIGITEL...")
print("-" * 80)

arquivos_baixados = []

for url in urls_possiveis:
    try:
        print(f"\nTentando: {url.split('/')[-1]}...", end=" ")

        # Tentar baixar
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            # Salvar arquivo
            nome_arquivo = url.split('/')[-1]
            caminho = pasta_dados / nome_arquivo

            with open(caminho, 'wb') as f:
                f.write(response.content)

            tamanho = len(response.content) / (1024*1024)  # MB
            print(f"✓ SUCESSO! ({tamanho:.1f} MB)")
            arquivos_baixados.append(str(caminho))
        else:
            print(f"✗ Não encontrado (HTTP {response.status_code})")

    except Exception as e:
        print(f"✗ Erro: {str(e)[:50]}")

print("\n" + "=" * 80)

if arquivos_baixados:
    print(f"✅ SUCESSO! {len(arquivos_baixados)} arquivo(s) baixado(s):")
    for arq in arquivos_baixados:
        print(f"  - {arq}")

    print("\n📊 PRÓXIMO PASSO:")
    print("Execute: python processar_vigitel.py")

else:
    print("⚠️ Nenhum arquivo baixado automaticamente")
    print("\n📥 DOWNLOAD MANUAL NECESSÁRIO:")
    print()
    print("1. Acesse: https://www.gov.br/saude/pt-br/composicao/svsa/inqueritos-de-saude/vigitel")
    print("2. Procure 'Base de Dados' ou 'Microdados'")
    print("3. Baixe VIGITEL 2023 (CSV ou DBF)")
    print(f"4. Coloque em: {pasta_dados.absolute()}")
    print()
    print("ALTERNATIVA:")
    print("- Acesse: https://svs.aids.gov.br/daent/cgdnt/vigitel/")
    print("- Navegue até 'download' ou 'dados'")
    print("- Baixe o arquivo mais recente")

print("=" * 80)
