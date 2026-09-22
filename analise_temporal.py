# -*- coding: utf-8 -*-
"""
MÓDULO DE ANÁLISE TEMPORAL
Gera gráficos de evolução do risco ao longo do tempo
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

def criar_grafico_evolucao(predicoes_usuario, nome_arquivo='evolucao_risco.png'):
    """
    Cria gráfico mostrando evolução do risco ao longo do tempo

    Args:
        predicoes_usuario: Lista de dicionários com predições do usuário
                          [{created_at, probabilidade, resultado, tipo_modelo}, ...]
        nome_arquivo: Nome do arquivo para salvar

    Returns:
        Caminho do arquivo gerado ou None se não houver dados suficientes
    """

    if len(predicoes_usuario) < 2:
        return None  # Precisa de pelo menos 2 predições

    # Converter para DataFrame
    df = pd.DataFrame(predicoes_usuario)

    # Converter created_at para datetime
    df['data'] = pd.to_datetime(df['created_at'])
    df = df.sort_values('data')

    # Criar figura
    fig, ax = plt.subplots(figsize=(12, 6))

    # Separar por tipo de modelo
    clinico = df[df['tipo_modelo'] == 'clinico']
    comportamental = df[df['tipo_modelo'] == 'comportamental']

    # Plotar linhas
    if len(clinico) > 0:
        ax.plot(clinico['data'], clinico['probabilidade'] * 100,
               marker='o', linestyle='-', linewidth=2.5, markersize=10,
               color='#1E88E5', label='Modelo Clínico', alpha=0.8)

    if len(comportamental) > 0:
        ax.plot(comportamental['data'], comportamental['probabilidade'] * 100,
               marker='s', linestyle='--', linewidth=2.5, markersize=10,
               color='#E53935', label='Modelo Comportamental', alpha=0.8)

    # Linha de threshold (35%)
    ax.axhline(y=35, color='orange', linestyle='--', linewidth=2,
              label='Threshold (35%)', alpha=0.7)

    # Zona de risco (acima de 35%)
    ax.axhspan(35, 100, color='red', alpha=0.1, label='Zona de Alto Risco')

    # Zona segura (abaixo de 35%)
    ax.axhspan(0, 35, color='green', alpha=0.1, label='Zona de Baixo Risco')

    # Formatação
    ax.set_xlabel('Data', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probabilidade de Diabetes (%)', fontsize=12, fontweight='bold')
    ax.set_title('Evolução do Risco de Diabetes ao Longo do Tempo',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

    # Limites
    ax.set_ylim(0, 100)

    # Formato de data no eixo X
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return nome_arquivo


def criar_grafico_progresso(predicoes_usuario, nome_arquivo='progresso.png'):
    """
    Cria gráfico de barras mostrando progresso (primeira vs última predição)

    Args:
        predicoes_usuario: Lista de predições
        nome_arquivo: Nome do arquivo

    Returns:
        Caminho do arquivo ou None
    """

    if len(predicoes_usuario) < 2:
        return None

    # Converter para DataFrame e ordenar
    df = pd.DataFrame(predicoes_usuario)
    df['data'] = pd.to_datetime(df['created_at'])
    df = df.sort_values('data')

    # Primeira e última predição
    primeira = df.iloc[0]
    ultima = df.iloc[-1]

    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 6))

    # Dados
    categorias = ['Primeira Avaliação\n' + primeira['data'].strftime('%d/%m/%Y'),
                  'Avaliação Atual\n' + ultima['data'].strftime('%d/%m/%Y')]
    valores = [primeira['probabilidade'] * 100, ultima['probabilidade'] * 100]

    # Cores baseadas no risco
    cores = []
    for v in valores:
        if v < 25:
            cores.append('#4CAF50')  # Verde
        elif v < 50:
            cores.append('#FFC107')  # Amarelo
        elif v < 75:
            cores.append('#FF9800')  # Laranja
        else:
            cores.append('#F44336')  # Vermelho

    # Criar barras
    bars = ax.bar(categorias, valores, color=cores, alpha=0.8, edgecolor='black', linewidth=2)

    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Linha de threshold
    ax.axhline(y=35, color='orange', linestyle='--', linewidth=2, label='Threshold (35%)')

    # Calcular diferença
    diferenca = ultima['probabilidade'] - primeira['probabilidade']
    if diferenca < 0:
        seta = '↓'
        cor_dif = 'green'
        texto_dif = f'Redução de {abs(diferenca)*100:.1f}%'
    elif diferenca > 0:
        seta = '↑'
        cor_dif = 'red'
        texto_dif = f'Aumento de {diferenca*100:.1f}%'
    else:
        seta = '='
        cor_dif = 'gray'
        texto_dif = 'Sem alteração'

    # Adicionar texto de diferença
    ax.text(0.5, max(valores) + 10, f'{seta} {texto_dif}',
           ha='center', fontsize=14, fontweight='bold', color=cor_dif,
           transform=ax.transData)

    # Formatação
    ax.set_ylabel('Probabilidade de Diabetes (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparação: Primeira vs Última Avaliação',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return nome_arquivo


def calcular_estatisticas_evolucao(predicoes_usuario):
    """
    Calcula estatísticas de evolução

    Args:
        predicoes_usuario: Lista de predições

    Returns:
        Dicionário com estatísticas
    """

    if len(predicoes_usuario) < 2:
        return None

    df = pd.DataFrame(predicoes_usuario)
    df['data'] = pd.to_datetime(df['created_at'])
    df = df.sort_values('data')

    primeira = df.iloc[0]
    ultima = df.iloc[-1]

    diferenca = (ultima['probabilidade'] - primeira['probabilidade']) * 100
    diferenca_pct = ((ultima['probabilidade'] - primeira['probabilidade']) / primeira['probabilidade']) * 100 if primeira['probabilidade'] > 0 else 0

    dias_decorridos = (ultima['data'] - primeira['data']).days

    return {
        'primeira_prob': primeira['probabilidade'] * 100,
        'ultima_prob': ultima['probabilidade'] * 100,
        'diferenca_absoluta': diferenca,
        'diferenca_percentual': diferenca_pct,
        'dias_decorridos': dias_decorridos,
        'total_predicoes': len(df),
        'media_prob': df['probabilidade'].mean() * 100,
        'melhorou': diferenca < 0,
        'piorou': diferenca > 0,
        'estavel': abs(diferenca) < 5
    }


if __name__ == "__main__":
    # Testar com dados simulados
    predicoes_teste = [
        {
            'created_at': '2024-01-01T10:00:00',
            'probabilidade': 0.65,
            'resultado': 1,
            'tipo_modelo': 'clinico'
        },
        {
            'created_at': '2024-02-01T10:00:00',
            'probabilidade': 0.55,
            'resultado': 1,
            'tipo_modelo': 'clinico'
        },
        {
            'created_at': '2024-03-01T10:00:00',
            'probabilidade': 0.45,
            'resultado': 1,
            'tipo_modelo': 'clinico'
        },
        {
            'created_at': '2024-04-01T10:00:00',
            'probabilidade': 0.30,
            'resultado': 0,
            'tipo_modelo': 'clinico'
        }
    ]

    criar_grafico_evolucao(predicoes_teste, 'teste_evolucao.png')
    criar_grafico_progresso(predicoes_teste, 'teste_progresso.png')
    stats = calcular_estatisticas_evolucao(predicoes_teste)
    print("Estatísticas:", stats)
    print("Gráficos de teste criados!")
