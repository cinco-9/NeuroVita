# -*- coding: utf-8 -*-
"""
COMPONENTE GAUGE - MEDIDOR DE RISCO VISUAL
Cria gráfico de gauge mostrando nível de risco
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def criar_gauge_risco(probabilidade, nome_arquivo='gauge_risco.png'):
    """
    Cria um gauge (medidor semicircular) mostrando risco de diabetes

    Args:
        probabilidade: Float de 0.0 a 1.0
        nome_arquivo: Nome do arquivo para salvar

    Returns:
        Caminho do arquivo gerado
    """

    # Converter para porcentagem
    risco_pct = probabilidade * 100

    # Criar figura
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Definir zonas de risco
    # Baixo: 0-25% (verde)
    # Moderado: 25-50% (amarelo)
    # Alto: 50-75% (laranja)
    # Muito Alto: 75-100% (vermelho)

    # Criar arco semicircular (de 0° a 180°)
    theta = np.linspace(0, np.pi, 100)

    # Raio externo e interno
    r_outer = 1.0
    r_inner = 0.7

    # Desenhar zonas coloridas
    zonas = [
        (0, 25, '#4CAF50', 'Baixo'),      # Verde
        (25, 50, '#FFC107', 'Moderado'),  # Amarelo
        (50, 75, '#FF9800', 'Alto'),      # Laranja
        (75, 100, '#F44336', 'Muito Alto') # Vermelho
    ]

    for inicio, fim, cor, label in zonas:
        # Converter porcentagem para ângulo (0% = 0°, 100% = 180°)
        theta_inicio = np.pi * (100 - fim) / 100
        theta_fim = np.pi * (100 - inicio) / 100

        # Criar arco
        theta_zona = np.linspace(theta_inicio, theta_fim, 50)

        # Coordenadas externas
        x_outer = r_outer * np.cos(theta_zona)
        y_outer = r_outer * np.sin(theta_zona)

        # Coordenadas internas (reverso)
        x_inner = r_inner * np.cos(theta_zona[::-1])
        y_inner = r_inner * np.sin(theta_zona[::-1])

        # Combinar
        x = np.concatenate([x_outer, x_inner])
        y = np.concatenate([y_outer, y_inner])

        # Desenhar zona
        ax.fill(x, y, color=cor, alpha=0.8, edgecolor='white', linewidth=2)

    # Desenhar ponteiro (agulha)
    # Converter risco para ângulo (0% = 180°, 100% = 0°)
    angulo_ponteiro = np.pi * (100 - risco_pct) / 100

    # Coordenadas do ponteiro
    ponteiro_x = [0, 0.85 * np.cos(angulo_ponteiro)]
    ponteiro_y = [0, 0.85 * np.sin(angulo_ponteiro)]

    # Desenhar ponteiro
    ax.plot(ponteiro_x, ponteiro_y, color='black', linewidth=4, zorder=10)
    ax.plot(ponteiro_x, ponteiro_y, color='white', linewidth=2, zorder=11)

    # Círculo central
    circle = plt.Circle((0, 0), 0.1, color='#333', zorder=12)
    ax.add_patch(circle)

    # Adicionar marcações de porcentagem
    marcacoes = [0, 25, 50, 75, 100]
    for marc in marcacoes:
        ang = np.pi * (100 - marc) / 100
        x_marc = 1.1 * np.cos(ang)
        y_marc = 1.1 * np.sin(ang)
        ax.text(x_marc, y_marc, f'{marc}%',
               ha='center', va='center', fontsize=10, fontweight='bold')

    # Título com valor
    if risco_pct < 25:
        cor_titulo = '#4CAF50'
        nivel = 'BAIXO'
    elif risco_pct < 50:
        cor_titulo = '#FFC107'
        nivel = 'MODERADO'
    elif risco_pct < 75:
        cor_titulo = '#FF9800'
        nivel = 'ALTO'
    else:
        cor_titulo = '#F44336'
        nivel = 'MUITO ALTO'

    # Valor central
    ax.text(0, -0.3, f'{risco_pct:.1f}%',
           ha='center', va='center', fontsize=32, fontweight='bold', color=cor_titulo)

    ax.text(0, -0.5, f'Risco {nivel}',
           ha='center', va='center', fontsize=16, fontweight='bold', color=cor_titulo)

    # Título superior
    ax.text(0, 1.3, 'Risco de Diabetes Tipo 2',
           ha='center', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return nome_arquivo


def criar_gauge_simples(probabilidade, nome_arquivo='gauge_simples.png'):
    """
    Cria um gauge mais simples e minimalista

    Args:
        probabilidade: Float de 0.0 a 1.0
        nome_arquivo: Nome do arquivo para salvar

    Returns:
        Caminho do arquivo gerado
    """

    risco_pct = probabilidade * 100

    # Criar figura
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Barra de fundo (cinza)
    ax.barh(0.5, 100, height=0.3, color='#E0E0E0', zorder=1)

    # Barra de progresso (colorida)
    if risco_pct < 25:
        cor = '#4CAF50'
    elif risco_pct < 50:
        cor = '#FFC107'
    elif risco_pct < 75:
        cor = '#FF9800'
    else:
        cor = '#F44336'

    ax.barh(0.5, risco_pct, height=0.3, color=cor, zorder=2)

    # Marcações
    for marc in [0, 25, 50, 75, 100]:
        ax.axvline(marc, color='white', linewidth=2, zorder=3)
        ax.text(marc, 0.1, f'{marc}%', ha='center', fontsize=9)

    # Valor atual
    ax.text(risco_pct, 0.9, f'{risco_pct:.1f}%',
           ha='center', fontsize=20, fontweight='bold', color=cor)

    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return nome_arquivo


if __name__ == "__main__":
    # Testar
    criar_gauge_risco(0.25, 'teste_gauge_25.png')
    criar_gauge_risco(0.65, 'teste_gauge_65.png')
    criar_gauge_simples(0.45, 'teste_gauge_simples.png')
    print("Gauges de teste criados!")
