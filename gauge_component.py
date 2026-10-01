# -*- coding: utf-8 -*-
"""
COMPONENTE GAUGE - MEDIDOR DE RISCO VISUAL
Cria gráfico de gauge mostrando nível de risco
"""

import io

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FUNDO = '#1b1b1b'
TEXTO = '#e8e8e8'
VERDE = '#4ade80'
VERMELHO = '#f87171'


def criar_gauge_risco(probabilidade, threshold):
    """
    Cria um medidor semicircular do risco de diabetes.

    Args:
        probabilidade: Float de 0.0 a 1.0
        threshold: corte do modelo; o valor fica vermelho a partir dele

    Returns:
        BytesIO com o PNG (em memória, para não haver arquivo compartilhado entre usuários)
    """
    risco_pct = probabilidade * 100

    fig, ax = plt.subplots(figsize=(6, 3.6))
    fig.patch.set_facecolor(FUNDO)
    ax.set_facecolor(FUNDO)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.45, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    r_outer = 1.0
    r_inner = 0.7

    zonas = [
        (0, 25, VERDE),
        (25, 50, '#fbbf24'),
        (50, 75, '#fb923c'),
        (75, 100, VERMELHO),
    ]

    for inicio, fim, cor in zonas:
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
        ax.fill(x, y, color=cor, alpha=0.85, edgecolor=FUNDO, linewidth=2)

    # Ponteiro (0% = 180°, 100% = 0°)
    angulo_ponteiro = np.pi * (100 - risco_pct) / 100
    ponteiro_x = [0, 0.85 * np.cos(angulo_ponteiro)]
    ponteiro_y = [0, 0.85 * np.sin(angulo_ponteiro)]
    ax.plot(ponteiro_x, ponteiro_y, color=TEXTO, linewidth=3, zorder=10, solid_capstyle='round')
    ax.add_patch(plt.Circle((0, 0), 0.08, color=TEXTO, zorder=12))

    for marc in [0, 25, 50, 75, 100]:
        ang = np.pi * (100 - marc) / 100
        ax.text(1.12 * np.cos(ang), 1.12 * np.sin(ang), f'{marc}%',
                ha='center', va='center', fontsize=9, color='#a3a3a3')

    cor_valor = VERMELHO if probabilidade >= threshold else VERDE
    ax.text(0, -0.28, f'{risco_pct:.1f}%',
            ha='center', va='center', fontsize=26, fontweight='bold', color=cor_valor)

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor=FUNDO)
    plt.close(fig)
    buffer.seek(0)
    return buffer


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
    for p in (0.25, 0.65):
        with open(f'teste_gauge_{int(p * 100)}.png', 'wb') as f:
            f.write(criar_gauge_risco(p, threshold=0.30).getvalue())
    criar_gauge_simples(0.45, 'teste_gauge_simples.png')
    print("Gauges de teste criados!")
