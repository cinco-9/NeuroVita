# -*- coding: utf-8 -*-
"""
MÓDULO SHAP - EXPLICABILIDADE DE PREDIÇÕES
Gera gráficos de explicação individuais
"""

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def gerar_grafico_shap_individual(modelo, X_input, feature_names, nome_arquivo='shap_plot.png'):
    """
    Gera gráfico SHAP waterfall para uma predição individual

    Args:
        modelo: Modelo treinado (XGBoost, LightGBM, etc.)
        X_input: Array numpy com os dados de entrada (1 amostra)
        feature_names: Lista com nomes das features
        nome_arquivo: Nome do arquivo para salvar

    Returns:
        Caminho do arquivo gerado
    """
    try:
        # Criar explainer
        explainer = shap.Explainer(modelo)

        # Calcular SHAP values
        shap_values = explainer(X_input)

        # Criar waterfall plot
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        plt.savefig(nome_arquivo, dpi=150, bbox_inches='tight')
        plt.close()

        return nome_arquivo

    except Exception as e:
        print(f"Erro ao gerar gráfico SHAP: {e}")
        return None

def gerar_grafico_shap_forcas(modelo, X_input, feature_names, nome_arquivo='shap_force.png'):
    """
    Gera gráfico SHAP de forças para uma predição individual

    Args:
        modelo: Modelo treinado
        X_input: Array numpy com os dados de entrada (1 amostra)
        feature_names: Lista com nomes das features
        nome_arquivo: Nome do arquivo para salvar

    Returns:
        Caminho do arquivo gerado
    """
    try:
        # Criar explainer
        explainer = shap.Explainer(modelo)

        # Calcular SHAP values
        shap_values = explainer(X_input)

        # Criar force plot
        plt.figure(figsize=(12, 3))
        shap.plots.force(shap_values[0], matplotlib=True, show=False)
        plt.tight_layout()
        plt.savefig(nome_arquivo, dpi=150, bbox_inches='tight')
        plt.close()

        return nome_arquivo

    except Exception as e:
        print(f"Erro ao gerar gráfico SHAP de forças: {e}")
        return None

def obter_top_features_shap(modelo, X_input, feature_names, top_n=5):
    """
    Retorna as top N features mais importantes para a predição

    Args:
        modelo: Modelo treinado
        X_input: Array numpy com os dados de entrada (1 amostra)
        feature_names: Lista com nomes das features
        top_n: Número de features para retornar

    Returns:
        Lista de tuplas (feature, valor, shap_value)
    """
    try:
        # Criar explainer
        explainer = shap.Explainer(modelo)

        # Calcular SHAP values
        shap_values = explainer(X_input)

        # Pegar valores SHAP da primeira (e única) amostra
        valores_shap = shap_values.values[0]

        # Criar DataFrame com features e seus valores SHAP
        df_shap = pd.DataFrame({
            'Feature': feature_names,
            'Valor': X_input[0],
            'SHAP_Value': valores_shap,
            'SHAP_Abs': np.abs(valores_shap)
        })

        # Ordenar por valor absoluto do SHAP (impacto)
        df_shap = df_shap.sort_values('SHAP_Abs', ascending=False)

        # Retornar top N
        top_features = []
        for idx, row in df_shap.head(top_n).iterrows():
            top_features.append({
                'feature': row['Feature'],
                'valor': row['Valor'],
                'shap_value': row['SHAP_Value'],
                'impacto': 'aumenta' if row['SHAP_Value'] > 0 else 'diminui'
            })

        return top_features

    except Exception as e:
        print(f"Erro ao obter top features SHAP: {e}")
        return []

def criar_grafico_barras_shap(modelo, X_input, feature_names, nome_arquivo='shap_bar.png'):
    """
    Cria gráfico de barras com importância SHAP

    Args:
        modelo: Modelo treinado
        X_input: Array numpy com os dados de entrada (1 amostra)
        feature_names: Lista com nomes das features
        nome_arquivo: Nome do arquivo para salvar

    Returns:
        Caminho do arquivo gerado
    """
    try:
        # Criar explainer
        explainer = shap.Explainer(modelo)

        # Calcular SHAP values
        shap_values = explainer(X_input)

        # Criar bar plot
        plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values[0], show=False)
        plt.tight_layout()
        plt.savefig(nome_arquivo, dpi=150, bbox_inches='tight')
        plt.close()

        return nome_arquivo

    except Exception as e:
        print(f"Erro ao gerar gráfico de barras SHAP: {e}")
        return None
