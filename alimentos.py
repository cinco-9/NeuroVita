# -*- coding: utf-8 -*-
"""
ALIMENTOS - TACO, MEDIDAS CASEIRAS E CÓDIGO DE BARRAS
- TACO 4ª ed. (NEPA/UNICAMP), versão processada por github.com/brolesi/taco
- Medidas caseiras: mediana dos pesos da POF 2017-2018 (IBGE) para cada medida
- Produtos industrializados: Open Food Facts (base aberta, preenchida pela comunidade)
Todos os valores nutricionais são por 100 g (ou 100 ml, para líquidos).
"""

import os
import unicodedata
from typing import Dict, Optional

import httpx
import pandas as pd
import streamlit as st

ARQUIVO_TACO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados_alimentos", "taco_alimentos.csv")

NUTRIENTES = ['kcal', 'carboidratos_g', 'acucares_g', 'proteinas_g', 'gorduras_g', 'fibra_g', 'sodio_mg']

# Peso típico de cada medida (mediana da POF/IBGE entre todos os alimentos).
# É uma aproximação: o app mostra os gramas resultantes e o paciente pode ajustar.
MEDIDAS_CASEIRAS = {
    "Gramas (g) / mililitros (ml)": 1,
    "Colher de chá": 5,
    "Colher de sobremesa": 12.5,
    "Colher de sopa": 25,
    "Colher de servir / de arroz": 50,
    "Escumadeira": 70,
    "Concha": 110,
    "Pedaço": 40,
    "Fatia": 40,
    "Unidade": 60,
    "Bife / filé": 100,
    "Xícara de café": 50,
    "Copo americano": 150,
    "Xícara de chá": 200,
    "Copo médio / requeijão": 240,
    "Copo grande": 300,
    "Prato raso": 140,
    "Prato fundo": 300,
}

OFF_URL = "https://world.openfoodfacts.org/api/v2/product/{codigo}.json"
OFF_HEADERS = {"User-Agent": "NeuroVita/1.0 (TCC; github.com/cinco-9/NeuroVita)"}


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acento.lower()


@st.cache_data
def carregar_taco() -> pd.DataFrame:
    taco = pd.read_csv(ARQUIVO_TACO, encoding="utf-8")
    taco['busca'] = taco['descricao'].map(_normalizar)
    return taco


def buscar_taco(termo: str, limite: int = 30) -> pd.DataFrame:
    """Alimentos cuja descrição contém todas as palavras digitadas (sem acento/maiúscula)"""
    taco = carregar_taco()
    palavras = _normalizar(termo).split()
    if not palavras:
        return taco.iloc[0:0]
    mascara = pd.Series(True, index=taco.index)
    for palavra in palavras:
        mascara &= taco['busca'].str.contains(palavra, regex=False)
    resultado = taco[mascara]
    # Descrições mais curtas primeiro: "Arroz, tipo 1, cozido" antes de preparações compostas
    return resultado.loc[resultado['descricao'].str.len().sort_values().index].head(limite)


def nutrientes_taco(linha: pd.Series) -> Dict:
    """Nutrientes por 100 g de um alimento da TACO; None quando a TACO não analisou"""
    valores = {}
    for chave in NUTRIENTES:
        valor = linha.get(chave)
        valores[chave] = None if valor is None or pd.isna(valor) else float(valor)
    return valores


def calcular_porcao(por_100g: Dict, gramas: float) -> Dict:
    """Converte valores por 100 g para a quantidade consumida (nutriente ausente conta como 0)"""
    fator = gramas / 100
    return {chave: round((por_100g.get(chave) or 0) * fator, 1) for chave in NUTRIENTES}


@st.cache_data(show_spinner=False, max_entries=50)
def ler_codigo_barras(imagem: bytes) -> Optional[str]:
    """Lê o código de barras (EAN/UPC) de uma foto; None se não encontrar"""
    import io
    import zxingcpp
    from PIL import Image, ImageOps

    foto = ImageOps.exif_transpose(Image.open(io.BytesIO(imagem))).convert("RGB")
    # Fotos de celular são grandes; reduzir acelera e não atrapalha a leitura
    foto.thumbnail((1600, 1600))
    formatos = (zxingcpp.BarcodeFormat.EAN13 | zxingcpp.BarcodeFormat.EAN8
                | zxingcpp.BarcodeFormat.UPCA | zxingcpp.BarcodeFormat.UPCE)
    for tentativa in (foto, ImageOps.grayscale(foto)):
        resultados = zxingcpp.read_barcodes(tentativa, formats=formatos)
        if resultados:
            return resultados[0].text
    return None


def codigo_valido(codigo: str) -> bool:
    return codigo.isdigit() and len(codigo) in (8, 12, 13, 14)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def buscar_open_food_facts(codigo: str) -> Optional[Dict]:
    """Consulta um código de barras no Open Food Facts.

    Só o número do código é enviado; nenhum dado do usuário sai do app.
    Retorna None se o produto não existir ou não tiver informação nutricional.
    Levanta exceção se o serviço estiver fora do ar.
    """
    resposta = httpx.get(
        OFF_URL.format(codigo=codigo),
        params={"fields": "product_name,product_name_pt,brands,quantity,serving_size,"
                          "serving_quantity,nutriments,image_front_small_url"},
        headers=OFF_HEADERS,
        timeout=10,
    )
    if resposta.status_code == 404:
        return None
    resposta.raise_for_status()
    dados = resposta.json()
    if dados.get("status") != 1:
        return None

    produto = dados.get("product", {})
    n = produto.get("nutriments", {})
    if n.get("energy-kcal_100g") is None and n.get("carbohydrates_100g") is None:
        return None

    def valor(chave, fator=1):
        v = n.get(chave)
        try:
            return None if v is None else round(float(v) * fator, 2)
        except (TypeError, ValueError):
            return None

    nome = produto.get("product_name_pt") or produto.get("product_name") or "Produto sem nome"
    if produto.get("brands"):
        nome = f"{nome} ({produto['brands'].split(',')[0].strip()})"

    try:
        porcao = float(produto.get("serving_quantity") or 0) or None
    except (TypeError, ValueError):
        porcao = None

    return {
        'nome': nome,
        'porcao_g': porcao,
        'porcao_texto': produto.get("serving_size"),
        'imagem': produto.get("image_front_small_url"),
        'por_100g': {
            'kcal': valor("energy-kcal_100g"),
            'carboidratos_g': valor("carbohydrates_100g"),
            'acucares_g': valor("sugars_100g"),
            'proteinas_g': valor("proteins_100g"),
            'gorduras_g': valor("fat_100g"),
            'fibra_g': valor("fiber_100g"),
            'sodio_mg': valor("sodium_100g", 1000),  # OFF guarda sódio em gramas
        },
    }

