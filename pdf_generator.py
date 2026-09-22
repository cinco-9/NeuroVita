# -*- coding: utf-8 -*-
"""
MÓDULO DE GERAÇÃO DE RELATÓRIOS PDF
Gera relatórios profissionais das predições de diabetes
"""

from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from datetime import datetime
import matplotlib.pyplot as plt
import io
import os

class RelatorioPDF:
    """Classe para gerar relatórios em PDF"""

    def __init__(self, nome_arquivo):
        """Inicializa gerador de PDF"""
        self.nome_arquivo = nome_arquivo
        self.doc = SimpleDocTemplate(nome_arquivo, pagesize=A4)
        self.story = []
        self.styles = getSampleStyleSheet()
        self._criar_estilos_customizados()

    def _criar_estilos_customizados(self):
        """Cria estilos customizados"""
        # Título principal
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1E88E5'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Subtítulo
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#666666'),
            spaceAfter=20,
            alignment=TA_CENTER
        ))

        # Seção
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1E88E5'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))

        # Texto normal
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=11,
            leading=16,
            alignment=TA_JUSTIFY
        ))

    def adicionar_cabecalho(self, nome_paciente, data):
        """Adiciona cabeçalho do relatório"""
        # Título
        titulo = Paragraph("🩺 Relatório de Predição de Diabetes", self.styles['CustomTitle'])
        self.story.append(titulo)

        # Subtítulo
        subtitulo = Paragraph("Sistema de IA para Estimativa de Risco de Diabetes Tipo 2", self.styles['CustomSubtitle'])
        self.story.append(subtitulo)

        self.story.append(Spacer(1, 0.3*inch))

        # Informações do paciente
        data_info = [
            ['Paciente:', nome_paciente],
            ['Data do Relatório:', data],
            ['Gerado por:', 'Sistema TCC - David Reis']
        ]

        tabela = Table(data_info, colWidths=[2*inch, 4*inch])
        tabela.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1E88E5')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))

        self.story.append(tabela)
        self.story.append(Spacer(1, 0.4*inch))

    def adicionar_resultado(self, tipo_modelo, resultado, probabilidade):
        """Adiciona resultado da predição"""
        # Título da seção
        self.story.append(Paragraph("📊 Resultado da Análise", self.styles['SectionHeader']))

        # Cor baseada no resultado
        if resultado == 1:
            cor_fundo = colors.HexColor('#ffebee')
            cor_texto = colors.HexColor('#c62828')
            texto_resultado = "⚠️ RISCO ELEVADO DE DIABETES"
        else:
            cor_fundo = colors.HexColor('#e8f5e9')
            cor_texto = colors.HexColor('#2e7d32')
            texto_resultado = "✅ RISCO BAIXO DE DIABETES"

        # Tabela de resultado
        data_resultado = [
            ['Modelo Utilizado:', tipo_modelo],
            ['Classificação:', texto_resultado],
            ['Probabilidade:', f'{probabilidade*100:.1f}%']
        ]

        tabela_resultado = Table(data_resultado, colWidths=[2*inch, 4*inch])
        tabela_resultado.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), cor_fundo),
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 11),
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 11),
            ('FONT', (1, 1), (1, 1), 'Helvetica-Bold', 14),
            ('TEXTCOLOR', (1, 1), (1, 1), cor_texto),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))

        self.story.append(tabela_resultado)
        self.story.append(Spacer(1, 0.3*inch))

    def adicionar_dados_paciente(self, dados):
        """Adiciona dados do paciente ao relatório"""
        self.story.append(Paragraph("👤 Dados do Paciente", self.styles['SectionHeader']))

        # Criar tabela com dados
        data_paciente = []
        for chave, valor in dados.items():
            if isinstance(valor, float):
                valor_formatado = f"{valor:.2f}"
            elif isinstance(valor, bool):
                valor_formatado = "Sim" if valor else "Não"
            else:
                valor_formatado = str(valor)

            data_paciente.append([chave + ':', valor_formatado])

        tabela = Table(data_paciente, colWidths=[2.5*inch, 3.5*inch])
        tabela.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ]))

        self.story.append(tabela)
        self.story.append(Spacer(1, 0.3*inch))

    def adicionar_recomendacoes(self, recomendacoes):
        """Adiciona recomendações personalizadas"""
        if not recomendacoes:
            return

        self.story.append(Paragraph("🎯 Recomendações Personalizadas", self.styles['SectionHeader']))

        for i, rec in enumerate(recomendacoes, 1):
            texto = f"{i}. {rec}"
            p = Paragraph(texto, self.styles['CustomBody'])
            self.story.append(p)
            self.story.append(Spacer(1, 0.1*inch))

        self.story.append(Spacer(1, 0.2*inch))

    def adicionar_grafico(self, fig_matplotlib, largura=5*inch, altura=3*inch):
        """Adiciona gráfico matplotlib ao PDF"""
        # Salvar figura em buffer
        img_buffer = io.BytesIO()
        fig_matplotlib.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)

        # Adicionar imagem
        img = Image(img_buffer, width=largura, height=altura)
        self.story.append(img)
        self.story.append(Spacer(1, 0.2*inch))

        plt.close(fig_matplotlib)

    def adicionar_rodape(self):
        """Adiciona rodapé com avisos"""
        self.story.append(Spacer(1, 0.5*inch))

        aviso = """
        <b>⚠️ AVISO IMPORTANTE:</b><br/>
        Este relatório foi gerado por um sistema de inteligência artificial para fins educacionais
        e de triagem inicial. Os resultados NÃO substituem uma consulta médica profissional.
        Procure um médico endocrinologista para diagnóstico e tratamento adequados.
        """

        p = Paragraph(aviso, self.styles['CustomBody'])

        # Box de aviso
        tabela_aviso = Table([[p]], colWidths=[6.5*inch])
        tabela_aviso.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff3cd')),
            ('BORDER', (0, 0), (-1, -1), 2, colors.HexColor('#ffc107')),
            ('PADDING', (0, 0), (-1, -1), 12),
        ]))

        self.story.append(tabela_aviso)

        # Assinatura
        self.story.append(Spacer(1, 0.3*inch))
        assinatura = Paragraph(
            f"<i>Relatório gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')}</i>",
            self.styles['CustomBody']
        )
        self.story.append(assinatura)

    def gerar(self):
        """Gera o arquivo PDF"""
        self.doc.build(self.story)
        return self.nome_arquivo


def gerar_relatorio_predicao(
    nome_arquivo,
    nome_paciente,
    tipo_modelo,
    resultado,
    probabilidade,
    dados_paciente,
    recomendacoes,
    grafico_shap=None
):
    """
    Função helper para gerar relatório completo

    Args:
        nome_arquivo: Caminho do arquivo PDF
        nome_paciente: Nome do paciente
        tipo_modelo: "Clínico" ou "Comportamental"
        resultado: 0 ou 1
        probabilidade: 0.0 a 1.0
        dados_paciente: Dict com dados
        recomendacoes: Lista de recomendações
        grafico_shap: Figura matplotlib opcional

    Returns:
        Caminho do arquivo gerado
    """
    pdf = RelatorioPDF(nome_arquivo)

    # Cabecalho
    pdf.adicionar_cabecalho(
        nome_paciente=nome_paciente,
        data=datetime.now().strftime('%d/%m/%Y')
    )

    # Resultado
    pdf.adicionar_resultado(tipo_modelo, resultado, probabilidade)

    # Dados do paciente
    pdf.adicionar_dados_paciente(dados_paciente)

    # Gráfico SHAP (se houver)
    if grafico_shap is not None:
        pdf.story.append(Paragraph("📈 Explicação da Predição (SHAP)", pdf.styles['SectionHeader']))
        pdf.adicionar_grafico(grafico_shap)

    # Recomendações
    pdf.adicionar_recomendacoes(recomendacoes)

    # Rodapé
    pdf.adicionar_rodape()

    # Gerar PDF
    return pdf.gerar()
