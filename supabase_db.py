# -*- coding: utf-8 -*-
"""
MÓDULO SUPABASE - INTEGRAÇÃO COM BANCO DE DADOS
Gerencia conexão e operações com Supabase
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv
from supabase import create_client, Client
import json

# Carregar variáveis de ambiente
load_dotenv()

class SupabaseDB:
    """Classe para gerenciar operações com Supabase"""

    def __init__(self):
        """Inicializa conexão com Supabase"""
        self.url = os.getenv('SUPABASE_URL')
        self.key = os.getenv('SUPABASE_KEY')
        self.client: Optional[Client] = None
        self.conectado = False

        # Tentar conectar
        self._conectar()

    def _conectar(self):
        """Conecta ao Supabase"""
        try:
            if self.url and self.key:
                self.client = create_client(self.url, self.key)
                self.conectado = True
                print("✅ Conectado ao Supabase!")
            else:
                print("⚠️ Credenciais do Supabase não configuradas (.env)")
                self.conectado = False
        except Exception as e:
            print(f"❌ Erro ao conectar Supabase: {e}")
            self.conectado = False

    def salvar_predicao(
        self,
        tipo_modelo: str,
        dados_input: Dict,
        resultado: int,
        probabilidade: float,
        session_id: str = None,
        user_agent: str = None,
        user_id: str = None
    ) -> bool:
        """
        Salva uma predição no banco

        Args:
            tipo_modelo: 'clinico' ou 'comportamental'
            dados_input: Dicionário com os dados de entrada
            resultado: 0 (baixo risco) ou 1 (alto risco)
            probabilidade: Probabilidade de diabetes (0.0 a 1.0)
            session_id: ID da sessão do usuário
            user_agent: User agent do navegador
            user_id: ID do usuário logado (UUID)

        Returns:
            True se salvou com sucesso, False caso contrário
        """
        if not self.conectado or not self.client:
            return False

        try:
            data = {
                'tipo_modelo': tipo_modelo,
                'dados_input': json.dumps(dados_input),
                'resultado': resultado,
                'user_id': user_id,
                'probabilidade': probabilidade,
                'session_id': session_id,
                'user_agent': user_agent
            }

            response = self.client.table('predicoes').insert(data).execute()
            return True

        except Exception as e:
            print(f"❌ Erro ao salvar predição: {e}")
            return False

    def obter_ultimas_predicoes(
        self,
        limite: int = 10,
        tipo_modelo: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Obtém as últimas predições do banco

        Args:
            limite: Número máximo de predições a retornar
            tipo_modelo: Filtrar por tipo ('clinico', 'comportamental' ou None para todos)
            user_id: Filtrar por usuário (opcional)

        Returns:
            Lista de predições
        """
        if not self.conectado or not self.client:
            return []

        try:
            query = self.client.table('predicoes').select('*')

            if user_id:
                query = query.eq('user_id', user_id)

            if tipo_modelo:
                query = query.eq('tipo_modelo', tipo_modelo)

            response = query.order('created_at', desc=True).limit(limite).execute()

            # Converter JSON strings de volta para dicts
            predicoes = []
            for item in response.data:
                predicao = item.copy()
                predicao['dados_input'] = json.loads(item['dados_input'])
                predicoes.append(predicao)

            return predicoes

        except Exception as e:
            print(f"❌ Erro ao obter predições: {e}")
            return []

    def obter_estatisticas(self) -> Dict:
        """
        Obtém estatísticas das predições

        Returns:
            Dicionário com estatísticas
        """
        if not self.conectado or not self.client:
            return {
                'total': 0,
                'alto_risco': 0,
                'baixo_risco': 0,
                'percent_alto_risco': 0,
                'media_probabilidade': 0,
                'por_tipo': {}
            }

        try:
            # Total de predições
            response = self.client.table('predicoes').select('*', count='exact').execute()
            total = response.count

            # Predições por resultado
            alto_risco = self.client.table('predicoes').select('*', count='exact').eq('resultado', 1).execute().count
            baixo_risco = total - alto_risco

            # Média de probabilidade
            all_data = self.client.table('predicoes').select('probabilidade').execute()
            if all_data.data:
                media_prob = sum(p['probabilidade'] for p in all_data.data) / len(all_data.data)
            else:
                media_prob = 0

            # Por tipo de modelo
            clinico = self.client.table('predicoes').select('*', count='exact').eq('tipo_modelo', 'clinico').execute().count
            comportamental = self.client.table('predicoes').select('*', count='exact').eq('tipo_modelo', 'comportamental').execute().count

            return {
                'total': total,
                'alto_risco': alto_risco,
                'baixo_risco': baixo_risco,
                'percent_alto_risco': (alto_risco / total * 100) if total > 0 else 0,
                'media_probabilidade': media_prob,
                'por_tipo': {
                    'clinico': clinico,
                    'comportamental': comportamental
                }
            }

        except Exception as e:
            print(f"❌ Erro ao obter estatísticas: {e}")
            return {
                'total': 0,
                'alto_risco': 0,
                'baixo_risco': 0,
                'percent_alto_risco': 0,
                'media_probabilidade': 0,
                'por_tipo': {}
            }

    def obter_predicoes_por_data(self, dias: int = 7) -> Dict:
        """
        Obtém contagem de predições por data (últimos N dias)

        Args:
            dias: Número de dias para considerar

        Returns:
            Dicionário com datas e contagens
        """
        if not self.conectado or not self.client:
            return {}

        try:
            from datetime import timedelta
            data_inicio = datetime.now() - timedelta(days=dias)

            response = self.client.table('predicoes').select('created_at').gte('created_at', data_inicio.isoformat()).execute()

            # Agrupar por data
            por_data = {}
            for item in response.data:
                data = datetime.fromisoformat(item['created_at'].replace('Z', '+00:00')).date()
                data_str = data.isoformat()
                por_data[data_str] = por_data.get(data_str, 0) + 1

            return por_data

        except Exception as e:
            print(f"❌ Erro ao obter predições por data: {e}")
            return {}

# Instância global
db = SupabaseDB()
