# -*- coding: utf-8 -*-
"""
TERMO DE CONSENTIMENTO E POLÍTICA DE PRIVACIDADE (LGPD)
Ao mudar o texto de forma relevante, aumente VERSAO_TERMO: todos os usuários
precisarão aceitar de novo no próximo acesso.
"""

import os

VERSAO_TERMO = "1.0"

# Email exibido para pedidos sobre os dados; configure em .env / Secrets do Streamlit
CONTATO_PRIVACIDADE = os.getenv("CONTATO_PRIVACIDADE", "")

_contato = (f"pelo email **{CONTATO_PRIVACIDADE}**" if CONTATO_PRIVACIDADE
            else "com o autor do projeto, David Reis")

TERMO_PRIVACIDADE = f"""
**Termo de Consentimento e Política de Privacidade** (versão {VERSAO_TERMO})

Este sistema foi desenvolvido como Trabalho de Conclusão de Curso, tem finalidade
**educacional e de triagem** e **não substitui avaliação médica**.

**1. Quais dados são coletados**
- Dados de conta: nome e email (a senha é guardada pelo serviço de autenticação, de forma criptografada, e não fica acessível ao autor).
- Dados de perfil, se você preencher: idade, sexo, altura, peso, IMC, histórico familiar de diabetes, alergias e medicações.
- Dados das avaliações: valores informados nos formulários (exames ou respostas do questionário), resultado e probabilidade calculados, data e hora.

Dados de saúde são **dados pessoais sensíveis** pela Lei Geral de Proteção de Dados (Lei 13.709/2018, art. 11).
O tratamento se baseia no seu **consentimento específico**, dado ao marcar a caixa de aceite.

**2. Para que os dados são usados**
Somente para calcular o seu risco, preencher os formulários com o seu perfil e mostrar o seu histórico de avaliações.
Os dados não são vendidos, não são compartilhados com terceiros e não são usados para publicidade.

**3. Quem tem acesso**
Cada usuário só consegue ver os próprios dados. O acesso administrativo ao banco é restrito ao autor do projeto.

**4. Onde os dados ficam**
O aplicativo roda no Streamlit Community Cloud e os dados são guardados no Supabase. Os servidores desses serviços
podem estar **fora do Brasil** (por exemplo, nos EUA). Ao aceitar, você concorda com essa transferência internacional
(LGPD, art. 33). A comunicação entre o seu aparelho e os servidores é criptografada (HTTPS).

**5. Por quanto tempo**
Enquanto a sua conta existir. Ao excluir a conta, o perfil e todas as avaliações são apagados definitivamente.

**6. Seus direitos (LGPD, art. 18)**
- Consultar os seus dados: páginas *Meu Perfil* e *Histórico & Estatísticas*.
- Corrigir: página *Meu Perfil*.
- Excluir a conta e todos os dados: botão *Excluir minha conta* em *Meu Perfil*.
- Revogar o consentimento: excluindo a conta, a qualquer momento.
- Outras solicitações ou dúvidas: {_contato}.

**Recomendação:** não informe dados de outras pessoas (por exemplo, de pacientes) sem a autorização delas.
"""
