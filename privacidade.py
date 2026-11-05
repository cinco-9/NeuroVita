# -*- coding: utf-8 -*-
"""
TERMO DE CONSENTIMENTO E POLÍTICA DE PRIVACIDADE (LGPD)
Ao mudar o texto de forma relevante, aumente VERSAO_TERMO: todos os usuários
precisarão aceitar de novo no próximo acesso.
"""

import os

VERSAO_TERMO = "1.1"

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
- Dados do diário, se você usar: alimentação (alimentos, quantidades, calorias e nutrientes), atividade física, glicemia, peso, pressão arterial, sono, sintomas e hábitos (álcool, cigarro, bebidas adoçadas).
- Produtos que você cadastrar a partir do rótulo (código de barras e tabela nutricional).
- Contas de médico: número e UF do CRM.

Dados de saúde são **dados pessoais sensíveis** pela Lei Geral de Proteção de Dados (Lei 13.709/2018, art. 11).
O tratamento se baseia no seu **consentimento específico**, dado ao marcar a caixa de aceite.

**2. Para que os dados são usados**
Somente para calcular o seu risco, preencher os formulários com o seu perfil, registrar a sua rotina e mostrar o seu
histórico e a sua evolução. Os dados não são vendidos e não são usados para publicidade.

Ao escanear um código de barras, **apenas o número do código** é consultado no Open Food Facts (base pública de
produtos alimentícios); nenhum dado seu é enviado.

**3. Quem tem acesso**
Cada usuário só consegue ver os próprios dados. O acesso administrativo ao banco é restrito ao autor do projeto.

**Compartilhamento com médico:** só acontece se você escolher, informando o email de uma conta de médico em
*Meu Perfil*. O médico passa a ver, **somente para leitura**, seu perfil, suas avaliações de risco e seu diário.
Você pode parar de compartilhar a qualquer momento, e o acesso termina na hora. O app **não verifica o CRM**
informado pelos médicos: compartilhe apenas com profissionais que você conhece.

**4. Onde os dados ficam**
O aplicativo roda no Streamlit Community Cloud e os dados são guardados no Supabase. Os servidores desses serviços
podem estar **fora do Brasil** (por exemplo, nos EUA). Ao aceitar, você concorda com essa transferência internacional
(LGPD, art. 33). A comunicação entre o seu aparelho e os servidores é criptografada (HTTPS).

**5. Por quanto tempo**
Enquanto a sua conta existir. Ao excluir a conta, o perfil, todas as avaliações, o diário, os produtos cadastrados
e os compartilhamentos com médicos são apagados definitivamente.

**6. Seus direitos (LGPD, art. 18)**
- Consultar os seus dados: páginas *Meu Perfil*, *Meu Diário* e *Histórico & Estatísticas*.
- Corrigir: página *Meu Perfil*; registros do diário podem ser excluídos um a um.
- Excluir a conta e todos os dados: botão *Excluir minha conta* em *Meu Perfil*.
- Revogar o compartilhamento com um médico: *Meu Perfil → Compartilhar com meu médico*.
- Revogar o consentimento: excluindo a conta, a qualquer momento.
- Outras solicitações ou dúvidas: {_contato}.

**Recomendação:** não informe dados de outras pessoas (por exemplo, de pacientes) sem a autorização delas.
"""
