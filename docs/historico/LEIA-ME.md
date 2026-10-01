# Documentos históricos

Registros de etapas anteriores do projeto, mantidos para histórico. **Os números e conclusões daqui estão
desatualizados** e não devem ser usados no texto do TCC.

Principais diferenças em relação ao estado atual (ver o [README](../../README.md)):

- O modelo clínico atual é um **Random Forest** escolhido por validação cruzada aninhada (F1 0,69), e não o
  XGBoost com F1 0,72, cujo limiar havia sido escolhido no conjunto de teste.
- O modelo comportamental atual é um XGBoost treinado de fato no BRFSS (F1 0,46, AUC 0,82); a versão antiga do
  app usava regras fixas.
- A análise Brasil x EUA foi refeita: o mapeamento antigo do VIGITEL trocava variáveis (estado de saúde, pressão
  alta, peso) e incluía colesterol, que o VIGITEL não pergunta.
- Os gráficos SHAP foram removidos do app.
