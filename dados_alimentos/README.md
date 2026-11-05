# Dados de alimentos

`taco_alimentos.csv`: Tabela Brasileira de Composição de Alimentos (TACO), 4ª edição, NEPA/UNICAMP (2011),
a partir da versão processada em [brolesi/taco](https://github.com/brolesi/taco) (`data/processed/taco/taco_composicao.csv`).
Mantidas só as colunas usadas pelo app; valores por 100 g de parte comestível. Células vazias são nutrientes que a
TACO não analisou.

Os pesos típicos das medidas caseiras em `alimentos.py` (`MEDIDAS_CASEIRAS`) são a mediana, entre todos os alimentos,
do peso de cada medida na tabela de medidas caseiras da POF 2017-2018 (IBGE), também de
[brolesi/taco](https://github.com/brolesi/taco) (`data/processed/pof/pof_medidas_caseiras.csv`).
