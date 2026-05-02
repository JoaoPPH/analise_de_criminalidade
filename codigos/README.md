# Scripts de Preparação de Dados

Scripts que criei para juntar e converter os dados de criminalidade do SINESP VDE, baixados do site do Ministério da Justiça:

https://www.gov.br/mj/pt-br/assuntos/sua-seguranca/seguranca-publica/estatistica/dados-nacionais-1/base-de-dados-e-notas-metodologicas-dos-gestores-estaduais-sinesp-vde-2022-e-2023

## Scripts

### `juntar_excel.py`
Junta todos os arquivos `BancoVDE <ano>.xlsx` em um único CSV (`BancoVDE_Consolidado.csv`). Também adiciona uma coluna `Ano` extraída do nome de cada arquivo. Usei CSV porque o Excel tem limite de ~1 milhão de linhas.

### `converter_csv.py`
Converte o `BancoVDE_Consolidado.xlsx` para `.csv` com separador `;` e encoding `utf-8-sig`.

## Como usar

1. Instalar as dependências: `pip install pandas openpyxl`
2. Alterar a variável `diretorio` nos scripts para o caminho onde estão os arquivos `.xlsx`
3. Rodar `python juntar_excel.py` para gerar o CSV consolidado
4. (Opcional) Rodar `python converter_csv.py` se precisar converter um `.xlsx` já consolidado
