import pandas as pd
import glob
import os
import re

# Diretorio onde estao os arquivos
diretorio = r"C:\Users\Aluno\Documents\dados"

# Busca todos os arquivos Excel que comecam com "BancoVDE"
arquivos = glob.glob(os.path.join(diretorio, "BancoVDE *.xlsx"))

# Lista para armazenar os DataFrames
lista_dfs = []

for arquivo in arquivos:
    nome_arquivo = os.path.basename(arquivo)

    # Extrai o ano do nome do arquivo (ex: "BancoVDE 2015.xlsx" -> 2015)
    match = re.search(r"(\d{4})", nome_arquivo)
    if not match:
        print(f"Nao foi possivel extrair o ano de: {nome_arquivo}")
        continue

    ano = int(match.group(1))

    print(f"Lendo: {nome_arquivo} (ano {ano})")

    # Le o arquivo Excel
    df = pd.read_excel(arquivo)

    # Adiciona a coluna com o ano da ocorrencia
    df["Ano"] = ano

    lista_dfs.append(df)

# Concatena todos os DataFrames em um so
df_final = pd.concat(lista_dfs, ignore_index=True)

# Salva o resultado em CSV (xlsx tem limite de 1.048.576 linhas)
arquivo_saida = os.path.join(diretorio, "BancoVDE_Consolidado.csv")
df_final.to_csv(arquivo_saida, index=False, sep=";", encoding="utf-8-sig")

print(f"\nConcluido! Total de {len(df_final)} linhas salvas em:")
print(arquivo_saida)
