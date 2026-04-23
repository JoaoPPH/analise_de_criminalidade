import pandas as pd
import os

# Diretorio dos arquivos
diretorio = r"C:\Users\Aluno\Documents\dados"

# Arquivo de entrada e saida
arquivo_entrada = os.path.join(diretorio, "BancoVDE_Consolidado.xlsx")
arquivo_saida = os.path.join(diretorio, "BancoVDE_Consolidado.csv")

print(f"Lendo: {arquivo_entrada}")
df = pd.read_excel(arquivo_entrada)

print(f"Total de linhas: {len(df)}")
print(f"Total de colunas: {len(df.columns)}")

# Salva como CSV
# sep=';' por ser o padrao brasileiro (Excel PT-BR)
# encoding='utf-8-sig' para abrir corretamente no Excel com acentos
print(f"\nSalvando CSV em: {arquivo_saida}")
df.to_csv(arquivo_saida, index=False, sep=";", encoding="utf-8-sig")

print("Concluido!")
