# Análise de Criminalidade

Repositório do trabalho de Análise de Criminalidade

**Matéria:** Projeto Integrador Computação 3

**Professor:** Howard Cruz Roatti

**Tema:** Análise de Criminalidade

**Grupo:**

- Gabriel Ferreira Paulo; 
- João Pereira Paes Henriques; 
- Murilo Henrique Costa Reis; 
- Pedro de Oliveira Guimarães; 
- Thaís Peroni Custódio Lino. 

--- 

## Onde os csv Utilizados Foram Adquiridos

Os csv utilizados neste trabalho foram adquiridos neste site do Governo Federal do Brasil: 

https://www.gov.br/mj/pt-br/assuntos/sua-seguranca/seguranca-publica/estatistica/dados-nacionais-1/base-de-dados-e-notas-metodologicas-dos-gestores-estaduais-sinesp-vde-2022-e-2023

--- 

## Organização do Repositório
### Codigos

Dentro da pasta **`Codigos`** está os scipts criados para juntar e converter os dados de criminalidade do SINESP VDE, juntamente com um README explicando como eles funcionam e como utilizá-los

### Jupyter_Notebook_C2

Dentro da pasta **`Jupyter_Notebook_C2`** está o código do Jupyter Notebook escrito em Python onde foram realizadas as análises da C2, o arquivo explicando cada análise realizada e o link da Apresentação do protótipo em vídeo da C2

### Streamlit_C3

Dentro da pasta **`Streamlit_C3`** estão os códigos em Python utilizando o framework Streamlit. Nesta etapa (C3), as análises exploratórias desenvolvidas na C2 foram transformadas em um painel interativo (Dashboard), com as seguintes melhorias:
- **Filtros Dinâmicos:** Inclusão de filtros opcionais (como ano, estado ou tipo de crime) para permitir uma exploração customizada dos dados.
- **Interface Organizada:** Utilização de abas e sub-abas (`st.tabs`) para separar as diferentes análises de forma clara e intuitiva.

Os principais arquivos desta pasta são:
- `Trabalho_C3_Analise_Criminalidade_PIC3_2026_1_VersaoFinal.py`: O script consolidado e oficial que executa a aplicação do dashboard.
- `Trabalho_C3_Analise_Criminalidade_PIC3_2026_1_JoaoPereira.py` e `ThaisPeroni.py`: Versões de desenvolvimento contendo a estrutura geral e as adaptações iniciais dos notebooks.

#### Como Executar a Aplicação (Streamlit)

Para rodar o dashboard interativo localmente na sua máquina, siga os passos abaixo:

1. **Instale as dependências necessárias:**
   Certifique-se de ter o Python instalado e execute no terminal:
   ```bash
   pip install streamlit pandas matplotlib plotly scikit-learn xgboost requests
   ```

2. **Navegue até a pasta do Streamlit**
3. **Execute o aplicativo:**
   ```bash
   streamlit run Trabalho_C3_Analise_Criminalidade_PIC3_2026_1_VersaoFinal.py
   ```
---

## Demonstração do Projeto
- Apresentação do protótipo em vídeo (C2): https://youtu.be/HB2Lu4qGYYw?si=wmgEZKQ_UPBb2aME
- Apresentação do protótipo em vídeo (C3): https://youtu.be/qa8oPMF2R3w?si=OlUmt2X4sVxFt-p9
