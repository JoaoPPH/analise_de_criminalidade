import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import json
import requests
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Configurando a página do Streamlit
st.set_page_config(page_title="Análise de Criminalidade - PIC3 - C3", layout="wide")

# Título principal do Dashboard
st.title("Análise de Criminalidade - Projeto Integrador Computação 3 - C3")

# Função para carregar os dados (com CACHE para exitar travar)
@st.cache_data
def carregar_dados():
    df = pd.read_csv('BancoVDE_Consolidado.csv', sep=';', low_memory=False)
    return df

with st.spinner('Carregando os Dados do CSV Consolidado'):
    df = carregar_dados()

# Informações Gerais Sobre o CSV Consolidado
st.markdown("### Visão Geral")
col1, col2 = st.columns(2)
col1.metric("Total de Registros", f"{len(df):,}")
col2.metric("Total de Colunas", f"{df.shape[1]}")

st.divider()

# -------------------------------------------------------------
# CRIAÇÃO DAS ABAS DE CADA ANÁLISE
# -------------------------------------------------------------
abas = st.tabs([
    "1. Análises Gerais de Crimes", 
    "2. Vitimas por Gênero", 
    "3. Armas apreendidas", 
    "4. Morte de Agente", 
    "5. Drogas Apreendidas"
])

# Parte da Thais: 1. Análise do número de crimes em cada Estado/Município
with abas[0]:
    st.header("1. Análise do número de crimes em cada Estado/Município")

# Parte do João: 2. Análise do número de vítimas por gênero
with abas[1]:
    st.header("2. Análise do Número de Vítimas por Gênero")
    
    # -------------------------------------------------------------
    # FILTROS ESPECÍFICOS PARA A ABA DE GÊNERO
    # -------------------------------------------------------------
    st.markdown("#### Filtros da Análise de Gênero")
    col_uf_g, col_ano_g = st.columns(2)
    
    with col_uf_g:
        lista_uf_g = sorted(df['uf'].dropna().unique())
        ufs_selecionadas_g = st.multiselect("Filtrar por Estado (UF):", options=lista_uf_g, key="multiselect_uf_g")
        
    with col_ano_g:
        # Filtrando anos menores que 2026 para manter consistência histórica do Jupyter
        lista_anos_g = sorted(df[df['Ano'] < 2026]['Ano'].unique())
        anos_selecionados_g = st.multiselect("Filtrar por Ano:", options=lista_anos_g, key="multiselect_ano_g")

    # Aplicando os filtros ao DataFrame dessa aba
    df_filtrado_g = df[df['Ano'] < 2026].copy()
    if ufs_selecionadas_g:
        df_filtrado_g = df_filtrado_g[df_filtrado_g['uf'].isin(ufs_selecionadas_g)]
    if anos_selecionados_g:
        df_filtrado_g = df_filtrado_g[df_filtrado_g['Ano'].isin(anos_selecionados_g)]

    # -------------------------------------------------------------
    # CRIAÇÃO DAS SUBABAS DE GÊNERO
    # -------------------------------------------------------------
    subabas_g = st.tabs([
        "Distribuição Total de Vítimas por Gênero", 
        "Análise Histórica de Vítimas", 
        "Distribuição de Vítimas por Gênero e Estado", 
        "Análise de Regressão", 
        "Comparação de Modelos: Regressão Linear vs Random Forest vs XGBoost"
    ])

    # Sub-aba 1: Distribuição Total de Vítimas por Gênero
    with subabas_g[0]:
        st.subheader("Distribuição Total de Vítimas por Gênero")
        total_masc = df_filtrado_g['masculino'].sum()
        total_fem = df_filtrado_g['feminino'].sum()
        total_ni = df_filtrado_g['nao_informado'].sum()
        
        # Exibição de métricas rápidas
        m1, m2, m3 = st.columns(3)
        m1.metric("Vítimas Masculinas", f"{int(total_masc):,}")
        m2.metric("Vítimas Femininas", f"{int(total_fem):,}")
        m3.metric("Não Informado", f"{int(total_ni):,}")
        
        # Gráfico de Pizza dinâmico com cores explícitas e personalizadas
        dados_pizza = pd.DataFrame({
            'Gênero': ['Masculino', 'Feminino', 'Não Informado'],
            'Total': [total_masc, total_fem, total_ni]
        })
        
        # Mapeamento manual de cores desejadas
        mapa_cores = {
            'Masculino': '#1f77b4',     # Azul padrão
            'Feminino': '#e377c2',      # Rosa / Magenta
            'Não Informado': '#949494'  # Cinza neutro
        }
        
        fig_pizza = px.pie(
            data_frame=dados_pizza, 
            values='Total', 
            names='Gênero', 
            title='Proporção Total de Vítimas por Gênero',
            color='Gênero',               # Diz ao plotly para usar a coluna Gênero para colorir
            color_discrete_map=mapa_cores # Aplica o dicionário de cores fixas
        )
        st.plotly_chart(fig_pizza, use_container_width=True)
        
        st.divider()
        
        # Tabela
        st.markdown("#### Total de Vítimas por Tipo de Evento")
        
        # Agrupamento utilizando a base filtrada da aba
        resultados_vitimas = df_filtrado_g.groupby("evento")[["feminino", "masculino", "nao_informado"]].sum()
        
        # Adicionando uma coluna de Total Geral para melhor leitura no Dashboard
        resultados_vitimas['Total Geral'] = resultados_vitimas.sum(axis=1)
        
        # Ordenando pelos eventos com maior ocorrência total
        resultados_vitimas = resultados_vitimas.sort_values(by='Total Geral', ascending=False)
        
        # Exibindo como uma tabela elegante formatando os números com separador de milhar
        st.dataframe(
            resultados_vitimas.style.format({
                'feminino': '{:,.0f}',
                'masculino': '{:,.0f}',
                'nao_informado': '{:,.0f}',
                'Total Geral': '{:,.0f}'
            })
        )

    # Sub-aba 2: Análise Histórica de Vítimas
    with subabas_g[1]:
        st.subheader("Análise Histórica de Vítimas")
        
        # Agrupamento incluindo 'nao_informado'
        df_historico_g = df_filtrado_g.groupby('Ano')[['masculino', 'feminino', 'nao_informado']].sum().reset_index()
        
        # Criação da coluna de Total Geral por Ano
        df_historico_g['Total Geral'] = df_historico_g['masculino'] + df_historico_g['feminino'] + df_historico_g['nao_informado']
        
        # Cálculo das porcentagens de cada gênero em relação ao total daquele ano
        df_historico_g['% Masculino'] = np.where(df_historico_g['Total Geral'] > 0, (df_historico_g['masculino'] / df_historico_g['Total Geral']) * 100, 0)
        df_historico_g['% Feminino'] = np.where(df_historico_g['Total Geral'] > 0, (df_historico_g['feminino'] / df_historico_g['Total Geral']) * 100, 0)
        df_historico_g['% Não Informado'] = np.where(df_historico_g['Total Geral'] > 0, (df_historico_g['nao_informado'] / df_historico_g['Total Geral']) * 100, 0)

        # Ajustando a estrutura mudando explicitamente o nome do argumento var_name para 'Gênero'
        df_melt = df_historico_g.melt(
            id_vars=['Ano', 'Total Geral', '% Masculino', '% Feminino', '% Não Informado'],
            value_vars=['masculino', 'feminino', 'nao_informado', 'Total Geral'],
            var_name='Gênero',
            value_name='Quantidade'
        )

        # Mapeando os nomes internos para as categorias corretas na legenda
        nomes_legenda = {
            'masculino': 'Masculino',
            'feminino': 'Feminino',
            'nao_informado': 'Não Informado',
            'Total Geral': 'Total Geral'
        }
        df_melt['Gênero'] = df_melt['Gênero'].map(nomes_legenda)

        # Vinculando o rótulo de porcentagem correto de cada linha para exibição direta sobre os pontos
        condicoes = [
            df_melt['Gênero'] == 'Masculino',
            df_melt['Gênero'] == 'Feminino',
            df_melt['Gênero'] == 'Não Informado'
        ]
        escolhas = [
            df_melt['% Masculino'].map('{:.1f}%'.format),
            df_melt['% Feminino'].map('{:.1f}%'.format),
            df_melt['% Não Informado'].map('{:.1f}%'.format)
        ]
        # Para a linha preta do total faz sentido ocultar ou exibir 100% (ocultaremos com string vazia '' para não poluir o gráfico)
        df_melt['Porcentagem_Texto'] = np.select(condicoes, escolhas, default='')

        # Dicionário de cores customizadas unificado
        mapa_cores_historico = {
            'Masculino': '#1f77b4',       # Azul
            'Feminino': '#e377c2',        # Rosa
            'Não Informado': '#949494',   # Cinza Neutro
            'Total Geral': '#000000'      # Preto
        }

        # Construindo o gráfico de linhas interativo
        fig_linha_g = px.line(
            data_frame=df_melt, 
            x='Ano', 
            y='Quantidade',
            color='Gênero', # Altera para agrupar por Gênero, corrigindo o lado direito da legenda
            color_discrete_map=mapa_cores_historico,
            title='Evolução Temporal e Proporção de Vítimas por Gênero',
            markers=True,
            text='Porcentagem_Texto', # Ativa a inserção automática de textos fixos em cima dos marcadores
            custom_data=['Total Geral']
        )

        # Ajustes finos de posição e formatação do texto fixo nos pontos e correção do Bug do hovertext
        fig_linha_g.update_traces(
            textposition="top center", # Força a porcentagem a ficar centralizada logo acima do marcador
            hovertemplate="<br>".join([
                "<b>Ano:</b> %{x}",
                "<b>Quantidade:</b> %{y:,.0f}",
                "<b>Total do Ano Geral:</b> %{customdata[0]:,.0f}"
            ])
        )

        # Altera o título da legenda e ajusta o layout do gráfico
        fig_linha_g.update_layout(
            yaxis_title="Quantidade de Vítimas", 
            xaxis_title="Ano",
            legend_title_text="Gênero" # Força o título da legenda a ser "Gênero"
        )

        # Exibindo o gráfico na tela
        st.plotly_chart(fig_linha_g, use_container_width=True)
        
        st.divider()

        # Exibindo a tabela final detalhada
        st.markdown("#### Detalhamento dos Dados Históricos")
        st.dataframe(
            df_historico_g.style.format({
                'masculino': '{:,.0f}', 
                '% Masculino': '{:.2f}%',
                'feminino': '{:,.0f}',
                '% Feminino': '{:.2f}%',
                'nao_informado': '{:,.0f}',
                '% Não Informado': '{:.2f}%',
                'Total Geral': '{:,.0f}'
            }),
            hide_index=True
        )

    # Sub-aba 3: Distribuição de Vítimas por Gênero e Estado
    with subabas_g[2]:
        st.subheader("Distribuição de Vítimas por Gênero e Estado")
        
        # Função interna com cache para carregar a malha do mapa do Brasil sem dar lentidão
        @st.cache_data
        def carregar_geojson():
            url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
            return requests.get(url).json()
            
        with st.spinner('Carregando mapa interativo...'):
            geojson_br = carregar_geojson()
        
        # Agrupamento e tratamento de strings com base nos filtros da aba
        df_mapa = df_filtrado_g.groupby('uf').agg({
            'feminino': 'sum',
            'masculino': 'sum'
        }).reset_index()
        
        df_mapa['uf'] = df_mapa['uf'].astype(str).str.upper().str.strip()
        df_mapa['total_vitimas'] = df_mapa['feminino'] + df_mapa['masculino']
        
        # Tratamento matemático para evitar divisões por zero se o estado não tiver registros filtrados
        df_mapa['perc_feminino'] = np.where(df_mapa['total_vitimas'] > 0, (df_mapa['feminino'] / df_mapa['total_vitimas'] * 100).round(2), 0)
        df_mapa['perc_masculino'] = np.where(df_mapa['total_vitimas'] > 0, (df_mapa['masculino'] / df_mapa['total_vitimas'] * 100).round(2), 0)
        
        # Construindo o Mapa Coroplético interativo do Plotly Express
        fig_mapa_br = px.choropleth(
            data_frame=df_mapa,
            geojson=geojson_br,
            locations='uf',
            featureidkey="properties.sigla",
            color='total_vitimas',
            hover_name='uf',
            hover_data={
                'uf': False, # Esconde a sigla repetida no hover
                'total_vitimas': ':,',
                'feminino': ':,',
                'masculino': ':,',
                'perc_feminino': ':.2f',
                'perc_masculino': ':.2f'
            },
            title='<b>Distribuição de Vítimas por Gênero e Estado (Mapa Interativo)</b>',
            color_continuous_scale="Reds",
            labels={
                'total_vitimas': 'Total de Vítimas',
                'feminino': 'Vítimas Fem.',
                'masculino': 'Vítimas Masc.',
                'perc_feminino': 'Feminino (%)',
                'perc_masculino': 'Masculino (%)'
            }
        )
        
        # Ajustando o enquadramento do mapa automático para focar no Brasil
        fig_mapa_br.update_geos(fitbounds="locations", visible=False)
        
        # Customizações de layout e margens
        fig_mapa_br.update_layout(
            margin={"r":0,"t":50,"l":0,"b":0},
            height=600,
            template="plotly_white"
        )
        
        # Renderizando o mapa de forma nativa e responsiva no Streamlit
        st.plotly_chart(fig_mapa_br, use_container_width=True)

        st.divider()
        
        # Exibe a tabela de suporte logo abaixo para conferência de valores absolutos
        st.markdown("#### Dados Consolidados por Unidade da Federação:")
        st.dataframe(
            df_mapa.sort_values(by='total_vitimas', ascending=False).style.format({
                'feminino': '{:,.0f}',
                'masculino': '{:,.0f}',
                'total_vitimas': '{:,.0f}',
                'perc_feminino': '{:.2f}%',
                'perc_masculino': '{:.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )

    # Sub-aba 4: Análise de Regressão
    with subabas_g[3]:
        st.subheader("Análise Histórica e Previsão de Vítimas")
        st.write("Esta seção executa um modelo de Regressão Linear baseado na tendência histórica nacional para projetar os próximos anos.")
    
        # Preparação dos dados históricos (Nacional)
        df_nacional = df[df['Ano'] < 2026].groupby('Ano')[['feminino', 'masculino']].sum().reset_index()
        df_nacional['total_vitimas'] = df_nacional['feminino'] + df_nacional['masculino']
    
        # Cálculos das porcentagens históricas
        df_nacional['pct_fem'] = (df_nacional['feminino'] / df_nacional['total_vitimas']) * 100
        df_nacional['pct_masc'] = (df_nacional['masculino'] / df_nacional['total_vitimas']) * 100
    
        # Modelagem de Regressão (Total)
        X_reg = df_nacional[['Ano']]
        y_reg = df_nacional['total_vitimas']
        modelo_lr = LinearRegression()
        modelo_lr.fit(X_reg, y_reg)
    
        # Modelagem de Regressão por Gênero para as Retas Ajustadas Históricas
        modelo_fem = LinearRegression().fit(X_reg, df_nacional['feminino'])
        modelo_masc = LinearRegression().fit(X_reg, df_nacional['masculino'])
    
        # Projeções de reta ajustada para o período histórico
        reta_total_hist = modelo_lr.predict(X_reg)
        reta_fem_hist = modelo_fem.predict(X_reg)
        reta_masc_hist = modelo_masc.predict(X_reg)
    
        # Predições futuras (2026 - 2030)
        anos_futuros = pd.DataFrame({'Ano': [2026, 2027, 2028, 2029, 2030]})
        previsoes_futuras_total = modelo_lr.predict(anos_futuros)
        previsoes_futuras_masc = modelo_masc.predict(anos_futuros)
        previsoes_futuras_fem = modelo_fem.predict(anos_futuros)
    
        # --- CONSTRUÇÃO DO GRÁFICO (Estilo Jupyter/Plotly Avançado) ---
        fig_previsao = go.Figure()
    
        # Histórico Masculino (Pontos + Linha) com Porcentagem
        fig_previsao.add_trace(go.Scatter(
            x=df_nacional['Ano'], 
            y=df_nacional['masculino'],
            mode='lines+markers+text',
            name='Histórico Masculino',
            text=df_nacional['pct_masc'].map('{:.1f}%'.format),
            textposition='top center',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8)
        ))
    
        # Reta Ajustada Masculino (Tracejada)
        fig_previsao.add_trace(go.Scatter(
            x=df_nacional['Ano'], 
            y=reta_masc_hist,
            mode='lines',
            name='Reta Ajustada Masc.',
            line=dict(color='#1f77b4', width=1.5, dash='dash')
        ))
    
        # Histórico Feminino (Pontos + Linha) com Porcentagem
        fig_previsao.add_trace(go.Scatter(
            x=df_nacional['Ano'], 
            y=df_nacional['feminino'],
            mode='lines+markers+text',
            name='Histórico Feminino',
            text=df_nacional['pct_fem'].map('{:.1f}%'.format),
            textposition='bottom center',
            line=dict(color='#e377c2', width=2),
            marker=dict(size=8)
        ))
    
        # Reta Ajustada Feminino (Tracejada)
        fig_previsao.add_trace(go.Scatter(
            x=df_nacional['Ano'], 
            y=reta_fem_hist,
            mode='lines',
            name='Reta Ajustada Fem.',
            line=dict(color='#e377c2', width=1.5, dash='dash')
        ))
    
        # Histórico Total
        fig_previsao.add_trace(go.Scatter(
            x=df_nacional['Ano'], 
            y=df_nacional['total_vitimas'],
            mode='lines+markers',
            name='Histórico Total',
            line=dict(color='#000000', width=3),
            marker=dict(size=8)
        ))
    
        # Reta Ajustada Total (Tracejada)
        fig_previsao.add_trace(go.Scatter(
            x=df_nacional['Ano'], 
            y=reta_total_hist,
            mode='lines',
            name='Reta Ajustada Total',
            line=dict(color='#000000', width=1.5, dash='dash')
        ))
    
        # Passado diretamente a lista de anos futuros (2026 a 2030)
        anos_previsao = list(anos_futuros['Ano'])
        
        # Passado apenas os arrays gerados pelos modelos do Sklearn (sem emendar o dado de 2025)
        valores_previsao_total = list(previsoes_futuras_total)
        valores_previsao_masculino = list(previsoes_futuras_masc)
        valores_previsao_feminino = list(previsoes_futuras_fem)
        
        # Ajusta os rótulos de porcentagem para terem exatamente os 5 elementos de 2026 a 2030
        pct_masc_futuro = [f"{(m/t)*100:.1f}%" for m, t in zip(previsoes_futuras_masc, previsoes_futuras_total)]
        pct_fem_futuro = [f"{(f/t)*100:.1f}%" for f, t in zip(previsoes_futuras_fem, previsoes_futuras_total)]
    
        # Plotagem das retas de previsão (Mantenha o mode='lines+markers' para criar as linhas pontilhadas)
        fig_previsao.add_trace(go.Scatter(
            x=anos_previsao, 
            y=valores_previsao_total,
            mode='lines+markers',
            name='Previsão Total (2026-2030)',
            line=dict(color="#585858", width=3, dash='dot'),
            marker=dict(size=8, symbol='star')
        ))
    
        fig_previsao.add_trace(go.Scatter(
            x=anos_previsao, 
            y=valores_previsao_masculino,
            mode='lines+markers+text',
            name='Previsão Masculino (2026-2030)',
            text=pct_masc_futuro,
            textposition='top center',
            line=dict(color="#004c83", width=2.5, dash='dot'),
            marker=dict(size=8, symbol='star')
        ))
    
        fig_previsao.add_trace(go.Scatter(
            x=anos_previsao, 
            y=valores_previsao_feminino,
            mode='lines+markers+text',
            name='Previsão Feminino (2026-2030)',
            text=pct_fem_futuro,
            textposition='bottom center',
            line=dict(color="#8a0061", width=2.5, dash='dot'),
            marker=dict(size=8, symbol='star')
        ))
    
        # Layout e formatação técnica do gráfico
        fig_previsao.update_layout(
            title='Análise Histórica e Tendência por Gênero (com Projeção de Modelos)',
            xaxis=dict(title='Ano', tickmode='linear'),
            yaxis=dict(title='Quantidade de Vítimas'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified',
            margin=dict(l=40, r=40, t=80, b=40)
        )
    
        st.plotly_chart(fig_previsao, use_container_width=True)
        
        st.divider()
        st.markdown("### Métricas Projetadas (2026 - 2030)")
        
        # Apresentação limpa das métricas em colunas por ano
        c1, c2, c3, c4, c5 = st.columns(5)
        anos_lista = [2026, 2027, 2028, 2029, 2030]
        colunas_lista = [c1, c2, c3, c4, c5]
        
        for idx, col in enumerate(colunas_lista):
            with col:
                st.markdown(f"#### **Ano {anos_lista[idx]}**")
                st.metric("Total Projetado", f"{int(previsoes_futuras_total[idx]):,}")
                st.metric("Masculino Projetado", f"{int(previsoes_futuras_masc[idx]):,}")
                st.metric("Feminino Projetado", f"{int(previsoes_futuras_fem[idx]):,}")

    # Sub-aba 5: Comparação de Modelos de Machine Learning
    with subabas_g[4]:
        st.subheader("Comparação de Modelos: Regressão Linear vs Random Forest vs XGBoost")
        
        # Agrupamento fixo nacional para o treinamento dos algoritmos
        df_ml = df[df['Ano'] < 2026].groupby('Ano')[['feminino', 'masculino']].sum().reset_index()
        df_ml['total_vitimas'] = df_ml['feminino'] + df_ml['masculino']
        
        X_ml = df_ml[['Ano']]
        y_ml = df_ml['total_vitimas']
        
        # Inicializando e treinando os modelos
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        xgb = XGBRegressor(n_estimators=100, random_state=42)
        
        lr.fit(X_ml, y_ml)
        rf.fit(X_ml, y_ml)
        xgb.fit(X_ml, y_ml)
        
        # Predições na base de treino para cálculo de métricas de avaliação
        y_pred_lr = lr.predict(X_ml)
        y_pred_rf = rf.predict(X_ml)
        y_pred_xgb = xgb.predict(X_ml)
        
        # Função interna para extração de métricas
        def calcular_metricas(y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            return mae, rmse, r2

        m_lr = calcular_metricas(y_ml, y_pred_lr)
        m_rf = calcular_metricas(y_ml, y_pred_rf)
        m_xgb = calcular_metricas(y_ml, y_pred_xgb)
        
        # Montagem do DataFrame comparativo de avaliação
        df_metricas = pd.DataFrame({
            'Métrica': ['MAE (Erro Médio Absoluto)', 'RMSE (Raiz do Erro Quadrático Médio)', 'R² (Coef. de Determinação)'],
            'Regressão Linear': [m_lr[0], m_lr[1], m_lr[2]],
            'Random Forest': [m_rf[0], m_rf[1], m_rf[2]],
            'XGBoost': [m_xgb[0], m_xgb[1], m_xgb[2]]
        })
        
        st.markdown("#### Métricas de performance computadas na base histórica:")
        st.dataframe(df_metricas.style.format({
            'Regressão Linear': '{:,.6f}', 
            'Random Forest': '{:,.6f}', 
            'XGBoost': '{:,.6f}'
            }),
            hide_index=True
        )

        st.divider()
        
        # Gráfico comparativo visual do ajuste das curvas de treino
        df_ajuste = pd.DataFrame({
            'Ano': df_ml['Ano'],
            'Real': y_ml,
            'Regr. Linear': y_pred_lr,
            'Random Forest': y_pred_rf,
            'XGBoost': y_pred_xgb
        })
        fig_comp = px.line(data_frame=df_ajuste, 
                           x='Ano', 
                           y=['Real', 'Regr. Linear', 'Random Forest', 'XGBoost'],
                           title='Comparação do Ajuste dos Modelos à Curva Real Histórica',
                           markers=True)
        st.plotly_chart(fig_comp, use_container_width=True)


# Parte do João: 3. Análise da Quantidade de Armas apreendidas por Tipo
with abas[2]:
    st.header("3. Análise da Quantidade de Armas Apreendidas por Tipo")
    
    # -------------------------------------------------------------
    # FILTROS ESPECÍFICOS PARA A ABA DE ARMAS
    # -------------------------------------------------------------
    st.markdown("#### Filtros da Análise de Armas")
    col_uf_a, col_ano_a = st.columns(2)
    
    with col_uf_a:
        lista_uf_a = sorted(df['uf'].dropna().unique())
        ufs_selecionadas_a = st.multiselect("Filtrar por Estado (UF):", options=lista_uf_a, key="multiselect_uf_a")
        
    with col_ano_a:
        lista_anos_a = sorted(df[df['Ano'] < 2026]['Ano'].unique())
        anos_selecionados_a = st.multiselect("Filtrar por Ano:", options=lista_anos_a, key="multiselect_ano_a")

    # Isolando registros exclusivos de apreensão de arma de fogo e aplicando filtros informados
    df_armas_base = df[(df['Ano'] < 2026) & (df['evento'] == 'Arma de Fogo Apreendida')].copy()
    if ufs_selecionadas_a:
        df_armas_base = df_armas_base[df_armas_base['uf'].isin(ufs_selecionadas_a)]
    if anos_selecionados_a:
        df_armas_base = df_armas_base[df_armas_base['Ano'].isin(anos_selecionados_a)]

    # -------------------------------------------------------------
    # CRIAÇÃO DAS SUBABAS DE ARMAS
    # -------------------------------------------------------------
    subabas_a = st.tabs([
        "Visão Geral de Armas Apreendidas", 
        "Evolução Temporal de Apreensões", 
        "Análise Espacial por Estado"
    ])

    # Sub-aba 1: Visão Geral de Armas Apreendidas
    with subabas_a[0]:
        st.subheader("Distribuição Total de Armas de Fogo Apreendidas por Tipo")
        
        # Agrupamento e ordenação base
        df_tipo_arma = df_armas_base.groupby('arma')['total'].sum().reset_index().sort_values(by='total', ascending=False)
        
        # Cálculo da Porcentagem Representativa de cada arma
        total_geral_armas = df_tipo_arma['total'].sum()
        df_tipo_arma['Porcentagem (%)'] = (df_tipo_arma['total'] / total_geral_armas * 100) if total_geral_armas > 0 else 0
        
        # Renomeando a coluna 'total' para ficar mais amigável na tabela final
        df_tipo_arma = df_tipo_arma.rename(columns={'total': 'Quantidade Apreendida', 'arma': 'Tipo de Arma'})
        
        # Gráfico de barras horizontais para melhor leitura dos tipos de armas
        fig_tipo_arma = px.bar(data_frame=df_tipo_arma, 
                               x='Quantidade Apreendida', 
                               y='Tipo de Arma', 
                               orientation='h',
                               labels={'Quantidade Apreendida': 'Quantidade Apreendida', 'Tipo de Arma': 'Tipo de Arma'},
                               title='Total de Apreensões Agrupadas por Tipo de Arma',
                               color='Quantidade Apreendida', 
                               color_continuous_scale='Blues')
        
        # Inverter o eixo Y para que a arma mais apreendida fique no topo do gráfico
        fig_tipo_arma.update_yaxes(autorange="reversed")
        
        st.plotly_chart(fig_tipo_arma, use_container_width=True)
        
        st.divider()

        # Exibindo a tabela final formatada com quantidade e porcentagem
        st.markdown("#### Detalhamento Estatístico das Armas:")
        st.dataframe(
            df_tipo_arma.style.format({
                'Quantidade Apreendida': '{:,.0f}',
                'Porcentagem (%)': '{:.2f}%'
            }),
            use_container_width=True,
            hide_index=True # Oculta a coluna de índices numéricos para deixar a tabela mais limpa
        )

    # Sub-aba 2: Evolução Temporal de Apreensões
    with subabas_a[1]:
        st.subheader("Evolução Histórica das Apreensões de Armas")
        df_temporal_arma = df_armas_base.groupby(['Ano', 'arma'])['total'].sum().reset_index()
        
        # Trocado para px.bar com barmode='group' para igualar ao Jupyter
        fig_temp_arma = px.bar(data_frame=df_temporal_arma, 
                               x='Ano', 
                               y='total', 
                               color='arma', 
                               barmode='group', # Agrupa as barras lado a lado por ano
                               labels={'total': 'Quantidade', 'arma': 'Tipo de Arma', 'Ano': 'Ano'},
                               title='Evolução Temporal de Apreensões por Categoria',
                               color_discrete_sequence=px.colors.qualitative.Plotly)
        
        # Garante que todos os anos apareçam explicitamente no eixo X sem quebrar como ponto flutuante
        fig_temp_arma.update_layout(xaxis=dict(tickmode='linear'))
        
        st.plotly_chart(fig_temp_arma, use_container_width=True)

    # Sub-aba 3: Análise Espacial por Estado
    with subabas_a[2]:
        st.subheader("Distribuição Geográfica das Apreensões")
        
        # Função interna com cache para carregar o GeoJSON do Brasil sem gerar lentidão
        @st.cache_data
        def carregar_geojson_armas():
            url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
            return requests.get(url).json()
            
        with st.spinner('Carregando mapa interativo de armas...'):
            geojson_br = carregar_geojson_armas()
        
        # Criando tabela pivotada de estados e armas conforme lógica do notebook
        df_mapa_armas = df_armas_base.pivot_table(
            index='uf', 
            columns='arma', 
            values='total', 
            aggfunc='sum', 
            fill_value=0
        ).reset_index()
        
        # Tratamento de strings das siglas dos estados
        df_mapa_armas['uf'] = df_mapa_armas['uf'].astype(str).str.upper().str.strip()
        
        # Adicionando somatório total de armas por estado
        colunas_armas = list(df_mapa_armas.columns[1:])
        df_mapa_armas['Total de Armas'] = df_mapa_armas[colunas_armas].sum(axis=1)
        
        # Cálculo das porcentagens de cada arma por estado (conforme lógica do Jupyter)
        for arma in colunas_armas:
            df_mapa_armas[f'{arma} (%)'] = np.where(
                df_mapa_armas['Total de Armas'] > 0, 
                (df_mapa_armas[arma] / df_mapa_armas['Total de Armas'] * 100).round(2), 
                0
            )
            
        # Construção dinâmica do dicionário de Hover (Informações ao passar o mouse)
        hover_dict = {'uf': False, 'Total de Armas': ':,'}
        for arma in colunas_armas:
            hover_dict[arma] = ':,'
            hover_dict[f'{arma} (%)'] = ':.2f'
            
        # Construindo o Mapa Coroplético Interativo do Plotly Express
        fig_mapa_armas = px.choropleth(
            data_frame=df_mapa_armas,
            geojson=geojson_br,
            locations='uf',
            featureidkey="properties.sigla",
            color='Total de Armas',
            hover_name='uf',
            hover_data=hover_dict,
            title='<b>Análise Espacial: Apreensão de Armas de Fogo por Estado (Mapa Interativo)</b>',
            color_continuous_scale="Viridis",
            labels={'Total de Armas': 'Total de Armas'}
        )
        
        # Ajustando o enquadramento geográfico focado no Brasil
        fig_mapa_armas.update_geos(fitbounds="locations", visible=False)
        
        # Ajustes finos de layout
        fig_mapa_armas.update_layout(
            margin={"r":0,"t":50,"l":0,"b":0},
            height=600,
            template="plotly_white"
        )
        
        # Renderizando o mapa de forma responsiva no Streamlit
        st.plotly_chart(fig_mapa_armas, use_container_width=True)
        
        st.divider()
        
        # Exibição da tabela matricial detalhada ordenada pelo volume absoluto
        st.write("Detalhamento matricial por Estado (UF) e Tipo de Arma:")
        
        # Dicionário de formatação automática para a tabela do Streamlit (.style.format)
        format_dict = {}
        for col in df_mapa_armas.columns:
            if col != 'uf':
                if '(%)' in col:
                    format_dict[col] = '{:.2f}%'
                else:
                    format_dict[col] = '{:,.0f}'
                    
        st.dataframe(
            df_mapa_armas.sort_values(by='Total de Armas', ascending=False).style.format(format_dict),
            use_container_width=True,
            hide_index=True
        )

# Parte do Pedro 
with abas[3]:
    st.header("4. Análise de Risco de Morte por Tipo de Agente")

# Parte do Gabriel   
with abas[4]:
    st.header("5. Análise da Quantidade de Drogas apreendidas (total_peso)")

