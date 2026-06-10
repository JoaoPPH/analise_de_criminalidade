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
    st.header("4. Analise de Risco de Morte por Tipo de Agente")

    st.markdown("#### Filtros da Analise de Morte de Agentes")
    col_uf_p, col_ano_p, col_evento_p = st.columns(3)

    eventos_agente = ['Morte de Agente do Estado', 'Suicidio de Agente do Estado']
    df_agentes_base = df[(df['Ano'] < 2026) & (df['evento'].isin(eventos_agente))].copy()

    with col_uf_p:
        lista_uf_p = sorted(df_agentes_base['uf'].dropna().unique())
        ufs_selecionadas_p = st.multiselect("Filtrar por Estado (UF):", options=lista_uf_p, key="multiselect_uf_p")

    with col_ano_p:
        lista_anos_p = sorted(df_agentes_base['Ano'].unique())
        anos_selecionados_p = st.multiselect("Filtrar por Ano:", options=lista_anos_p, key="multiselect_ano_p")

    with col_evento_p:
        evento_selecionado_p = st.multiselect("Filtrar por Tipo de Evento:", options=eventos_agente, key="multiselect_evento_p")

    df_filtrado_p = df_agentes_base.copy()
    if ufs_selecionadas_p:
        df_filtrado_p = df_filtrado_p[df_filtrado_p['uf'].isin(ufs_selecionadas_p)]
    if anos_selecionados_p:
        df_filtrado_p = df_filtrado_p[df_filtrado_p['Ano'].isin(anos_selecionados_p)]
    if evento_selecionado_p:
        df_filtrado_p = df_filtrado_p[df_filtrado_p['evento'].isin(evento_selecionado_p)]

    subabas_p = st.tabs([
        "Visao Geral de Mortes de Agentes",
        "Evolucao Temporal",
        "Analise Geografica por Estado",
        "Analise de Regressao e Tendencias",
        "Comparacao de Modelos: Regressao Linear vs Random Forest vs XGBoost"
    ])

    with subabas_p[0]:
        st.subheader("Visao Geral de Mortes e Suicidios de Agentes do Estado")

        total_vitimas_agentes = df_filtrado_p['total_vitima'].sum()
        total_registros_agentes = len(df_filtrado_p)
        tipos_agentes = df_filtrado_p['agente'].dropna().nunique()
        estados_afetados = df_filtrado_p['uf'].dropna().nunique()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total de Vitimas", f"{int(total_vitimas_agentes):,}")
        m2.metric("Total de Registros", f"{total_registros_agentes:,}")
        m3.metric("Tipos de Agentes", f"{tipos_agentes}")
        m4.metric("Estados com Registros", f"{estados_afetados}")

        st.divider()

        st.markdown("#### Total de Mortes por Tipo de Agente no Brasil")
        mortes_por_agente = (
            df_filtrado_p
            .groupby('agente')['total_vitima']
            .sum()
            .reset_index()
            .sort_values(by='total_vitima', ascending=False)
        )
        total_geral_agentes = mortes_por_agente['total_vitima'].sum()
        mortes_por_agente['Porcentagem (%)'] = np.where(
            total_geral_agentes > 0,
            (mortes_por_agente['total_vitima'] / total_geral_agentes * 100).round(2),
            0
        )
        mortes_por_agente = mortes_por_agente.rename(columns={'total_vitima': 'Total de Vitimas', 'agente': 'Tipo de Agente'})

        fig_agente_bar = px.bar(
            data_frame=mortes_por_agente,
            x='Total de Vitimas',
            y='Tipo de Agente',
            orientation='h',
            title='Total de Mortes por Tipo de Agente',
            color='Total de Vitimas',
            color_continuous_scale='Reds'
        )
        fig_agente_bar.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_agente_bar, use_container_width=True)

        st.dataframe(
            mortes_por_agente.style.format({
                'Total de Vitimas': '{:,.0f}',
                'Porcentagem (%)': '{:.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.markdown("#### Top 10 Estados com Mais Mortes de Agentes")
        mortes_total_estado = (
            df_filtrado_p
            .groupby('uf')['total_vitima']
            .sum()
            .reset_index()
            .sort_values(by='total_vitima', ascending=False)
        )

        fig_top_estados = px.bar(
            data_frame=mortes_total_estado.head(10),
            x='uf',
            y='total_vitima',
            title='Top 10 Estados com Mais Mortes de Agentes do Estado',
            labels={'uf': 'Estado', 'total_vitima': 'Total de Vitimas'},
            color='total_vitima',
            color_continuous_scale='Reds'
        )
        fig_top_estados.update_layout(xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig_top_estados, use_container_width=True)

        st.divider()

        st.markdown("#### Tipo de Agente com Maior Numero de Mortes por Estado")
        mortes_por_estado_agente = (
            df_filtrado_p
            .groupby(['uf', 'agente'])['total_vitima']
            .sum()
            .reset_index()
        )
        agente_mais_mortes_estado = (
            mortes_por_estado_agente
            .sort_values(['uf', 'total_vitima'], ascending=[True, False])
            .groupby('uf')
            .first()
            .reset_index()
        )
        agente_mais_mortes_estado = agente_mais_mortes_estado.rename(columns={
            'uf': 'Estado',
            'agente': 'Agente com Mais Mortes',
            'total_vitima': 'Total de Vitimas'
        })

        fig_agente_estado = px.bar(
            data_frame=agente_mais_mortes_estado,
            x='Estado',
            y='Total de Vitimas',
            color='Agente com Mais Mortes',
            title='Tipo de Agente com Maior Numero de Mortes por Estado',
            labels={'Total de Vitimas': 'Total de Vitimas', 'Estado': 'Estado'}
        )
        fig_agente_estado.update_layout(xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig_agente_estado, use_container_width=True)

        st.dataframe(
            agente_mais_mortes_estado.style.format({
                'Total de Vitimas': '{:,.0f}'
            }),
            use_container_width=True,
            hide_index=True
        )

    with subabas_p[1]:
        st.subheader("Evolucao Temporal de Mortes de Agentes do Estado")

        mortes_brasil_ano = df_filtrado_p.groupby('Ano')['total_vitima'].sum().reset_index()

        fig_evolucao_total = px.line(
            data_frame=mortes_brasil_ano,
            x='Ano',
            y='total_vitima',
            title='Evolucao do Total de Mortes de Agentes do Estado no Brasil',
            labels={'Ano': 'Ano', 'total_vitima': 'Total de Vitimas'},
            markers=True
        )
        fig_evolucao_total.update_traces(line=dict(color='darkred', width=3), marker=dict(size=8))
        fig_evolucao_total.update_layout(xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig_evolucao_total, use_container_width=True)

        st.divider()

        st.markdown("#### Evolucao de Mortes por Tipo de Agente ao Longo dos Anos")
        mortes_agente_ano = (
            df_filtrado_p
            .groupby(['Ano', 'agente'])['total_vitima']
            .sum()
            .reset_index()
        )

        fig_agente_ano = px.line(
            data_frame=mortes_agente_ano,
            x='Ano',
            y='total_vitima',
            color='agente',
            title='Evolucao de Mortes por Tipo de Agente ao Longo dos Anos',
            labels={'Ano': 'Ano', 'total_vitima': 'Total de Vitimas', 'agente': 'Tipo de Agente'},
            markers=True
        )
        fig_agente_ano.update_layout(xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig_agente_ano, use_container_width=True)

        tabela_agente_ano = mortes_agente_ano.pivot(index='Ano', columns='agente', values='total_vitima').fillna(0)
        tabela_agente_ano['Total Geral'] = tabela_agente_ano.sum(axis=1)

        format_dict_agente = {col: '{:,.0f}' for col in tabela_agente_ano.columns}
        st.markdown("#### Detalhamento de Mortes por Tipo de Agente e Ano")
        st.dataframe(
            tabela_agente_ano.style.format(format_dict_agente),
            use_container_width=True
        )

        st.divider()

        st.markdown("#### Morte vs Suicidio de Agentes do Estado por Ano")
        mortes_evento_ano = (
            df_filtrado_p
            .groupby(['Ano', 'evento'])['total_vitima']
            .sum()
            .reset_index()
        )

        mapa_cores_evento = {
            'Morte de Agente do Estado': '#b22222',
            'Suicidio de Agente do Estado': '#4682b4'
        }

        fig_evento_ano = px.line(
            data_frame=mortes_evento_ano,
            x='Ano',
            y='total_vitima',
            color='evento',
            title='Morte vs Suicidio de Agentes do Estado por Ano',
            labels={'Ano': 'Ano', 'total_vitima': 'Total de Vitimas', 'evento': 'Tipo de Evento'},
            markers=True,
            color_discrete_map=mapa_cores_evento
        )
        fig_evento_ano.update_layout(xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig_evento_ano, use_container_width=True)

        tabela_evento_ano = mortes_evento_ano.pivot(index='Ano', columns='evento', values='total_vitima').fillna(0)
        tabela_evento_ano['Total'] = tabela_evento_ano.sum(axis=1)

        format_dict_evento = {col: '{:,.0f}' for col in tabela_evento_ano.columns}
        st.dataframe(
            tabela_evento_ano.style.format(format_dict_evento),
            use_container_width=True
        )

    with subabas_p[2]:
        st.subheader("Distribuicao Geografica de Mortes de Agentes por Estado")

        @st.cache_data
        def carregar_geojson_agentes():
            url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
            return requests.get(url).json()

        with st.spinner('Carregando mapa interativo...'):
            geojson_agentes = carregar_geojson_agentes()

        df_mapa_agentes = df_filtrado_p.groupby('uf').agg({
            'total_vitima': 'sum'
        }).reset_index()
        df_mapa_agentes['uf'] = df_mapa_agentes['uf'].astype(str).str.upper().str.strip()

        df_morte_uf = df_filtrado_p[df_filtrado_p['evento'] == 'Morte de Agente do Estado'].groupby('uf')['total_vitima'].sum().reset_index()
        df_morte_uf.columns = ['uf', 'mortes']
        df_suicidio_uf = df_filtrado_p[df_filtrado_p['evento'] == 'Suicidio de Agente do Estado'].groupby('uf')['total_vitima'].sum().reset_index()
        df_suicidio_uf.columns = ['uf', 'suicidios']

        df_mapa_agentes = df_mapa_agentes.merge(df_morte_uf, on='uf', how='left').merge(df_suicidio_uf, on='uf', how='left')
        df_mapa_agentes['mortes'] = df_mapa_agentes['mortes'].fillna(0)
        df_mapa_agentes['suicidios'] = df_mapa_agentes['suicidios'].fillna(0)

        df_mapa_agentes['perc_mortes'] = np.where(
            df_mapa_agentes['total_vitima'] > 0,
            (df_mapa_agentes['mortes'] / df_mapa_agentes['total_vitima'] * 100).round(2),
            0
        )
        df_mapa_agentes['perc_suicidios'] = np.where(
            df_mapa_agentes['total_vitima'] > 0,
            (df_mapa_agentes['suicidios'] / df_mapa_agentes['total_vitima'] * 100).round(2),
            0
        )

        fig_mapa_agentes = px.choropleth(
            data_frame=df_mapa_agentes,
            geojson=geojson_agentes,
            locations='uf',
            featureidkey="properties.sigla",
            color='total_vitima',
            hover_name='uf',
            hover_data={
                'uf': False,
                'total_vitima': ':,',
                'mortes': ':,',
                'suicidios': ':,',
                'perc_mortes': ':.2f',
                'perc_suicidios': ':.2f'
            },
            title='<b>Distribuicao de Mortes de Agentes do Estado por UF (Mapa Interativo)</b>',
            color_continuous_scale="Reds",
            labels={
                'total_vitima': 'Total de Vitimas',
                'mortes': 'Mortes em Servico',
                'suicidios': 'Suicidios',
                'perc_mortes': 'Mortes (%)',
                'perc_suicidios': 'Suicidios (%)'
            }
        )
        fig_mapa_agentes.update_geos(fitbounds="locations", visible=False)
        fig_mapa_agentes.update_layout(
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            height=600,
            template="plotly_white"
        )
        st.plotly_chart(fig_mapa_agentes, use_container_width=True)

        st.divider()

        st.markdown("#### Dados Consolidados por Unidade da Federacao:")
        st.dataframe(
            df_mapa_agentes.sort_values(by='total_vitima', ascending=False).style.format({
                'total_vitima': '{:,.0f}',
                'mortes': '{:,.0f}',
                'suicidios': '{:,.0f}',
                'perc_mortes': '{:.2f}%',
                'perc_suicidios': '{:.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )

    with subabas_p[3]:
        st.subheader("Analise de Regressao e Tendencias de Mortes de Agentes")
        st.write("Esta secao executa modelos de Regressao Linear para identificar tendencias nacionais, estaduais e por tipo de agente, projetando os proximos anos.")

        df_nacional_ag = df_agentes_base.groupby('Ano')['total_vitima'].sum().reset_index()

        X_reg_ag = df_nacional_ag[['Ano']]
        y_reg_ag = df_nacional_ag['total_vitima']
        modelo_ag = LinearRegression()
        modelo_ag.fit(X_reg_ag, y_reg_ag)

        y_pred_hist_ag = modelo_ag.predict(X_reg_ag)

        anos_futuros_ag = pd.DataFrame({'Ano': [2026, 2027, 2028, 2029, 2030]})
        previsoes_ag = modelo_ag.predict(anos_futuros_ag)

        r2_ag = r2_score(y_reg_ag, y_pred_hist_ag)
        mae_ag = mean_absolute_error(y_reg_ag, y_pred_hist_ag)

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Coeficiente (Inclinacao)", f"{modelo_ag.coef_[0]:.2f}")
        col_r2.metric("R2 (Determinacao)", f"{r2_ag:.4f}")
        col_r3.metric("MAE (Erro Medio Absoluto)", f"{mae_ag:,.2f}")

        fig_regressao_ag = go.Figure()

        fig_regressao_ag.add_trace(go.Scatter(
            x=df_nacional_ag['Ano'],
            y=df_nacional_ag['total_vitima'],
            mode='lines+markers',
            name='Dados Reais',
            line=dict(color='darkred', width=2),
            marker=dict(size=8)
        ))

        fig_regressao_ag.add_trace(go.Scatter(
            x=df_nacional_ag['Ano'],
            y=y_pred_hist_ag,
            mode='lines',
            name='Reta Ajustada',
            line=dict(color='red', width=1.5, dash='dash')
        ))

        fig_regressao_ag.add_trace(go.Scatter(
            x=list(anos_futuros_ag['Ano']),
            y=list(previsoes_ag),
            mode='lines+markers',
            name='Previsao (2026-2030)',
            line=dict(color='green', width=2.5, dash='dot'),
            marker=dict(size=10, symbol='star')
        ))

        fig_regressao_ag.update_layout(
            title='Regressao Linear - Mortes de Agentes do Estado no Brasil',
            xaxis=dict(title='Ano', tickmode='linear'),
            yaxis=dict(title='Total de Vitimas'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified',
            margin=dict(l=40, r=40, t=80, b=40)
        )
        st.plotly_chart(fig_regressao_ag, use_container_width=True)

        st.divider()
        st.markdown("### Metricas Projetadas (2026 - 2030)")

        c1, c2, c3, c4, c5 = st.columns(5)
        anos_lista_ag = [2026, 2027, 2028, 2029, 2030]
        colunas_lista_ag = [c1, c2, c3, c4, c5]

        for idx, col in enumerate(colunas_lista_ag):
            with col:
                st.markdown(f"#### **Ano {anos_lista_ag[idx]}**")
                st.metric("Total Projetado", f"{int(previsoes_ag[idx]):,}")

        st.divider()

        st.markdown("### Tendencia de Mortes por Estado (Coeficiente da Regressao)")
        resultados_estado_ag = []

        for estado in df_agentes_base['uf'].unique():
            dados_est = (
                df_agentes_base[df_agentes_base['uf'] == estado]
                .groupby('Ano')['total_vitima']
                .sum()
                .reset_index()
            )

            if len(dados_est) > 1:
                X_est = dados_est[['Ano']]
                y_est = dados_est['total_vitima']
                modelo_est = LinearRegression()
                modelo_est.fit(X_est, y_est)
                coef_est = modelo_est.coef_[0]
                prev_2027 = modelo_est.predict(pd.DataFrame({'Ano': [2027]}))[0]

                if coef_est > 0:
                    tendencia = 'Aumento'
                elif coef_est < 0:
                    tendencia = 'Diminuicao'
                else:
                    tendencia = 'Estavel'

                resultados_estado_ag.append([estado, coef_est, prev_2027, tendencia])

        df_tendencia_estado_ag = pd.DataFrame(
            resultados_estado_ag,
            columns=['Estado', 'Coeficiente', 'Previsao 2027', 'Tendencia']
        )
        df_tend_sorted_ag = df_tendencia_estado_ag.sort_values('Coeficiente', ascending=False)

        df_tend_sorted_ag['Cor'] = df_tend_sorted_ag['Coeficiente'].apply(
            lambda c: 'Aumento' if c > 0 else ('Diminuicao' if c < 0 else 'Estavel')
        )

        mapa_cores_tend = {
            'Aumento': '#b22222',
            'Diminuicao': '#4682b4',
            'Estavel': '#808080'
        }

        fig_tendencia_estado = px.bar(
            data_frame=df_tend_sorted_ag,
            x='Estado',
            y='Coeficiente',
            color='Cor',
            color_discrete_map=mapa_cores_tend,
            title='Tendencia de Mortes de Agentes por Estado (Coeficiente da Regressao Linear)',
            labels={'Coeficiente': 'Coeficiente (positivo = aumento)', 'Estado': 'Estado', 'Cor': 'Tendencia'}
        )
        fig_tendencia_estado.update_layout(xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig_tendencia_estado, use_container_width=True)

        st.dataframe(
            df_tend_sorted_ag[['Estado', 'Coeficiente', 'Previsao 2027', 'Tendencia']].style.format({
                'Coeficiente': '{:.4f}',
                'Previsao 2027': '{:,.0f}'
            }),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.markdown("### Resumo: Estados com Aumento vs Diminuicao")
        contagem_tend = df_tendencia_estado_ag['Tendencia'].value_counts().reset_index()
        contagem_tend.columns = ['Tendencia', 'Quantidade de Estados']

        mapa_cores_resumo = {
            'Aumento': '#b22222',
            'Diminuicao': '#4682b4',
            'Estavel': '#808080'
        }

        fig_resumo_tend = px.bar(
            data_frame=contagem_tend,
            x='Tendencia',
            y='Quantidade de Estados',
            color='Tendencia',
            color_discrete_map=mapa_cores_resumo,
            title='Quantidade de Estados por Tendencia de Mortes de Agentes'
        )
        st.plotly_chart(fig_resumo_tend, use_container_width=True)

        st.divider()

        st.markdown("### Tendencia por Estado e Tipo de Agente")
        resultados_estado_agente_ag = []

        for estado in df_agentes_base['uf'].unique():
            for agente in df_agentes_base['agente'].dropna().unique():
                dados_ea = (
                    df_agentes_base[
                        (df_agentes_base['uf'] == estado) &
                        (df_agentes_base['agente'] == agente)
                    ]
                    .groupby('Ano')['total_vitima']
                    .sum()
                    .reset_index()
                )

                if len(dados_ea) > 1:
                    X_ea = dados_ea[['Ano']]
                    y_ea = dados_ea['total_vitima']
                    modelo_ea = LinearRegression()
                    modelo_ea.fit(X_ea, y_ea)
                    coef_ea = modelo_ea.coef_[0]
                    prev_2027_ea = modelo_ea.predict(pd.DataFrame({'Ano': [2027]}))[0]

                    if coef_ea > 0:
                        tend_ea = 'Aumento'
                    elif coef_ea < 0:
                        tend_ea = 'Diminuicao'
                    else:
                        tend_ea = 'Estavel'

                    resultados_estado_agente_ag.append([estado, agente, coef_ea, prev_2027_ea, tend_ea])

        df_tend_estado_agente = pd.DataFrame(
            resultados_estado_agente_ag,
            columns=['Estado', 'Agente', 'Coeficiente', 'Previsao 2027', 'Tendencia']
        )

        aumento_agente_estado = (
            df_tend_estado_agente[df_tend_estado_agente['Coeficiente'] > 0]
            .sort_values(by='Coeficiente', ascending=False)
            .groupby('Estado')
            .first()
            .reset_index()
        )

        diminuicao_agente_estado = (
            df_tend_estado_agente[df_tend_estado_agente['Coeficiente'] < 0]
            .sort_values(by='Coeficiente', ascending=True)
            .groupby('Estado')
            .first()
            .reset_index()
        )

        if len(aumento_agente_estado) > 0:
            st.markdown("#### Tipo de Agente com Maior Tendencia de Aumento por Estado")
            aumento_agente_estado['Descricao'] = aumento_agente_estado['Estado'] + ' - ' + aumento_agente_estado['Agente']

            fig_aumento = px.bar(
                data_frame=aumento_agente_estado.sort_values('Coeficiente', ascending=False),
                x='Descricao',
                y='Coeficiente',
                title='Tipo de Agente com Maior Tendencia de Aumento de Mortes por Estado',
                labels={'Descricao': 'Estado - Agente', 'Coeficiente': 'Coeficiente'},
                color_discrete_sequence=['#b22222']
            )
            fig_aumento.update_layout(xaxis=dict(tickangle=90))
            st.plotly_chart(fig_aumento, use_container_width=True)

        if len(diminuicao_agente_estado) > 0:
            st.markdown("#### Tipo de Agente com Maior Tendencia de Diminuicao por Estado")
            diminuicao_agente_estado['Descricao'] = diminuicao_agente_estado['Estado'] + ' - ' + diminuicao_agente_estado['Agente']

            fig_diminuicao = px.bar(
                data_frame=diminuicao_agente_estado.sort_values('Coeficiente', ascending=True),
                x='Descricao',
                y='Coeficiente',
                title='Tipo de Agente com Maior Tendencia de Diminuicao de Mortes por Estado',
                labels={'Descricao': 'Estado - Agente', 'Coeficiente': 'Coeficiente'},
                color_discrete_sequence=['#4682b4']
            )
            fig_diminuicao.update_layout(xaxis=dict(tickangle=90))
            st.plotly_chart(fig_diminuicao, use_container_width=True)

        st.divider()

        st.markdown("### Tabela Resumo: Maior Aumento e Diminuicao por Estado")
        resumo_agentes = aumento_agente_estado[['Estado', 'Agente', 'Coeficiente']].merge(
            diminuicao_agente_estado[['Estado', 'Agente', 'Coeficiente']],
            on='Estado',
            how='outer',
            suffixes=('_Aumento', '_Diminuicao')
        )
        st.dataframe(
            resumo_agentes.style.format({
                'Coeficiente_Aumento': '{:.4f}',
                'Coeficiente_Diminuicao': '{:.4f}'
            }),
            use_container_width=True,
            hide_index=True
        )


    with subabas_p[4]:
        st.subheader("Comparacao de Modelos: Regressao Linear vs Random Forest vs XGBoost")

        df_ml_ag = df_agentes_base.groupby('Ano')['total_vitima'].sum().reset_index()
        X_ml_ag = df_ml_ag[['Ano']]
        y_ml_ag = df_ml_ag['total_vitima']

        lr_ag = LinearRegression()
        rf_ag = RandomForestRegressor(n_estimators=100, random_state=42)
        xgb_ag = XGBRegressor(n_estimators=100, random_state=42)

        lr_ag.fit(X_ml_ag, y_ml_ag)
        rf_ag.fit(X_ml_ag, y_ml_ag)
        xgb_ag.fit(X_ml_ag, y_ml_ag)

        y_pred_lr_ag = lr_ag.predict(X_ml_ag)
        y_pred_rf_ag = rf_ag.predict(X_ml_ag)
        y_pred_xgb_ag = xgb_ag.predict(X_ml_ag)

        def calcular_metricas(y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            return mae, rmse, r2

        m_lr_ag = calcular_metricas(y_ml_ag, y_pred_lr_ag)
        m_rf_ag = calcular_metricas(y_ml_ag, y_pred_rf_ag)
        m_xgb_ag = calcular_metricas(y_ml_ag, y_pred_xgb_ag)

        df_metricas_ag = pd.DataFrame({
            'Metrica': ['MAE (Erro Medio Absoluto)', 'RMSE (Raiz do Erro Quadratico Medio)', 'R2 (Coef. de Determinacao)'],
            'Regressao Linear': [m_lr_ag[0], m_lr_ag[1], m_lr_ag[2]],
            'Random Forest': [m_rf_ag[0], m_rf_ag[1], m_rf_ag[2]],
            'XGBoost': [m_xgb_ag[0], m_xgb_ag[1], m_xgb_ag[2]]
        })

        st.markdown("#### Metricas de Performance Computadas na Base Historica:")
        st.dataframe(df_metricas_ag.style.format({
            'Regressao Linear': '{:,.6f}',
            'Random Forest': '{:,.6f}',
            'XGBoost': '{:,.6f}'
        }),
            hide_index=True
        )

        st.divider()

        df_ajuste_ag = pd.DataFrame({
            'Ano': df_ml_ag['Ano'],
            'Real': y_ml_ag,
            'Regr. Linear': y_pred_lr_ag,
            'Random Forest': y_pred_rf_ag,
            'XGBoost': y_pred_xgb_ag
        })
        fig_comp_ag = px.line(
            data_frame=df_ajuste_ag,
            x='Ano',
            y=['Real', 'Regr. Linear', 'Random Forest', 'XGBoost'],
            title='Comparacao do Ajuste dos Modelos a Curva Real Historica',
            markers=True
        )
        fig_comp_ag.update_layout(xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig_comp_ag, use_container_width=True)

        st.divider()

        st.markdown("### Projecoes Futuras por Modelo (2026-2030)")
        anos_futuros_ml = pd.DataFrame({'Ano': [2026, 2027, 2028, 2029, 2030]})
        prev_lr_fut = lr_ag.predict(anos_futuros_ml)
        prev_rf_fut = rf_ag.predict(anos_futuros_ml)
        prev_xgb_fut = xgb_ag.predict(anos_futuros_ml)

        df_proj_modelos = pd.DataFrame({
            'Ano': anos_futuros_ml['Ano'],
            'Regressao Linear': prev_lr_fut,
            'Random Forest': prev_rf_fut,
            'XGBoost': prev_xgb_fut
        })

        fig_proj_modelos = px.line(
            data_frame=df_proj_modelos,
            x='Ano',
            y=['Regressao Linear', 'Random Forest', 'XGBoost'],
            title='Projecoes Futuras de Mortes de Agentes por Modelo (2026-2030)',
            markers=True
        )
        fig_proj_modelos.update_layout(xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig_proj_modelos, use_container_width=True)

        st.dataframe(
            df_proj_modelos.style.format({
                'Regressao Linear': '{:,.0f}',
                'Random Forest': '{:,.0f}',
                'XGBoost': '{:,.0f}'
            }),
            hide_index=True
        )

# Parte do Gabriel   
with abas[4]:
    st.header("5. Análise da Quantidade de Drogas apreendidas (total_peso)")

    # Preparação da base de drogas
    df_drogas_base = df[df['evento'].str.contains('coca|macon', case=False, na=False)].copy()
    df_drogas_base['uf'] = df_drogas_base['uf'].astype(str).str.upper().str.strip()
    
    # -------------------------------------------------------------
    # FILTROS DA ABA DE DROGAS
    # -------------------------------------------------------------
    st.markdown("#### Filtros da Análise de Drogas")
    col_drug, col_yr, col_state = st.columns(3)
    
    with col_drug:
        opcoes_droga = ["Ambos", "Cocaína", "Maconha"]
        droga_selecionada = st.radio("Tipo de Droga / Evento:", opcoes_droga, index=0, key="radio_drug_d")
        
    with col_yr:
        anos_disponiveis = sorted(df_drogas_base['Ano'].unique())
        anos_selecionados_d = st.multiselect("Selecionar Anos:", anos_disponiveis, default=anos_disponiveis, key="ms_anos_d")
        
    with col_state:
        estados_disponiveis = sorted(df_drogas_base['uf'].dropna().unique())
        estados_selecionados_d = st.multiselect("Selecionar Estados (UF):", estados_disponiveis, key="ms_estados_d")

    # Verificação e Aplicação de Filtros
    if not anos_selecionados_d:
        st.warning("⚠️ Selecione pelo menos um ano na caixa de filtros.")
    else:
        df_filtrado_d = df_drogas_base[df_drogas_base['Ano'].isin(anos_selecionados_d)].copy()
        
        if droga_selecionada != "Ambos":
            termo_busca = "coca" if droga_selecionada == "Cocaína" else "macon"
            df_filtrado_d = df_filtrado_d[df_filtrado_d['evento'].str.contains(termo_busca, case=False, na=False)]
            
        if estados_selecionados_d:
            df_filtrado_d = df_filtrado_d[df_filtrado_d['uf'].isin(estados_selecionados_d)]
            
        if df_filtrado_d.empty:
            st.warning("Nenhum registro encontrado para os filtros selecionados.")
        else:
            # -------------------------------------------------------------
            # SUBABAS - VISÃO GERAL AGORA É UMA SUB-ABA SEPARADA
            # -------------------------------------------------------------
            subabas_d = st.tabs([
                "Visão Geral",
                "Evolução Temporal", 
                "Análise Geográfica", 
                "Previsões e Tendências"
            ])
            
            # Sub-aba 1: Visão Geral
            with subabas_d[0]:
                st.subheader("Métricas Gerais de Apreensões")
                
                total_kg = df_filtrado_d['total_peso'].sum()
                total_ton = total_kg / 1000
                
                anos_ativos = [y for y in df_filtrado_d['Ano'].unique() if y != 2026]
                qtd_anos = len(anos_ativos) if len(anos_ativos) > 0 else df_filtrado_d['Ano'].nunique()
                avg_kg = df_filtrado_d[df_filtrado_d['Ano'] != 2026]['total_peso'].sum() / qtd_anos if qtd_anos > 0 else 0
                avg_ton = avg_kg / 1000
                
                totais_anuais = df_filtrado_d.groupby('Ano')['total_peso'].sum()
                ano_recorde = totais_anuais.idxmax() if not totais_anuais.empty else "N/A"
                max_ano_val = totais_anuais.max() / 1000 if not totais_anuais.empty else 0
                
                c_k1, c_k2, c_k3, c_k4 = st.columns(4)
                c_k1.metric("Peso Total Apreendido", f"{total_ton:,.2f} t", f"{total_kg:,.1f} kg")
                c_k2.metric("Média Anual (exceto 2026)", f"{avg_ton:,.2f} t/ano")
                c_k3.metric("Ano Recorde", f"{ano_recorde}", f"{max_ano_val:,.2f} t")
                c_k4.metric("Número de Ocorrências", f"{len(df_filtrado_d):,}")
            
            # Sub-aba 2: Evolução Temporal (Similar à lógica estruturada da parte 2)
            with subabas_d[1]:
                st.subheader("Evolução Temporal de Apreensões por Ano")
                
                drogas_tipo_ano = df_filtrado_d.groupby(['Ano', 'evento'])['total_peso'].sum().reset_index()
                drogas_tipo_ano['total_peso_t'] = drogas_tipo_ano['total_peso'] / 1000
                drogas_tipo_ano['Droga'] = drogas_tipo_ano['evento'].apply(lambda x: "Cocaína" if "coca" in str(x).lower() else "Maconha")
                
                fig_bar_d = px.bar(
                    drogas_tipo_ano, x='Ano', y='total_peso_t', color='Droga', barmode='group',
                    title='Apreensões de Drogas por Ano e Substância (Toneladas)',
                    template="plotly_white"
                )
                fig_bar_d.update_layout(xaxis=dict(tickmode='linear'))
                st.plotly_chart(fig_bar_d, use_container_width=True)
                
                st.markdown("#### Detalhes Consolidados por Ano (Toneladas)")
                pivot_drogas = drogas_tipo_ano.pivot_table(index='Ano', columns='Droga', values='total_peso_t', aggfunc='sum', fill_value=0)
                pivot_drogas['Total'] = pivot_drogas.sum(axis=1)
                st.dataframe(pivot_drogas.style.format("{:,.2f} t"), use_container_width=True)

            # Sub-aba 3: Análise Geográfica
            with subabas_d[2]:
                st.subheader("Distribuição Geográfica das Apreensões")
                
                @st.cache_data
                def carregar_geojson_drogas():
                    url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
                    return requests.get(url).json()
                    
                with st.spinner('Carregando mapa interativo...'):
                    geojson_d = carregar_geojson_drogas()
                
                df_mapa_d = df_filtrado_d.groupby('uf')['total_peso'].sum().reset_index()
                df_mapa_d['total_peso_t'] = df_mapa_d['total_peso'] / 1000
                
                fig_mapa_d = px.choropleth(
                    df_mapa_d, geojson=geojson_d, locations='uf', featureidkey="properties.sigla",
                    color='total_peso_t', hover_name='uf',
                    title='Distribuição de Apreensões de Drogas (Toneladas)',
                    color_continuous_scale="Reds",
                    template="plotly_white"
                )
                fig_mapa_d.update_geos(fitbounds="locations", visible=False)
                fig_mapa_d.update_layout(margin={"r":0,"t":50,"l":0,"b":0}, height=600)
                st.plotly_chart(fig_mapa_d, use_container_width=True)

            # Sub-aba 4: Previsões e Tendências
            with subabas_d[3]:
                st.subheader("Modelagem Preditiva e Ajuste de Curvas")
                incluir_2026 = st.checkbox("Incluir 2026 no treinamento (dados parciais)", value=False, key="chk_2026_drogas")
                
                drogas_reg = df_filtrado_d.groupby('Ano')['total_peso'].sum().reset_index()
                drogas_reg['total_peso_t'] = drogas_reg['total_peso'] / 1000
                
                treino_df = drogas_reg if incluir_2026 else drogas_reg[drogas_reg['Ano'] < 2026]
                    
                if len(treino_df) < 2:
                    st.warning("⚠️ Dados insuficientes para realizar a regressão linear.")
                else:
                    X_d = treino_df[['Ano']]
                    y_d = treino_df['total_peso_t']
                    
                    modelo_d = LinearRegression()
                    modelo_d.fit(X_d, y_d)
                    
                    anos_futuros_d = pd.DataFrame({'Ano': [2026, 2027, 2028, 2029, 2030]})
                    preds_futuras_d = modelo_d.predict(anos_futuros_d)
                    
                    fig_proj_d = go.Figure()
                    fig_proj_d.add_trace(go.Scatter(x=treino_df['Ano'], y=y_d, mode='lines+markers', name='Histórico Real'))
                    fig_proj_d.add_trace(go.Scatter(x=anos_futuros_d['Ano'], y=preds_futuras_d, mode='lines+markers', name='Previsão (2026-2030)', line=dict(dash='dot')))
                    
                    fig_proj_d.update_layout(title='Projeções de Apreensões de Drogas', template="plotly_white")
                    st.plotly_chart(fig_proj_d, use_container_width=True)
                    
                    st.markdown("#### Tabela de Valores Projetados")
                    df_proj_d = pd.DataFrame({'Ano': anos_futuros_d['Ano'], 'Previsão (Toneladas)': preds_futuras_d})
                    st.dataframe(df_proj_d.style.format({'Previsão (Toneladas)': '{:,.2f} t'}), hide_index=True)
