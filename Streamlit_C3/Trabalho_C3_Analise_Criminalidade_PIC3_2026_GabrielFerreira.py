import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configure Streamlit Page
st.set_page_config(
    page_title="Painel de Apreensão de Drogas - PIC3",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and geojson with cache
@st.cache_data
def load_data():
    df = pd.read_csv("drogas_consolidado.csv", sep=";")
    # Ensure State code is clean, uppercase, and stripped
    df['uf'] = df['uf'].astype(str).str.upper().str.strip()
    return df

@st.cache_data
def load_geojson():
    with open("brazil-states.geojson", "r", encoding="utf-8") as f:
        return json.load(f)

# Initialize data
try:
    df_drogas = load_data()
    geojson_br = load_geojson()
except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}")
    st.stop()

# Custom CSS for Premium Design
def apply_theme():
    st.markdown("""
    <style>
        /* Modern Fonts & Styling */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap');
        
        .main-title {
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            background: linear-gradient(135deg, #059669, #0284C7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            margin-bottom: 0px;
        }
        
        .subtitle {
            font-family: 'Inter', sans-serif;
            font-size: 1.05rem;
            color: #4B5563;
            margin-bottom: 25px;
            font-weight: 400;
        }
        
        /* Metric cards custom styling */
        .kpi-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
            border-top: 4px solid #059669;
            transition: all 0.3s ease;
            margin-bottom: 15px;
        }
        
        .kpi-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        }
        
        .kpi-label {
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
            font-weight: 600;
            color: #6B7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .kpi-value {
            font-family: 'Outfit', sans-serif;
            font-size: 1.9rem;
            font-weight: 700;
            color: #111827;
            margin-top: 5px;
        }
        
        .kpi-subtext {
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
            color: #059669;
            font-weight: 500;
            margin-top: 5px;
        }
        
        /* Custom alert box */
        .custom-info {
            background-color: #ECFDF5;
            border-left: 4px solid #10B981;
            padding: 15px;
            border-radius: 8px;
            color: #065F46;
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            margin-bottom: 20px;
        }
        
        /* Adjust standard Streamlit elements */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-family: 'Outfit', sans-serif;
            font-size: 1rem;
            font-weight: 600;
            padding: 8px 16px;
            background-color: #F3F4F6;
            border-radius: 8px 8px 0px 0px;
            color: #374151;
            border: 1px solid #E5E7EB;
            border-bottom: none;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #059669 !important;
            color: white !important;
            border-color: #059669 !important;
        }
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# Header Section
st.markdown("<h1 class='main-title'> Análise da Quantidade de Drogas Apreendidas</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Visualização interativa baseada no Trabalho C2 (Análise de Criminalidade) • PIC3</p>", unsafe_allow_html=True)

# ================= SIDEBAR FILTERS =================
st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 10px;'>
    <h2 style='color: #0F172A; font-family: Outfit, sans-serif; font-weight: 600; font-size: 1.4rem; margin-bottom: 0;'>Filtros de Análise</h2>
    <hr style='margin-top: 5px; margin-bottom: 15px; border-color: #10B981;'>
</div>
""", unsafe_allow_html=True)

# 1. Filter by Drug Type (Evento)
drug_options = {
    "Ambos": "Ambos",
    "Cocaína": "Apreensão de Cocaína",
    "Maconha": "Apreensão de Maconha"
}
selected_drug_label = st.sidebar.radio(
    "Tipo de Droga / Evento",
    list(drug_options.keys()),
    index=0,
    help="Filtre os dados por Cocáina, Maconha ou mantenha ambas."
)

# Apply drug filter
if selected_drug_label != "Ambos":
    # Using case-insensitive match for security
    event_search = "coca" if selected_drug_label == "Cocaína" else "macon"
    df_filtered = df_drogas[df_drogas['evento'].str.contains(event_search, case=False, na=False)].copy()
else:
    df_filtered = df_drogas.copy()

# 2. Filter by Year
years_available = sorted(df_filtered['Ano'].unique())
selected_years = st.sidebar.multiselect(
    "Selecionar Anos",
    years_available,
    default=years_available,
    help="Escolha um ou mais anos para visualizar."
)

if not selected_years:
    st.sidebar.warning("⚠️ Selecione pelo menos um ano.")
    selected_years = years_available

df_filtered = df_filtered[df_filtered['Ano'].isin(selected_years)]

# 3. Filter by State (UF)
states_available = sorted(df_filtered['uf'].dropna().unique())
selected_states = st.sidebar.multiselect(
    "Selecionar Estados (UF)",
    states_available,
    help="Deixe em branco para exibir todos os estados."
)

if selected_states:
    df_filtered = df_filtered[df_filtered['uf'].isin(selected_states)]

# Additional Info in Sidebar
st.sidebar.markdown("""
<div style='margin-top: 30px; padding: 15px; background-color: #F8FAFC; border-radius: 8px; border: 1px solid #E2E8F0;'>
    <h4 style='margin-top:0; color:#334155; font-size:0.9rem; font-weight:600;'>Sobre o Banco de Dados</h4>
    <p style='font-size:0.8rem; color:#64748B; margin-bottom:5px; line-height:1.4;'>
        Os dados de apreensão cobrem o período de 2015 a 2026.
    </p>
    <p style='font-size:0.8rem; color:#64748B; margin-bottom:0; line-height:1.4;'>
        <b>Nota:</b> O ano de 2026 possui registros parciais de apreensões coletadas.
    </p>
</div>
""", unsafe_allow_html=True)


# ================= MAIN METRIC CARDS =================
def render_kpi_cards(df):
    total_kg = df['total_peso'].sum()
    total_ton = total_kg / 1000
    
    # Calculate average yearly weight (excluding 2026 from averages as it is partial)
    active_years = [y for y in df['Ano'].unique() if y != 2026]
    years_count = len(active_years) if len(active_years) > 0 else df['Ano'].nunique()
    
    if years_count > 0:
        avg_kg = df[df['Ano'] != 2026]['total_peso'].sum() / years_count if 2026 in df['Ano'].unique() else total_kg / years_count
        avg_ton = avg_kg / 1000
    else:
        avg_kg = 0
        avg_ton = 0
        
    # Max Year and value
    yearly_totals = df.groupby('Ano')['total_peso'].sum()
    if not yearly_totals.empty:
        max_year = yearly_totals.idxmax()
        max_year_val = yearly_totals.max() / 1000
    else:
        max_year = "N/A"
        max_year_val = 0
        
    # Record counts
    records = len(df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Peso Total Apreendido</div>
            <div class="kpi-value">{total_ton:,.2f} t</div>
            <div class="kpi-subtext">{total_kg:,.1f} kg acumulados</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Média Anual (2015-2025)</div>
            <div class="kpi-value">{avg_ton:,.2f} t/ano</div>
            <div class="kpi-subtext">{avg_kg:,.1f} kg médios por ano</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Ano Recorde</div>
            <div class="kpi-value">{max_year}</div>
            <div class="kpi-subtext">{max_year_val:,.2f} t apreendidas</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Número de Registros</div>
            <div class="kpi-value">{records:,}</div>
            <div class="kpi-subtext">Ocorrências filtradas</div>
        </div>
        """, unsafe_allow_html=True)

# Render KPIs if dataframe is not empty
if not df_filtered.empty:
    render_kpi_cards(df_filtered)
else:
    st.warning("Nenhum dado encontrado com os filtros selecionados.")
    st.stop()


# ================= TABS NAVIGATION =================
tab_time, tab_geo, tab_pred = st.tabs([
    "📈 Evolução Temporal", 
    "🗺️ Análise Geográfica", 
    "🔮 Previsões e Tendências"
])

# ----------------- TAB 1: EVOLUÇÃO TEMPORAL -----------------
with tab_time:
    st.markdown("### Análise da Evolução Temporal")
    
    # Sub-tabs selector styled nicely
    sub_tab_1 = st.radio(
        "Selecione a visualização detalhada:",
        ["Histórico Geral", "Detalhamento por Droga e Tabela Anual"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if sub_tab_1 == "Histórico Geral":
        drogas_ano = df_filtered.groupby('Ano')['total_peso'].sum().reset_index()
        drogas_ano['total_peso_t'] = drogas_ano['total_peso'] / 1000
        
        # Line chart matching notebook style but interactive
        fig_line = px.line(
            drogas_ano,
            x='Ano',
            y='total_peso_t',
            markers=True,
            title='Evolução Total do Peso de Drogas Apreendidas (Brasil)',
            labels={'total_peso_t': 'Peso Total (Toneladas)', 'Ano': 'Ano'},
            color_discrete_sequence=['#059669']
        )
        fig_line.update_layout(
            template='plotly_white',
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#E2E8F0', tickmode='linear'),
            yaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Info Box about 2026
        if 2026 in selected_years:
            st.markdown("""
            <div class="custom-info">
                ℹ️ <b>Nota sobre 2026:</b> As apreensões deste ano apresentam valores inferiores devido a dados parciais coletados (até abril), assim como observado nas conclusões do projeto de criminalidade.
            </div>
            """, unsafe_allow_html=True)
            
    else:
        # Grouped Bar chart
        drogas_tipo_ano = df_filtered.groupby(['Ano', 'evento'])['total_peso'].sum().reset_index()
        drogas_tipo_ano['total_peso_t'] = drogas_tipo_ano['total_peso'] / 1000
        drogas_tipo_ano['Droga'] = drogas_tipo_ano['evento'].apply(
            lambda x: "Cocaína" if "coca" in str(x).lower() else "Maconha"
        )
        
        fig_bar = px.bar(
            drogas_tipo_ano,
            x='Ano',
            y='total_peso_t',
            color='Droga',
            title='Apreensões de Drogas por Ano e Substância',
            labels={'total_peso_t': 'Peso Total (Toneladas)', 'Ano': 'Ano', 'Droga': 'Substância'},
            color_discrete_map={'Cocaína': '#0284C7', 'Maconha': '#10B981'},
            barmode='group'
        )
        fig_bar.update_layout(
            template='plotly_white',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#E2E8F0', tickmode='linear'),
            yaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Data table for detailed inspection
        st.markdown("#### Tabela de Pesos Consolidados por Ano (Toneladas)")
        pivot_table = drogas_tipo_ano.pivot_table(
            index='Ano',
            columns='Droga',
            values='total_peso_t',
            aggfunc='sum',
            fill_value=0
        )
        pivot_table['Total'] = pivot_table.sum(axis=1)
        st.dataframe(pivot_table.style.format("{:,.2f} t"), use_container_width=True)


# ----------------- TAB 2: ANÁLISE GEOGRÁFICA -----------------
with tab_geo:
    st.markdown("### Análise da Distribuição Geográfica")
    
    sub_tab_2 = st.radio(
        "Selecione a visualização geográfica:",
        ["Mapa Coroplético por Estado", "Ranking de Estados"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if sub_tab_2 == "Mapa Coroplético por Estado":
        # Aggregate data by State
        df_map = df_filtered.groupby('uf')['total_peso'].sum().reset_index()
        df_map['total_peso_t'] = df_map['total_peso'] / 1000
        df_map['uf'] = df_map['uf'].astype(str).str.upper().str.strip()
        
        # Ensure we have all states in the map (even if weight is 0) to avoid blank states
        all_states = pd.DataFrame({'uf': list(geojson_br['features'][i]['properties']['sigla'] for i in range(len(geojson_br['features'])))})
        df_map = pd.merge(all_states, df_map, on='uf', how='left').fillna(0)
        
        fig_map = px.choropleth(
            df_map,
            geojson=geojson_br,
            locations='uf',
            featureidkey="properties.sigla",
            color='total_peso_t',
            hover_name='uf',
            title='<b>Distribuição Geográfica de Apreensões de Drogas (Toneladas)</b>',
            color_continuous_scale="Reds",
            labels={'total_peso_t': 'Total (Toneladas)'}
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(
            margin={"r":0,"t":50,"l":0,"b":0},
            height=600,
            template='plotly_white'
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
    else:
        # Layout: Top 10 States Ranking chart on the left, and data table on the right
        col_l, col_r = st.columns(2)
        
        with col_l:
            df_state = df_filtered.groupby('uf')['total_peso'].sum().reset_index()
            df_state['total_peso_t'] = df_state['total_peso'] / 1000
            df_state_top = df_state.sort_values(by='total_peso_t', ascending=False).head(10)
            
            fig_state = px.bar(
                df_state_top,
                x='total_peso_t',
                y='uf',
                orientation='h',
                title='Top 10 Estados (UF) em Apreensões (Toneladas)',
                labels={'total_peso_t': 'Total (t)', 'uf': 'Estado'},
                color='total_peso_t',
                color_continuous_scale='Greens'
            )
            fig_state.update_layout(
                yaxis=dict(autorange="reversed"),
                template='plotly_white',
                showlegend=False,
                coloraxis_showscale=False,
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#E2E8F0')
            )
            st.plotly_chart(fig_state, use_container_width=True)
            
        with col_r:
            st.markdown("#### Detalhamento das Apreensões (Top 10 Estados)")
            df_state_table = df_state_top.rename(columns={'uf': 'Estado (UF)', 'total_peso_t': 'Total (Toneladas)'})
            df_state_table['Total (Kilogramas)'] = df_state_table['total_peso']
            st.dataframe(
                df_state_table[['Estado (UF)', 'Total (Toneladas)', 'Total (Kilogramas)']].style.format({
                    'Total (Toneladas)': '{:,.2f} t',
                    'Total (Kilogramas)': '{:,.1f} kg'
                }),
                use_container_width=True,
                hide_index=True
            )


# ----------------- TAB 3: PREVISÕES E TENDÊNCIAS -----------------
with tab_pred:
    st.markdown("### Modelagem Preditiva & Análise de Tendências")
    
    sub_tab_3 = st.radio(
        "Selecione a modelagem:",
        ["Ajuste do Modelo (Histórico vs Regressão)", "Projeções Futuras (2026 - 2030)"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Checkbox to include 2026 in model training
    incluir_2026 = st.checkbox(
        "Incluir o ano de 2026 no treinamento do modelo (dados parciais)", 
        value=False,
        help="Por padrão, o ano de 2026 é excluído do treinamento por conter dados incompletos, evitando puxar a linha de tendência para baixo de forma artificial."
    )
    
    # Perform grouping
    drogas_ano_reg = df_filtered.groupby('Ano')['total_peso'].sum().reset_index()
    drogas_ano_reg['total_peso_t'] = drogas_ano_reg['total_peso'] / 1000
    
    # Filter out 2026 if requested
    if not incluir_2026:
        train_df = drogas_ano_reg[drogas_ano_reg['Ano'] < 2026].copy()
    else:
        train_df = drogas_ano_reg.copy()
        
    if len(train_df) < 2:
        st.warning("⚠️ Dados insuficientes para treinar o modelo de regressão linear. Ajuste os filtros na barra lateral para selecionar pelo menos 2 anos.")
    else:
        # Fit Linear Regression Model
        X = train_df['Ano'].values.reshape(-1, 1)
        y = train_df['total_peso_t'].values
        
        model = LinearRegression()
        model.fit(X, y)
        previsao = model.predict(X)
        
        # Calculate regression quality metrics
        r2 = r2_score(y, previsao)
        mae = mean_absolute_error(y, previsao)
        rmse = np.sqrt(mean_squared_error(y, previsao))
        
        if sub_tab_3 == "Ajuste do Modelo (Histórico vs Regressão)":
            st.markdown("#### Linha de Tendência de Apreensão de Drogas")
            
            # Interactive plot combining actual scatter and trend line
            fig_reg = go.Figure()
            # Actual values
            fig_reg.add_trace(go.Scatter(
                x=train_df['Ano'],
                y=y,
                mode='markers+lines',
                name='Dados Reais Históricos',
                marker=dict(size=10, color='#0284C7'),
                line=dict(color='#0284C7', width=1.5, dash='dot')
            ))
            # Predicted line
            fig_reg.add_trace(go.Scatter(
                x=train_df['Ano'],
                y=previsao,
                mode='lines',
                name='Linha de Tendência (Regressão Linear)',
                line=dict(color='#EF4444', width=3)
            ))
            
            # Include 2026 marker if it was excluded but we want to show where it lies relative to trend
            if not incluir_2026 and 2026 in drogas_ano_reg['Ano'].values:
                val_2026 = drogas_ano_reg[drogas_ano_reg['Ano'] == 2026]['total_peso_t'].values[0]
                fig_reg.add_trace(go.Scatter(
                    x=[2026],
                    y=[val_2026],
                    mode='markers',
                    name='Ano 2026 (Excluído do Modelo)',
                    marker=dict(size=12, color='#F59E0B', symbol='triangle-up')
                ))
            
            fig_reg.update_layout(
                title='Ajuste da Regressão Linear sobre o Histórico',
                xaxis_title='Ano',
                yaxis_title='Peso Total (Toneladas)',
                template='plotly_white',
                xaxis=dict(tickmode='linear', showgrid=True, gridcolor='#E2E8F0'),
                yaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_reg, use_container_width=True)
            
            # Show Metrics
            st.markdown("#### Métricas de Desempenho do Modelo")
            m1, m2, m3 = st.columns(3)
            m1.metric(
                label="Coeficiente de Determinação (R²)",
                value=f"{r2:.4f}",
                help="Varia de 0 a 1. Indica a proporção da variabilidade nos dados reais que é explicada pelo modelo. Próximo de 1 indica um excelente ajuste."
            )
            m2.metric(
                label="Erro Médio Absoluto (MAE)",
                value=f"{mae:.2f} t",
                help="A média aritmética das diferenças absolutas entre os valores preditos e os valores reais, em toneladas."
            )
            m3.metric(
                label="Raiz do Erro Quadrático Médio (RMSE)",
                value=f"{rmse:.2f} t",
                help="Mede a magnitude média dos erros do modelo, penalizando desvios maiores. Também medido em toneladas."
            )
            
        else:
            # Projections for 2026-2030
            st.markdown("#### Projeção Futura de Apreensões")
            
            future_years = np.array([2026, 2027, 2028, 2029, 2030])
            future_preds = model.predict(future_years.reshape(-1, 1))
            
            fig_proj = go.Figure()
            
            # History
            fig_proj.add_trace(go.Scatter(
                x=train_df['Ano'],
                y=y,
                mode='lines+markers',
                name='Histórico Real',
                line=dict(color='#059669', width=3),
                marker=dict(size=8)
            ))
            
            # Projections
            fig_proj.add_trace(go.Scatter(
                x=future_years,
                y=future_preds,
                mode='lines+markers',
                name='Previsão do Modelo (2026-2030)',
                line=dict(color='#EF4444', width=3, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))
            
            fig_proj.update_layout(
                title='Projeções de Apreensões de Drogas para o Próximo Período',
                xaxis_title='Ano',
                yaxis_title='Peso Total (Toneladas)',
                template='plotly_white',
                xaxis=dict(tickmode='linear', showgrid=True, gridcolor='#E2E8F0'),
                yaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_proj, use_container_width=True)
            
            # Table of Predictions
            st.markdown("#### Tabela de Valores Projetados")
            df_proj = pd.DataFrame({
                'Ano': future_years,
                'Previsão (Toneladas)': future_preds,
                'Previsão (Kilogramas)': future_preds * 1000
            })
            
            # Add growth rate relative to the last real year
            last_real_year = train_df['Ano'].max()
            last_real_val = train_df[train_df['Ano'] == last_real_year]['total_peso_t'].values[0]
            
            df_proj['Variação vs último ano real (%)'] = ((df_proj['Previsão (Toneladas)'] - last_real_val) / last_real_val) * 100
            
            st.dataframe(
                df_proj.style.format({
                    'Previsão (Toneladas)': '{:,.2f} t',
                    'Previsão (Kilogramas)': '{:,.0f} kg',
                    'Variação vs último ano real (%)': '{:+.2f}%'
                }),
                use_container_width=True,
                hide_index=True
            )
