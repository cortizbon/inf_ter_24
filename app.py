import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots
from utils import gini, COLORES, gini_by_department, lorenz_curve

# configuración streamlit

st.set_page_config(
    page_title="Ingresos territoriales - 2024",
    layout="wide",)

colors = ["#009999", "#CBECEF"]  
cmap = LinearSegmentedColormap.from_list("my_cmap", colors, N=256)

st.title("Ingresos territoriales - 2024")

# Cargar los datos

## carga de presupuesto

pres = pd.read_csv('datasets/ejec_2324_clean.csv')
pres['recaudo_cons'] = (pres['recaudo_cons'] / 1000000).round(2).round(2)  # Convertir a millones de pesos

## carga de sgp

sgp = pd.read_csv('datasets/sgp_2324.csv')
## carga de sgr

sgr = pd.read_csv('datasets/sgr_2324.csv')

## carga de mapas


# tabs (3)

tab1, tab2, tab3 = st.tabs(["Presupuesto", "SGP", "SGR"])

# Tab 1: Presupuesto

with tab1:
    st.header("Presupuesto")
    st.subheader("Entidades territoriales")

    a = (pres[(pres['Año'] == 2024)]
     .groupby('clas_gen')['Total Recaudo']
     .sum())
    fig = go.Figure(data=[go.Pie(labels=a.index,
                                 values=a.values,
                                 hole=0.7,
                                 textinfo='percent'
                                 )])
    # dont show names or percent in the pie chart
    fig.update_traces(textposition='inside')
    
    fig.update_traces(marker=dict(colors=["#D9D9ED", "#2F399B", '#1A1F63',"#2635bf"]),
                      pull=[0.1, 0.1, 0.1, 0.1],
                      showlegend=False)
    fig.update_layout(title_text="Presupuesto 2024",
                      title_font=dict(size=20, color="#1A1F63"),
                      font=dict(size=14, color="#1A1F63"),
                      paper_bgcolor="#FFE9C5",
                      plot_bgcolor="#FFE9C5",
                      legend=dict(title_text="Clasificación general", title_font=dict(size=16, color="#1A1F63")))
    # reduce margins in plot
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    # change hover info: only show name and increase the size of the hover text
    fig.update_traces(hoverinfo='label+percent', hoverlabel=dict(font_size=16, font_color="#1A1F63"))
    st.plotly_chart(fig, key=1)

    # lollipop para cambio en el recaudo entre 2023 y 2024 por clasificacion general
    pres_2023 = pres[(pres['Año'] == 2023)].groupby('clas_gen')['recaudo_cons'].sum().reset_index()
    pres_2024 = pres[(pres['Año'] == 2024)].groupby('clas_gen')['recaudo_cons'].sum().reset_index()
    pres_comparison = pd.merge(pres_2023, pres_2024, on='clas_gen', suffixes=('_2023', '_2024'))
    pres_comparison = pres_comparison.sort_values(by='recaudo_cons_2024', ascending=False).head(15).sort_values(by='recaudo_cons_2024')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pres_comparison['recaudo_cons_2024'],
        y=pres_comparison['clas_gen'],
        mode='markers',
        text=pres_comparison['clas_gen'],
        name='2024',
        textposition='top center',
        marker=dict(size=10, color='#1A1F63'),
        hoverinfo='text',
        hovertext=[f"{row['clas_gen']}: {row['recaudo_cons_2024']}" for index, row in pres_comparison.iterrows()]
    ))
    fig.add_trace(go.Scatter(
        x=pres_comparison['recaudo_cons_2023'],
        y=pres_comparison['clas_gen'],
        mode='markers',
        name='2023',
        marker=dict(size=10, color="#81D3CD"),
        hoverinfo='text',
        hovertext=[f"{row['clas_gen']}: {row['recaudo_cons_2023']}" for index, row in pres_comparison.iterrows()]
    ))
    fig.update_layout(
        title="Comparación de recaudo por clasificación general (2023 vs 2024)",
        title_font=dict(size=20, color="#1A1F63"),
        xaxis_title="Recaudo (en millones de pesos)",
        yaxis_title="Clasificación general",
        font=dict(size=14, color="#1A1F63"),
        paper_bgcolor="#FFE9C5",
        plot_bgcolor="#FFE9C5",
        legend=dict(title_text="Año", title_font=dict(size=16, color="#81D3CD")),
        xaxis=dict(tickformat=".2f"),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_layout(
        height=45 * len(pres_comparison),  # 40 px per classification
        margin=dict(l=100, r=50, t=50, b=50),  # More left margin for long classification names
        yaxis=dict(automargin=True)  # Avoid cutting labels
    )
    # change hover info: only show name and increase the size of the hover text
    fig.update_traces(hoverinfo='text', hoverlabel=dict(font_size=16, font_color="#1A1F63"))
    st.plotly_chart(fig, key=2)

    # b = (pres[(pres['Año'] == 2024) & (pres['Tipo de Entidad'] == 'Departamento')]
    #      .groupby(['Código DANE', 'Entidad'])['Total Recaudo']
    #      .sum()
    #      .reset_index()
    #      .assign(dpto_ccdgo=lambda x: x['Código DANE'] // 1000)
    #      .drop(columns=['Código DANE']))
    
    # geo_deptos = gpd.GeoDataFrame(b.astype({'dpto_ccdgo':'int'}).merge(deptos[['dpto_ccdgo', 'geometry']].astype({"dpto_ccdgo":'int'}), on='dpto_ccdgo', how='left'))
    # b1 = (pres[(pres['Año'] == 2024) & (pres['Tipo de Entidad'] == 'Departamento')]
    #      .groupby(['Código DANE', 'Entidad'])['recaudo_pc_cons']
    #      .sum()
    #      .reset_index()
    #      .assign(dpto_ccdgo=lambda x: x['Código DANE'] // 1000)
    #      .drop(columns=['Código DANE']))
    
    # geo_deptos1 = gpd.GeoDataFrame(b1.astype({'dpto_ccdgo':'int'}).merge(deptos[['dpto_ccdgo', 'geometry']].astype({"dpto_ccdgo":'int'}), on='dpto_ccdgo', how='left'))
    # fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    # # gráfico de mapa de departamentos con matplotlib y geopandas, usando paleta de colores de "#1A1F63" a "#81D3CD"
    # geo_deptos.plot(column='Total Recaudo', ax=ax[0], legend=True,
    #                legend_kwds={'label': "Total Recaudo (millones de pesos)",
    #                             'orientation': "horizontal"},
    #                cmap=cmap, edgecolor="#1A1F63", linewidth=0.2)
    # fig.patch.set_facecolor("#FFE9C5")
    # ax[0].set_title('Recaudo por Departamento - 2024', fontsize=20, color="#1A1F63")
    # ax[0].set_axis_off()
    
    
    # geo_deptos1.plot(column='recaudo_pc_cons', ax=ax[1], legend=True,
    #                legend_kwds={'label': "Recaudo per cápita",
    #                             'orientation': "horizontal"},
    #                cmap=cmap, edgecolor="#1A1F63", linewidth=0.2)
    # fig.patch.set_facecolor("#FFE9C5")
    # ax[1].set_title('Recaudo por Departamento - 2024', fontsize=20, color="#1A1F63")
    # ax[1].set_axis_off()
    
    # plt.tight_layout()
    # st.pyplot(fig)

    # b = (pres[(pres['Año'] == 2024) & (pres['Tipo de Entidad'] == 'Municipio')]
    #      .groupby(['Código DANE', 'Entidad'])['Total Recaudo']
    #      .sum()
    #      .reset_index()
    #      .rename(columns={'Código DANE': 'mpio_cdpmp'}))
    
    # geo_muns = gpd.GeoDataFrame(b.astype({'mpio_cdpmp':'int'}).merge(muns[['mpio_cdpmp', 'geometry']].astype({"mpio_cdpmp":'int'}), on='mpio_cdpmp', how='left'))

    # b1 = (pres[(pres['Año'] == 2024) & (pres['Tipo de Entidad'] == 'Municipio')]
    #      .groupby(['Código DANE', 'Entidad'])['recaudo_pc_cons']
    #      .sum()
    #      .reset_index()
    #      .rename(columns={'Código DANE': 'mpio_cdpmp'}))
    
    # geo_muns1 = gpd.GeoDataFrame(b1.astype({'mpio_cdpmp':'int'}).merge(muns[['mpio_cdpmp', 'geometry']].astype({"mpio_cdpmp":'int'}), on='mpio_cdpmp', how='left'))

    # # gráfico de mapa de departamentos con matplotlib y geopandas, usando paleta de colores de "#1A1F63" a "#81D3CD"
    # fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    # geo_muns.plot(column='Total Recaudo', ax=ax[0], legend=True,
    #               missing_kwds={
    #                   "color": "lightgrey",},
    #                legend_kwds={'label': "Total Recaudo (millones de pesos)",
    #                             'orientation': "horizontal"},
    #                cmap=cmap, edgecolor="#1A1F63", linewidth=0.2)
    
    # ax[0].set_title('Recaudo por Departamento - 2024', fontsize=20, color="#1A1F63")
    # ax[0].set_axis_off()
    # geo_muns1.plot(column='recaudo_pc_cons', ax=ax[1], legend=True,
    #               missing_kwds={
    #                   "color": "lightgrey",},
    #                legend_kwds={'label': "Recaudo per cápita",
    #                             'orientation': "horizontal"},
    #                cmap=cmap, edgecolor="#1A1F63", linewidth=0.2)
    # fig.patch.set_facecolor("#FFE9C5")
    # ax[1].set_title('Recaudo por Departamento - 2024', fontsize=20, color="#1A1F63")
    # ax[1].set_axis_off()
    
    # plt.tight_layout()
    # st.pyplot(fig)
    





    
    
   

    st.divider()
    st.subheader("Por departamento")

    # crear un panel para tener dos gráficos con make_subplots

    # --- Data for Donut Chart ---
    a = (pres[(pres['Año'] == 2024) & (pres['Tipo de Entidad'] == 'Departamento')]
        .groupby('clas_gen')['Total Recaudo']
        .sum())

    # --- Data for Bar Chart ---
    b = (pres[(pres['Año'] == 2024) & (pres['Tipo de Entidad'] == 'Departamento')]
        .groupby(['Categoría', 'clas_gen'])['recaudo_cons']
        .sum()
        .unstack())
    b = b.div(b.sum(axis=1), axis=0) * 100  # proportions

        # --- Create subplot figure ---
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'domain'}, {'type': 'xy'}]],
        subplot_titles=("Presupuesto 2024", "Presupuesto por categoría"),
        horizontal_spacing=0.1  # Reduce space between plots
    )

    # Donut Chart
    fig.add_trace(
        go.Pie(
            labels=a.index,
            values=a.values,
            hole=0.7,
            textinfo='percent',
            marker=dict(colors=["#D9D9ED", "#2F399B", '#1A1F63', "#2635bf"]),
            pull=[0.1, 0.1, 0.1, 0.1],
            hoverinfo='label+percent',
            showlegend=True  # Show legend here
        ),
        row=1, col=1
    )

    # Bar Chart
    for i, col in enumerate(b.columns):
        fig.add_trace(
            go.Bar(
                x=b.index,
                y=b[col],
                name=col,
                marker=dict(color=["#D9D9ED", "#2F399B", '#1A1F63', "#2635bf"][i]),
                hovertemplate=f"{col}",
                hoverinfo='none',
                text=b[col].apply(lambda x: f"{x:.1f}"),
            ),
            row=1, col=2
        )

    # Layout and Aesthetics
    fig.update_layout(
        title='Presupuesto 2024',
        title_font=dict(size=20, color="#1A1F63"),
        font=dict(size=14, color="#1A1F63"),
        paper_bgcolor="#FFE9C5",
        plot_bgcolor="#FFE9C5",
        legend=dict(
            title_text="Clasificación general",
            title_font=dict(size=16, color="#1A1F63"),
            orientation="h",  # horizontal legend
            y=-0.1,  # move below the chart
            x=0.5,
            xanchor="center"
        ),
        xaxis_title="Categoría",
        xaxis=dict(tickangle=-45),
        margin=dict(l=40, r=40, t=80, b=80),  # more bottom margin for legend
        barmode='stack',
        hoverlabel=dict(font_size=16, font_color="#1A1F63"),
        height=500,
        width=900,  # make the figure wider,
        showlegend=False # Show legend for the bar chart
    )
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)


    col1, col2 = st.columns(2)
    with col1:
        # treemap con las proporciones de recaudo por clasificacion ofpuj
        a = pres[pres['Año'] == 2024].groupby(['clas_gen', 'clasificacion_ofpuj'])['recaudo_cons'].sum().reset_index()
        fig = px.treemap(a, path=[px.Constant('Presupuesto'), 'clas_gen', 'clasificacion_ofpuj'],
                        values='recaudo_cons',
                        color='clasificacion_ofpuj',
                        color_discrete_sequence=["#262947", "#1A1F63","#2F399B" ,"#D9D9ED",],
                        title="Proporciones de recaudo por clasificación OFPUJ",
                        labels={'recaudo_cons': 'Recaudo (millones de pesos)', 'clasificacion_ofpuj': 'Clasificación OFPUJ'},
                        hover_data=['clas_gen'])
        fig.update_layout(
            title_font=dict(size=20, color="#1A1F63"),
            font=dict(size=14, color="#1A1F63"),
            paper_bgcolor="#FFE9C5",
            plot_bgcolor="#FFE9C5",
            legend=dict(title_text="Clasificación general", title_font=dict(size=16, color="#1A1F63")),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        # change hover info: only show name and increase the size of the hover text
        fig.update_traces(hoverinfo='text', hoverlabel=dict(font_size=16, font_color="#1A1F63"))
        st.plotly_chart(fig, key=3)

    with col2:
        # lollipop chart 2023 vs 2024, x axis: total recaudo, y axis: departament, no lines
        pres_2023 = pres[(pres['Año'] == 2023) & (pres['Tipo de Entidad'] == 'Departamento')].groupby('Entidad')['recaudo_cons'].sum().reset_index()
        pres_2024 = pres[(pres['Año'] == 2024) & (pres['Tipo de Entidad'] == 'Departamento')].groupby('Entidad')['recaudo_cons'].sum().reset_index() 
        pres_comparison = pd.merge(pres_2023, pres_2024, on='Entidad', suffixes=('_2023', '_2024'))
        pres_comparison = pres_comparison.sort_values(by='recaudo_cons_2024', ascending=False).head(15).sort_values(by='recaudo_cons_2024')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pres_comparison['recaudo_cons_2024'],
            y=pres_comparison['Entidad'],
            mode='markers',
            text=pres_comparison['Entidad'],
            name='2024',
            textposition='top center',
            marker=dict(size=10, color='#1A1F63'),
            hoverinfo='text',
            hovertext=[f"{row['Entidad']}: {row['recaudo_cons_2024']}" for index, row in pres_comparison.iterrows()]
        ))
        fig.add_trace(go.Scatter(
            x=pres_comparison['recaudo_cons_2023'],
            y=pres_comparison['Entidad'],
            mode='markers',
            name='2023',
            marker=dict(size=10, color="#81D3CD"),
            hoverinfo='text',
            hovertext=[f"{row['Entidad']}: {row['recaudo_cons_2023']}" for index, row in pres_comparison.iterrows()]
        ))
        fig.update_layout(
            title="Comparación de recaudo por departamento (2023 vs 2024)",
            title_font=dict(size=20, color="#1A1F63"),
            xaxis_title="Recaudo (en millones de pesos)",
            yaxis_title="Departamento",
            font=dict(size=14, color="#1A1F63"),
            paper_bgcolor="#FFE9C5",
            plot_bgcolor="#FFE9C5",
            legend=dict(title_text="Año", title_font=dict(size=16, color="#81D3CD")),
            xaxis=dict(tickformat=".2f"),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        fig.update_layout(
            height=35 * len(pres_comparison),  # 40 px per department
            margin=dict(l=100, r=50, t=50, b=50),  # More left margin for long department names
            yaxis=dict(automargin=True)  # Avoid cutting labels
        )   

        # change hover info: only show name and increase the size of the hover text
        fig.update_traces(hoverinfo='text', hoverlabel=dict(font_size=16, font_color="#1A1F63"))
        st.plotly_chart(fig, key=4)

    


    depar = st.selectbox("Selecciona un departamento", options=pres[pres['Tipo de Entidad'] == 'Departamento']['Entidad'].unique())
    fil_dep = pres[pres['Departamento'] == depar]
    fild = pres[pres['Entidad'] == depar]
    cols_dep = st.columns(2)
    with cols_dep[0]:
        tot_24 = fild[fild['Año'] == 2024]['recaudo_cons'].sum()
        tot_23 = fild[fild['Año'] == 2023]['recaudo_cons'].sum()
        variacion = (tot_24 - tot_23) / tot_23 * 100
        variacion = variacion.round(1)
        tot_24 = round((tot_24 / 1000000), 1)
        st.metric(label="Total recaudo 2024", value=f"{tot_24:,.1f} bill.", delta=f"{variacion:.1f}%")
        a = fild[fild['Año'] == 2024].groupby(['clas_gen', 'clasificacion_ofpuj'])['recaudo_cons'].sum().reset_index()
        fig = px.treemap(a, path=[px.Constant('Presupuesto'), 'clas_gen', 'clasificacion_ofpuj'],
                        values='recaudo_cons',
                        color='clasificacion_ofpuj',
                        color_discrete_sequence=["#262947", "#1A1F63","#2F399B" ,"#D9D9ED",],
                        title="Proporciones de recaudo por clasificación OFPUJ",
                        labels={'recaudo_cons': 'Recaudo (millones de pesos)', 'clasificacion_ofpuj': 'Clasificación OFPUJ'},
                        hover_data=['clas_gen'])
        fig.update_layout(
            title_font=dict(size=20, color="#1A1F63"),
            font=dict(size=14, color="#1A1F63"),
            paper_bgcolor="#FFE9C5",
            plot_bgcolor="#FFE9C5",
            legend=dict(title_text="Clasificación general", title_font=dict(size=16, color="#1A1F63")),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        # change hover info: only show name and increase the size of the hover text
        fig.update_traces(hoverinfo='text', hoverlabel=dict(font_size=16, font_color="#1A1F63"))
        st.plotly_chart(fig, key=5)        
    with cols_dep[1]:
        tot_24_pc = fild[fild['Año'] == 2024]['recaudo_pc_cons'].sum()
        tot_23_pc = fild[fild['Año'] == 2023]['recaudo_pc_cons'].sum()
        variacion_pc = tot_24_pc - tot_23_pc
        variacion_pc = (variacion_pc / tot_23_pc * 100).round(1)
        tot_24_pc = round((tot_24_pc / 1000000), 1)
        st.metric(label="Recaudo per cápita 2024", value=f"{tot_24_pc:,.0f} mill.", delta=f"{variacion_pc:.1f}%")
        pres_2023 = fild[fild['Año'] == 2023].groupby('clas_gen')['recaudo_cons'].sum().reset_index()
        pres_2024 = fild[fild['Año'] == 2024].groupby('clas_gen')['recaudo_cons'].sum().reset_index()
        pres_comparison = pd.merge(pres_2023, pres_2024, on='clas_gen', suffixes=('_2023', '_2024'))
        pres_comparison = pres_comparison.sort_values(by='recaudo_cons_2024')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pres_comparison['recaudo_cons_2024'],
            y=pres_comparison['clas_gen'],
            mode='markers',
            text=pres_comparison['clas_gen'],
            name='2024',
            textposition='top center',
            marker=dict(size=10, color='#1A1F63'),
            hoverinfo='text',
            hovertext=[f"{row['clas_gen']}: {row['recaudo_cons_2024']}" for index, row in pres_comparison.iterrows()]
        ))
        fig.add_trace(go.Scatter(
            x=pres_comparison['recaudo_cons_2023'],
            y=pres_comparison['clas_gen'],
            mode='markers',
            name='2023',
            marker=dict(size=10, color="#81D3CD"),
            hoverinfo='text',
            hovertext=[f"{row['clas_gen']}: {row['recaudo_cons_2023']}" for index, row in pres_comparison.iterrows()]
        ))
        fig.update_layout(
            title=f"Comparación de recaudo por clasificación OFPUJ en {depar} (2023 vs 2024)",
            title_font=dict(size=20, color="#81D3CD"),
            xaxis_title="Recaudo (en millones de pesos)",
            yaxis_title="Clasificación general",
            font=dict(size=14, color="#1A1F63"),
            paper_bgcolor="#FFE9C5",
            plot_bgcolor="#FFE9C5",
            legend=dict(title_text="Año", title_font=dict(size=16, color="#81D3CD")),
            xaxis=dict(tickformat=".2f"),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        fig.update_layout(
            height=50 * len(pres_comparison),  # 40 px per department
            margin=dict(l=100, r=50, t=50, b=50),  # More left margin for long department names
            yaxis=dict(automargin=True)  # Avoid cutting labels
        )
        st.plotly_chart(fig, key=85)
        # calculate the absolute variation per clas_gen between 2023 and 2024

    g = (fild.pivot_table(index='clasificacion_ofpuj', columns='Año', values='recaudo_pc', aggfunc='sum')
         .assign(var=lambda x: x[2024] - x[2023])
         .sort_values(by='var', ascending=False))
    vari = (g[2024].sum() - g[2023].sum()) / g[2023].sum()
    g['vari'] = (g['var'] * vari) / g['var'].sum()

    val_y = list(g['vari'].values)
    val_y = [round(x * 100, 2) for x in val_y]
    val_y.insert(0, 100)
    val_y.append(0)
    # waterfall con las variaciones (total 2023, variaciones, total 2024) 



    val_x = list(g.index)
    val_x.insert(0, "Total 2023")
    val_x.append("Total 2024")
    val_measure = ["relative"] * (len(val_x) - 1) 
    val_measure.append("total")


    fig = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = val_measure,
        x = val_x,
        textposition = "outside",
        y = val_y,
        connector = {"line":{"color":"#FFE9C5"}},
        decreasing = {"marker":{"color":"#D8841C", "line":{"color":"#D8841C"}}},
        increasing = {"marker":{"color":"#0FB7B3", "line":{"color":"#0FB7B3"}}},
        totals = {"marker":{"color":"#1A1F63", "line":{"color":"#1A1F63"}}}  
    ))

    fig.update_layout(
            title = f"Cambio en el recaudo en {depar} - 2023 a 2024",
            showlegend = False
    )

    st.plotly_chart(fig, key=7)
    # 

    st.divider()



    
    st.subheader("Por municipio")
    # --- Data for Donut Chart ---
    a = (pres[(pres['Año'] == 2024) & (pres['Tipo de Entidad'] == 'Municipio')]
        .groupby('clas_gen')['Total Recaudo']
        .sum())

    # --- Data for Bar Chart ---
    b = (pres[(pres['Año'] == 2024) & (pres['Tipo de Entidad'] == 'Municipio')]
        .groupby(['Categoría', 'clas_gen'])['recaudo_cons']
        .sum()
        .unstack())
    b = b.div(b.sum(axis=1), axis=0) * 100  # proportions

        # --- Create subplot figure ---
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'domain'}, {'type': 'xy'}]],
        subplot_titles=("Presupuesto 2024", "Presupuesto por categoría"),
        horizontal_spacing=0.1  # Reduce space between plots
    )

    # Donut Chart
    fig.add_trace(
        go.Pie(
            labels=a.index,
            values=a.values,
            hole=0.7,
            textinfo='percent',
            marker=dict(colors=["#D9D9ED", "#2F399B", '#1A1F63', "#2635bf"]),
            pull=[0.1, 0.1, 0.1, 0.1],
            hoverinfo='label+percent',
            showlegend=True  # Show legend here
        ),
        row=1, col=1
    )

    # Bar Chart
    for i, col in enumerate(b.columns):
        fig.add_trace(
            go.Bar(
                x=b.index,
                y=b[col],
                name=col,
                marker=dict(color=["#D9D9ED", "#2F399B", '#1A1F63', "#2635bf"][i]),
                hovertemplate=f"{col}",
                hoverinfo='none',
                text=b[col].apply(lambda x: f"{x:.1f}"),
            ),
            row=1, col=2
        )

    # Layout and Aesthetics
    fig.update_layout(
        title='Presupuesto 2024',
        title_font=dict(size=20, color="#1A1F63"),
        font=dict(size=14, color="#1A1F63"),
        paper_bgcolor="#FFE9C5",
        plot_bgcolor="#FFE9C5",
        legend=dict(
            title_text="Clasificación general",
            title_font=dict(size=16, color="#1A1F63"),
            orientation="h",  # horizontal legend
            y=-0.1,  # move below the chart
            x=0.5,
            xanchor="center"
        ),
        xaxis_title="Categoría",
        xaxis=dict(tickangle=-45),
        margin=dict(l=40, r=40, t=80, b=80),  # more bottom margin for legend
        barmode='stack',
        hoverlabel=dict(font_size=16, font_color="#1A1F63"),
        height=500,
        width=900,  # make the figure wider,
        showlegend=False # Show legend for the bar chart
    )
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)


    col1, col2 = st.columns(2)
    with col1:
        # treemap con las proporciones de recaudo por clasificacion ofpuj
        a = pres[pres['Año'] == 2024].groupby(['clas_gen', 'clasificacion_ofpuj'])['recaudo_cons'].sum().reset_index()
        fig = px.treemap(a, path=[px.Constant('Presupuesto'), 'clas_gen', 'clasificacion_ofpuj'],
                        values='recaudo_cons',
                        color='clasificacion_ofpuj',
                        color_discrete_sequence=["#262947", "#1A1F63","#2F399B" ,"#D9D9ED",],
                        title="Proporciones de recaudo por clasificación OFPUJ",
                        labels={'recaudo_cons': 'Recaudo (millones de pesos)', 'clasificacion_ofpuj': 'Clasificación OFPUJ'},
                        hover_data=['clas_gen'])
        fig.update_layout(
            title_font=dict(size=20, color="#1A1F63"),
            font=dict(size=14, color="#1A1F63"),
            paper_bgcolor="#FFE9C5",
            plot_bgcolor="#FFE9C5",
            legend=dict(title_text="Clasificación general", title_font=dict(size=16, color="#1A1F63")),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        # change hover info: only show name and increase the size of the hover text
        fig.update_traces(hoverinfo='text', hoverlabel=dict(font_size=16, font_color="#1A1F63"))
        st.plotly_chart(fig, key=8)

    with col2:
        # lollipop chart 2023 vs 2024, x axis: total recaudo, y axis: departament, no lines
        pres_2023 = pres[(pres['Año'] == 2023) & (pres['Tipo de Entidad'] == 'Municipio')].groupby('Entidad')['recaudo_cons'].sum().reset_index()
        pres_2024 = pres[(pres['Año'] == 2024) & (pres['Tipo de Entidad'] == 'Municipio')].groupby('Entidad')['recaudo_cons'].sum().reset_index() 
        pres_comparison = pd.merge(pres_2023, pres_2024, on='Entidad', suffixes=('_2023', '_2024'))
        pres_comparison = pres_comparison.sort_values(by='recaudo_cons_2024', ascending=False).head(15).sort_values(by='recaudo_cons_2024')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pres_comparison['recaudo_cons_2024'],
            y=pres_comparison['Entidad'],
            mode='markers',
            text=pres_comparison['Entidad'],
            name='2024',
            textposition='top center',
            marker=dict(size=10, color='#1A1F63'),
            hoverinfo='text',
            hovertext=[f"{row['Entidad']}: {row['recaudo_cons_2024']}" for index, row in pres_comparison.iterrows()]
        ))
        fig.add_trace(go.Scatter(
            x=pres_comparison['recaudo_cons_2023'],
            y=pres_comparison['Entidad'],
            mode='markers',
            name='2023',
            marker=dict(size=10, color="#81D3CD"),
            hoverinfo='text',
            hovertext=[f"{row['Entidad']}: {row['recaudo_cons_2023']}" for index, row in pres_comparison.iterrows()]
        ))
        fig.update_layout(
            title="Comparación de recaudo por municipio (2023 vs 2024)",
            title_font=dict(size=20, color="#1A1F63"),
            xaxis_title="Recaudo (en millones de pesos)",
            yaxis_title="Municipio",
            font=dict(size=14, color="#1A1F63"),
            paper_bgcolor="#FFE9C5",
            plot_bgcolor="#FFE9C5",
            legend=dict(title_text="Año", title_font=dict(size=16, color="#81D3CD")),
            xaxis=dict(tickformat=".2f"),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        fig.update_layout(
            height=35 * len(pres_comparison),  # 40 px per department
            margin=dict(l=100, r=50, t=50, b=50),  # More left margin for long department names
            yaxis=dict(automargin=True)  # Avoid cutting labels
        )   

        # change hover info: only show name and increase the size of the hover text
        fig.update_traces(hoverinfo='text', hoverlabel=dict(font_size=16, font_color="#1A1F63"))
        st.plotly_chart(fig, key=9)


    mun = st.selectbox("Selecciona un municipio", options=fil_dep[fil_dep['Tipo de Entidad'] == 'Municipio']['Entidad'].unique())
    fild = fil_dep[fil_dep['Entidad'] == mun]
    cols_dep = st.columns(2)
    with cols_dep[0]:
        tot_24 = fild[fild['Año'] == 2024]['recaudo_cons'].sum()
        tot_23 = fild[fild['Año'] == 2023]['recaudo_cons'].sum()
        variacion = (tot_24 - tot_23) / tot_23 * 100
        variacion = variacion.round(1)
        tot_24 = round((tot_24 / 1000000), 2)
        st.metric(label="Total recaudo 2024", value=f"{tot_24:,.2f} bill.", delta=f"{variacion:.1f}%")
        a = fild[fild['Año'] == 2024].groupby(['clas_gen', 'clasificacion_ofpuj'])['recaudo_cons'].sum().reset_index()
        fig = px.treemap(a, path=[px.Constant('Presupuesto'), 'clas_gen', 'clasificacion_ofpuj'],
                        values='recaudo_cons',
                        color='clasificacion_ofpuj',
                        color_discrete_sequence=["#262947", "#1A1F63","#2F399B" ,"#D9D9ED",],
                        title="Proporciones de recaudo por clasificación OFPUJ",
                        labels={'recaudo_cons': 'Recaudo (millones de pesos)', 'clasificacion_ofpuj': 'Clasificación OFPUJ'},
                        hover_data=['clas_gen'])
        fig.update_layout(
            title_font=dict(size=20, color="#1A1F63"),
            font=dict(size=14, color="#1A1F63"),
            paper_bgcolor="#FFE9C5",
            plot_bgcolor="#FFE9C5",
            legend=dict(title_text="Clasificación general", title_font=dict(size=16, color="#1A1F63")),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        # change hover info: only show name and increase the size of the hover text
        fig.update_traces(hoverinfo='text', hoverlabel=dict(font_size=16, font_color="#1A1F63"))
        st.plotly_chart(fig, key=10)        
    with cols_dep[1]:
        tot_24_pc = fild[fild['Año'] == 2024]['recaudo_pc_cons'].sum()
        tot_23_pc = fild[fild['Año'] == 2023]['recaudo_pc_cons'].sum()
        variacion_pc = tot_24_pc - tot_23_pc
        variacion_pc = (variacion_pc / tot_23_pc * 100).round(2)
        tot_24_pc = round((tot_24_pc / 1000000), 2)
        st.metric(label="Recaudo per cápita 2024", value=f"{tot_24_pc:,.2f} mill.", delta=f"{variacion_pc:.1f}%")
        pres_2023 = fild[fild['Año'] == 2023].groupby('clas_gen')['recaudo_cons'].sum().reset_index()
        pres_2024 = fild[fild['Año'] == 2024].groupby('clas_gen')['recaudo_cons'].sum().reset_index()
        pres_comparison = pd.merge(pres_2023, pres_2024, on='clas_gen', suffixes=('_2023', '_2024'))
        pres_comparison = pres_comparison.sort_values(by='recaudo_cons_2024')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pres_comparison['recaudo_cons_2024'],
            y=pres_comparison['clas_gen'],
            mode='markers',
            text=pres_comparison['clas_gen'],
            name='2024',
            textposition='top center',
            marker=dict(size=10, color='#1A1F63'),
            hoverinfo='text',
            hovertext=[f"{row['clas_gen']}: {row['recaudo_cons_2024']}" for index, row in pres_comparison.iterrows()]
        ))
        fig.add_trace(go.Scatter(
            x=pres_comparison['recaudo_cons_2023'],
            y=pres_comparison['clas_gen'],
            mode='markers',
            name='2023',
            marker=dict(size=10, color="#81D3CD"),
            hoverinfo='text',
            hovertext=[f"{row['clas_gen']}: {row['recaudo_cons_2023']}" for index, row in pres_comparison.iterrows()]
        ))
        fig.update_layout(
            title=f"Comparación de recaudo por clasificación OFPUJ en {depar} (2023 vs 2024)",
            title_font=dict(size=20, color="#81D3CD"),
            xaxis_title="Recaudo (en millones de pesos)",
            yaxis_title="Clasificación general",
            font=dict(size=14, color="#1A1F63"),
            paper_bgcolor="#FFE9C5",
            plot_bgcolor="#FFE9C5",
            legend=dict(title_text="Año", title_font=dict(size=16, color="#81D3CD")),
            xaxis=dict(tickformat=".2f"),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        fig.update_layout(
            height=50 * len(pres_comparison),  # 40 px per department
            margin=dict(l=100, r=50, t=50, b=50),  # More left margin for long department names
            yaxis=dict(automargin=True)  # Avoid cutting labels
        )
        st.plotly_chart(fig, key=11)

    g = (fild.pivot_table(index='clasificacion_ofpuj', columns='Año', values='recaudo_pc', aggfunc='sum')
         .assign(var=lambda x: x[2024] - x[2023])
         .sort_values(by='var', ascending=False))
    vari = (g[2024].sum() - g[2023].sum()) / g[2023].sum()
    g['vari'] = (g['var'] * vari) / g['var'].sum()

    val_y = list(g['vari'].values)
    val_y = [round(x * 100, 2) for x in val_y]
    val_y.insert(0, 100)
    val_y.append(0)
    # waterfall con las variaciones (total 2023, variaciones, total 2024) 


    val_x = list(g.index)
    val_x.insert(0, "Total 2023")
    val_x.append("Total 2024")
    val_measure = ["relative"] * (len(val_x) - 1) 
    val_measure.append("total")

    fig = go.Figure(go.Waterfall(
        name = "20", orientation = "h",
        measure = val_measure,
        y = val_x,
        textposition = "outside",
        x = val_y,
        connector = {"line":{"color":"#FFE9C5"}},
        decreasing = {"marker":{"color":"#D8841C", "line":{"color":"#D8841C"}}},
        increasing = {"marker":{"color":"#0FB7B3", "line":{"color":"#0FB7B3"}}},
        totals = {"marker":{"color":"#1A1F63", "line":{"color":"#1A1F63"}}}  
    ))

    fig.update_layout(
            title = f"Cambio en el recaudo en {mun} - 2023 a 2024",
            showlegend = False
    )

    st.plotly_chart(fig, key=12)


    # a = (pres[(pres['Año'] == 2024) & (pres['Tipo de Entidad'] == 'Municipio')]
    #  .groupby('clas_gen')['Total Recaudo']
    #  .sum())
    
    # fig = go.Figure(data=[go.Pie(labels=a.index,
    #                              values=a.values,
    #                              hole=0.7,
    #                              textinfo='percent'
    #                              )])
    # # dont show names or percent in the pie chart
    # fig.update_traces(textposition='inside')
    
    # fig.update_traces(marker=dict(colors=["#D9D9ED", "#2F399B", '#1A1F63',"#2635bf"]),
    #                   pull=[0.1, 0.1, 0.1, 0.1],
    #                   showlegend=False)
    # fig.update_layout(title_text="Presupuesto 2024",
    #                   title_font=dict(size=20, color="#1A1F63"),
    #                   font=dict(size=14, color="#1A1F63"),
    #                   paper_bgcolor="#FFE9C5",
    #                   plot_bgcolor="#FFE9C5",
    #                   legend=dict(title_text="Clasificación general", title_font=dict(size=16, color="#1A1F63")))
    # # reduce margins in plot
    # fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    # # change hover info: only show name and increase the size of the hover text
    # fig.update_traces(hoverinfo='label+percent', hoverlabel=dict(font_size=16, font_color="#1A1F63"))
    # st.plotly_chart(fig)

    # # lollipop chart 2023 vs 2024, x axis: total recaudo, y axis: municipio, no lines
    # pres_2023 = fil_dep[(fil_dep['Año'] == 2023) & (fil_dep['Tipo de Entidad'] == 'Municipio')].groupby('Entidad')['recaudo_cons'].sum().reset_index()
    # pres_2024 = fil_dep[(fil_dep['Año'] == 2024) & (fil_dep['Tipo de Entidad'] == 'Municipio')].groupby('Entidad')['recaudo_cons'].sum().reset_index() 
    # pres_comparison = pd.merge(pres_2023, pres_2024, on='Entidad', suffixes=('_2023', '_2024'))
    # pres_comparison = pres_comparison.sort_values(by='recaudo_cons_2024', ascending=False).head(10).sort_values(by='recaudo_cons_2024')
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #     x=pres_comparison['recaudo_cons_2024'],
    #     y=pres_comparison['Entidad'],
    #     mode='markers',
    #     text=pres_comparison['Entidad'],
    #     name='2024',
    #     textposition='top center',
    #     marker=dict(size=10, color='#1A1F63'),
    #     hoverinfo='text',
    #     hovertext=[f"{row['Entidad']}: {row['recaudo_cons_2024']}" for index, row in pres_comparison.iterrows()]
    # ))
    # fig.add_trace(go.Scatter(
    #     x=pres_comparison['recaudo_cons_2023'],
    #     y=pres_comparison['Entidad'],
    #     mode='markers',
    #     name='2023',
    #     marker=dict(size=10, color="#81D3CD"),
    #     hoverinfo='text',
    #     hovertext=[f"{row['Entidad']}: {row['recaudo_cons_2023']}" for index, row in pres_comparison.iterrows()]
    # ))
    # fig.update_layout(
    #     title="Comparación de recaudo por municipio (2023 vs 2024)",
    #     title_font=dict(size=20, color="#1A1F63"),
    #     xaxis_title="Recaudo (en millones de pesos)",
    #     yaxis_title="Municipio",
    #     font=dict(size=14, color="#1A1F63"),
    #     paper_bgcolor="#FFE9C5",
    #     plot_bgcolor="#FFE9C5",
    #     legend=dict(title_text="Año", title_font=dict(size=16, color="#81D3CD")),
    #     xaxis=dict(tickformat=".2f"),
    #     margin=dict(l=20, r=20, t=50, b=20)
    # )
    # fig.update_layout(
    #     height=30 * len(pres_comparison),  # 40 px per department
    #     margin=dict(l=100, r=50, t=50, b=50),  # More left margin for long department names
    #     yaxis=dict(automargin=True)  # Avoid cutting labels
    # )
    # st.plotly_chart(fig)    

    # mun = st.selectbox("Selecciona un municipio", options=fil_dep[fil_dep['Tipo de Entidad'] == 'Municipio']['Entidad'].unique())

    # fil_mun = fil_dep[fil_dep['Entidad'] == mun]




    # pres_2023 = fil_mun[(fil_mun['Año'] == 2023) & (fil_mun['Tipo de Entidad'] == 'Municipio')].groupby('clasificacion_ofpuj')['recaudo_cons'].sum().reset_index()
    # pres_2024 = fil_mun[(fil_mun['Año'] == 2024) & (fil_mun['Tipo de Entidad'] == 'Municipio')].groupby('clasificacion_ofpuj')['recaudo_cons'].sum().reset_index() 
    # pres_comparison = pd.merge(pres_2023, pres_2024, on='clasificacion_ofpuj', suffixes=('_2023', '_2024'))
    # pres_comparison = pres_comparison.sort_values(by='recaudo_cons_2024')
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #     x=pres_comparison['recaudo_cons_2024'],
    #     y=pres_comparison['clasificacion_ofpuj'],
    #     mode='markers',
    #     text=pres_comparison['clasificacion_ofpuj'],
    #     name='2024',
    #     textposition='top center',
    #     marker=dict(size=10, color='#1A1F63'),
    #     hoverinfo='text',
    #     hovertext=[f"{row['clasificacion_ofpuj']}: {row['recaudo_cons_2024']}" for index, row in pres_comparison.iterrows()]
    # ))
    # fig.add_trace(go.Scatter(
    #     x=pres_comparison['recaudo_cons_2023'],
    #     y=pres_comparison['clasificacion_ofpuj'],
    #     mode='markers',
    #     name='2023',
    #     marker=dict(size=10, color="#81D3CD"),
    #     hoverinfo='text',
    #     hovertext=[f"{row['clasificacion_ofpuj']}: {row['recaudo_cons_2023']}" for index, row in pres_comparison.iterrows()]
    # ))
    # fig.update_layout(
    #     title="Comparación de recaudo por municipio (2023 vs 2024)",
    #     title_font=dict(size=20, color="#1A1F63"),
    #     xaxis_title="Recaudo (en millones de pesos)",
    #     yaxis_title="Municipio",
    #     font=dict(size=14, color="#1A1F63"),
    #     paper_bgcolor="#FFE9C5",
    #     plot_bgcolor="#FFE9C5",
    #     legend=dict(title_text="Año", title_font=dict(size=16, color="#81D3CD")),
    #     xaxis=dict(tickformat=".2f"),
    #     margin=dict(l=20, r=20, t=50, b=20)
    # )
    # fig.update_layout(
    #     height=30 * len(pres_comparison),  # 40 px per department
    #     margin=dict(l=100, r=50, t=50, b=50),  # More left margin for long department names
    #     yaxis=dict(automargin=True)  # Avoid cutting labels
    # )   

    # # change hover info: only show name and increase the size of the hover text
    # fig.update_traces(hoverinfo='text', hoverlabel=dict(font_size=16, font_color="#1A1F63"))
    # st.plotly_chart(fig) 


# Tab 2: SGP
with tab2:
    st.header("SGP")
    st.subheader("General")

    # donut chart using the SGP data, dividing it by Concepto and year 2024
    a = sgp.query("Año == 2024")
    a = a.groupby('Concepto')['Valor_pc'].sum().reset_index()
    a = a.sort_values(by='Valor_pc', ascending=False)
    fig = go.Figure(data=[go.Pie(labels=a['Concepto'],
                                 values=a['Valor_pc'],
                                    hole=0.7,
                                    textinfo='percent',
                                    marker=dict(colors=["#D9D9ED", "#2F399B", '#1A1F63', "#2635bf"]),
                                    pull=[0.1, 0.1, 0.1, 0.1],
                                    hoverinfo='label+percent',
                                    showlegend=True)])
    fig.update_layout(title_text="Presupuesto SGP 2024",
                      title_font=dict(size=20, color="#1A1F63"),
                      font=dict(size=14, color="#1A1F63"),
                      paper_bgcolor="#FFE9C5",
                        plot_bgcolor="#FFE9C5",
                        legend=dict(title_text="Concepto", title_font=dict(size=16, color="#1A1F63")))
    # reduce margins in plot
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    # change hover info: only show name and increase the size of the hover text
    fig.update_traces(hoverinfo='label+percent', hoverlabel=dict(font_size=16, font_color="#1A1F63"), showlegend=False)
    st.plotly_chart(fig, key=13)

    # grouped bar chart using the SGP data, dividing it by Concepto and year 2024, filter by Tipo de Entidad == departamento, color refers to Concepto
    b = sgp.query("Año == 2024 & TipoEntidad == 'Departamento'").groupby(['Concepto', 'Categoría'])['Valor_pop'].mean().reset_index()

    fig = px.bar(b, x='Categoría', y='Valor_pop', color='Concepto', barmode='group',
                 color_discrete_sequence=["#D9D9ED", "#2F399B", '#1A1F63', "#2635bf"],
                 title="Presupuesto SGP 2024 en los departamentos",
                 labels={'Valor_pop': 'Valor (millones de pesos)', 'Concepto': 'Concepto', 'Categoría': 'Categoría'},
                 hover_data=['Categoría'])
    fig.update_layout(
        title_font=dict(size=20, color="#1A1F63"),
        font=dict(size=14, color="#1A1F63"),
        paper_bgcolor="#FFE9C5",
        plot_bgcolor="#FFE9C5",
        legend=dict(title_text="Categoría", title_font=dict(size=16, color="#1A1F63")),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    # change hover info: only show name and increase the size of the hover text
    fig.update_traces(hoverinfo='text', hoverlabel=dict(font_size=16, font_color="#1A1F63"))
    st.plotly_chart(fig, key=134)

    # grouped bar chart using the SGP data, dividing it by Concepto and year 2024, filter by Tipo de Entidad == departamento, color refers to Concepto
    c = sgp.query("Año == 2024 & TipoEntidad == 'Municipio'").groupby(['Concepto', 'Categoría'])['Valor_pop'].mean().reset_index()

    fig = px.bar(c, x='Categoría', y='Valor_pop', color='Concepto', barmode='group',
                 color_discrete_sequence=["#D9D9ED", "#2F399B", '#1A1F63', "#2635bf"],
                 title="Presupuesto SGP 2024 por concepto y categoría",
                 labels={'Valor_pop': 'Valor (millones de pesos)', 'Concepto': 'Concepto', 'Categoría': 'Categoría'},
                 hover_data=['Categoría'])
    fig.update_layout(
        title_font=dict(size=20, color="#1A1F63"),
        font=dict(size=14, color="#1A1F63"),
        paper_bgcolor="#FFE9C5",
        plot_bgcolor="#FFE9C5",
        legend=dict(title_text="Categoría", title_font=dict(size=16, color="#1A1F63")),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    # change hover info: only show name and increase the size of the hover text
    fig.update_traces(hoverinfo='text', hoverlabel=dict(font_size=16, font_color="#1A1F63"))
    st.plotly_chart(fig, key=144)



    g = sgp.groupby(['Año', 'NombreEntidad', 'TipoEntidad','Concepto'])['Valor_pop'].sum().reset_index()

    index = ['Total', 'Educación', 'Salud', 'Propósito General']
    tabla_deptos = {2023: [],
             2024: []}
    tabla_munis = {2023: [],
             2024: []}
    # total
    g_total_depto_23 = g.groupby(['Año', 'NombreEntidad','TipoEntidad'])['Valor_pop'].sum().reset_index().query("Año == 2023 & TipoEntidad == 'Departamento'")['Valor_pop'].values
    g_total_depto_24 = g.groupby(['Año', 'NombreEntidad','TipoEntidad'])['Valor_pop'].sum().reset_index().query("Año == 2024 & TipoEntidad == 'Departamento'")['Valor_pop'].values
    g_total_muni_23 = g.groupby(['Año', 'NombreEntidad','TipoEntidad'])['Valor_pop'].sum().reset_index().query("Año == 2023 & TipoEntidad == 'Municipio'")['Valor_pop'].values
    g_total_muni_24 = g.groupby(['Año', 'NombreEntidad','TipoEntidad'])['Valor_pop'].sum().reset_index().query("Año == 2024 & TipoEntidad == 'Municipio'")['Valor_pop'].values
    
    for year in tabla_deptos.keys():
        for tipo_entidad in ['Departamento', 'Municipio']:
            for concepto in index:
                if concepto == 'Total':
                    if tipo_entidad == 'Departamento':
                        valor = g.groupby(['Año', 'NombreEntidad','TipoEntidad'])['Valor_pop'].sum().reset_index().query(f"Año == {year} & TipoEntidad == '{tipo_entidad}'")['Valor_pop'].values
                        valor = gini(valor)
                    else:
                        valor = g.groupby(['Año', 'NombreEntidad','TipoEntidad'])['Valor_pop'].sum().reset_index().query(f"Año == {year} & TipoEntidad == '{tipo_entidad}'")['Valor_pop'].values
                        valor = gini(valor)
                else:
                    valor = g.query(f"Concepto == '{concepto}' & Año == {year} & TipoEntidad == '{tipo_entidad}'")['Valor_pop'].values
                    valor = gini(valor)
                if tipo_entidad == 'Departamento':
                    tabla_deptos[year].append(valor)
                else:
                    tabla_munis[year].append(valor)
    tab_deptos =  pd.DataFrame(tabla_deptos, index=index)
    tab_munis = pd.DataFrame(tabla_munis, index=index)

    # lollipop chart 2023 vs. 2024, x axis: gini, y axis: Total, Educación, Salud, Propósito general, no lines
    g_2023 = tab_deptos.iloc[:-1, 0].reset_index()
    g_2024 = tab_deptos.iloc[:-1, 1].reset_index()
    g_comparison = pd.merge(g_2023, g_2024, on='index', suffixes=('_2023', '_2024'))
    g_comparison = g_comparison.sort_values(by=2024)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g_comparison[2024],
        y=g_comparison['index'],
        mode='markers',
        text=g_comparison['index'],
        name=2024,
        textposition='top center',
        marker=dict(size=10, color='#1A1F63'),
        hoverinfo='text',
        hovertext=[f"{row['index']}: {row[2024]}" for index, row in g_comparison.iterrows()]
    ))
    fig.add_trace(go.Scatter(
        x=g_comparison[2023],
        y=g_comparison['index'],
        mode='markers',
        name=2023,
        marker=dict(size=10, color="#81D3CD"),
        hoverinfo='text',
        hovertext=[f"{row['index']}: {row[2023]}" for index, row in g_comparison.iterrows()]
    ))
    fig.update_layout(
        title="Comparación del índice de Gini por concepto (2023 vs 2024)",
        title_font=dict(size=20, color="#1A1F63"),
        xaxis_title="Índice de Gini",
        yaxis_title="Concepto",
        font=dict(size=14, color="#1A1F63"),
        paper_bgcolor="#FFE9C5",
        plot_bgcolor="#FFE9C5",
        legend=dict(title_text="Año", title_font=dict(size=16, color="#81D3CD")),
        xaxis=dict(tickformat=".2f"),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_layout(
        height=60 * len(g_comparison),  # 40 px per concept
        margin=dict(l=100, r=50, t=50, b=50),  # More left margin for long concept names
        yaxis=dict(automargin=True)  # Avoid cutting labels
    )
    # change hover info: only show name and increase the size of the hover text
    fig.update_traces(hoverinfo='text', hoverlabel=dict(font_size=16, font_color="#1A1F63"))
    st.plotly_chart(fig, key=15)
    # lollipop chart 2023 vs. 2024, x axis: gini, y axis: Total, Educación, Salud, Propósito general, no lines
    g_2023 = tab_munis.iloc[:, 0].reset_index()
    g_2024 = tab_munis.iloc[:, 1].reset_index()
    g_comparison = pd.merge(g_2023, g_2024, on='index', suffixes=('_2023', '_2024'))
    g_comparison = g_comparison.sort_values(by=2024)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g_comparison[2024],
        y=g_comparison['index'],
        mode='markers',
        text=g_comparison['index'],
        name=2024,
        textposition='top center',
        marker=dict(size=10, color='#1A1F63'),
        hoverinfo='text',
        hovertext=[f"{row['index']}: {row[2024]}" for index, row in g_comparison.iterrows()]
    ))
    fig.add_trace(go.Scatter(
        x=g_comparison[2023],
        y=g_comparison['index'],
        mode='markers',
        name=2023,
        marker=dict(size=10, color="#81D3CD"),
        hoverinfo='text',
        hovertext=[f"{row['index']}: {row[2023]}" for index, row in g_comparison.iterrows()]
    ))
    fig.update_layout(
        title="Comparación del índice de Gini por concepto (2023 vs 2024)",
        title_font=dict(size=20, color="#1A1F63"),
        xaxis_title="Índice de Gini",
        yaxis_title="Concepto",
        font=dict(size=14, color="#1A1F63"),
        paper_bgcolor="#FFE9C5",
        plot_bgcolor="#FFE9C5",
        legend=dict(title_text="Año", title_font=dict(size=16, color="#81D3CD")),
        xaxis=dict(tickformat=".2f"),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_layout(
        height=45 * len(g_comparison),  # 40 px per concept
        margin=dict(l=100, r=50, t=50, b=50),  # More left margin for long concept names
        yaxis=dict(automargin=True)  # Avoid cutting labels
    )
    # change hover info: only show name and increase the size of the hover text
    fig.update_traces(hoverinfo='text', hoverlabel=dict(font_size=16, font_color="#1A1F63"))
    st.plotly_chart(fig, key=155)





    


    st.divider()
    st.subheader("Por departamento")
    # donut chart with totals
    a = sgp.query("Año == 2024 & TipoEntidad == 'Departamento'")
    a = a.groupby('Concepto')['Valor_pc'].sum().reset_index()
    a = a.sort_values(by='Valor_pc', ascending=False)
    fig = go.Figure(data=[go.Pie(labels=a['Concepto'],                                 values=a['Valor_pc'],
                                    hole=0.7,
                                    textinfo='percent',
                                    marker=dict(colors=["#D9D9ED", "#2F399B", '#1A1F63', "#2635bf"]),
                                    pull=[0.1, 0.1, 0.1, 0.1],
                                    hoverinfo='label+percent',
                                    showlegend=True)])
    fig.update_layout(title_text="Presupuesto SGP 2024 por departamento",
                      title_font=dict(size=20, color="#1A1F63"),
                      font=dict(size=14, color="#1A1F63"),
                      paper_bgcolor="#FFE9C5",
                        plot_bgcolor="#FFE9C5",
                        legend=dict(title_text="Concepto", title_font=dict(size=16, color="#1A1F63")))
    # reduce margins in plot
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    # change hover info: only show name and increase the size of the hover text
    fig.update_traces(hoverinfo='label+percent', hoverlabel=dict(font_size=16, font_color="#1A1F63"), showlegend=False)
    st.plotly_chart(fig, key=1555)
    # gini general por departamento

    deps = sgp[(sgp['TipoEntidad'] == 'Departamento') & (sgp['Año'] == 2024)]['NombreEntidad'].unique()
    depto = st.selectbox("Selecciona un departamento", options=deps)
    depar = sgp[sgp['NombreEntidad'] == depto]

    # treemap del departamento por concepto
    a = depar.groupby(['Concepto', 'Subconcepto', 'Subsubconcepto'])['Valor_pop'].mean().reset_index()
    fig = px.treemap(a, path=[px.Constant('SGP'), 'Concepto', 'Subconcepto', 'Subsubconcepto'],
                        values='Valor_pop',
                        color='Concepto',
                        color_discrete_sequence=["#D9D9ED", "#2F399B", '#1A1F63', "#2635bf"],
                        title=f"Proporciones de recaudo por concepto en {depto}",
                        labels={'Valor_pop': 'Valor (millones de pesos)', 'Concepto': 'Concepto'},
                        hover_data=['Concepto'])
    fig.update_layout(
        title_font=dict(size=20, color="#1A1F63"),
        font=dict(size=14, color="#1A1F63"),
        paper_bgcolor="#FFE9C5",
        plot_bgcolor="#FFE9C5",
        legend=dict(title_text="Concepto", title_font=dict(size=16, color="#1A1F63")),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    # change hover info: only show name and increase the size of the hover text
    fig.update_traces(hoverinfo='text', hoverlabel=dict(font_size=16, font_color="#1A1F63"))
    st.plotly_chart(fig, key=15555) 
    
    g = (depar.pivot_table(index='Concepto', columns='Año', values='Valor_pc', aggfunc='sum')
         .assign(var=lambda x: x[2024] - x[2023])
         .sort_values(by='var', ascending=False))
    vari = (g[2024].sum() - g[2023].sum()) / g[2023].sum()
    g['vari'] = (g['var'] * vari) / g['var'].sum()

    val_y = list(g['vari'].values)
    val_y = [round(x * 100, 2) for x in val_y]
    val_y.insert(0, 100)
    val_y.append(0)
    # waterfall con las variaciones (total 2023, variaciones, total 2024) 



    val_x = list(g.index)
    val_x.insert(0, "Total 2023")
    val_x.append("Total 2024")
    val_measure = ["relative"] * (len(val_x) - 1) 
    val_measure.append("total")


    fig = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = val_measure,
        x = val_x,
        textposition = "outside",
        y = val_y,
        connector = {"line":{"color":"#FFE9C5"}},
        decreasing = {"marker":{"color":"#D8841C", "line":{"color":"#D8841C"}}},
        increasing = {"marker":{"color":"#0FB7B3", "line":{"color":"#0FB7B3"}}},
        totals = {"marker":{"color":"#1A1F63", "line":{"color":"#1A1F63"}}}  
    ))

    fig.update_layout(
            title = f"Cambio en el recaudo en {depar} - 2023 a 2024",
            showlegend = False
    )

    st.plotly_chart(fig, key=555)       


    st.divider()
    st.subheader("Por municipio")

    # donut chart with totals
    a = sgp[(sgp['TipoEntidad'] == 'Municipio') & (sgp['Año'] == 2024)].groupby('Concepto')['Valor_pc'].sum().reset_index()
    a = a.sort_values(by='Valor_pc', ascending=False)
    fig = go.Figure(data=[go.Pie(labels=a['Concepto'], values=a['Valor_pc'],
                                    hole=0.7,
                                    textinfo='percent',
                                    marker=dict(colors=["#D9D9ED", "#2F399B", '#1A1F63', "#2635bf"]),
                                    pull=[0.1, 0.1, 0.1, 0.1],
                                    hoverinfo='label+percent',
                                    showlegend=True)])
    fig.update_layout(title_text=f"Presupuesto SGP 2024",
                      title_font=dict(size=20, color="#1A1F63"),
                      font=dict(size=14, color="#1A1F63"),
                      paper_bgcolor="#FFE9C5",
                        plot_bgcolor="#FFE9C5",
                        legend=dict(title_text="Subconcepto", title_font=dict(size=16, color="#1A1F63")))
    # reduce margins in plot
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    # change hover info: only show name and increase the size of the hover text
    fig.update_traces(hoverinfo='label+percent', hoverlabel=dict(font_size=16, font_color="#1A1F63"), showlegend=False)
    st.plotly_chart(fig, key=55)
    # gini general por departamento

    depto = st.selectbox("Selecciona un departamento", options=deps, key=3445)
    depar = sgp[(sgp['NombreDepartamento'] == depto)]
    muns = depar[(depar['TipoEntidad'] == 'Municipio') & (sgp['Año'] == 2024)]['NombreEntidad'].unique()
    mun = st.selectbox("Selecciona un municipio", options=muns)
    muni = depar[depar['NombreEntidad'] == mun]

    # treemap del municipio por concepto
    a = muni[muni['Año'] == 2024].groupby(['Concepto', 'Subconcepto', 'Subsubconcepto'])['Valor_pc'].sum().reset_index()
    fig = px.treemap(a, path=[px.Constant('SGP'), 'Concepto', 'Subconcepto', 'Subsubconcepto'],
                        values='Valor_pc',
                        color='Concepto',
                        color_discrete_sequence=["#D9D9ED", "#2F399B", '#1A1F63', "#2635bf"],
                        title=f"Proporciones de recaudo por concepto en {depto} - {mun}",
                        labels={'Valor_pc': 'Valor (millones de pesos)', 'Concepto': 'Concepto'},
                        hover_data=['Concepto'])
    fig.update_layout(
        title_font=dict(size=20, color="#1A1F63"),
        font=dict(size=14, color="#1A1F63"),
        paper_bgcolor="#FFE9C5",
        plot_bgcolor="#FFE9C5",
        legend=dict(title_text="Concepto", title_font=dict(size=16, color="#1A1F63")),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    # change hover info: only show name and increase the size of the hover text
    fig.update_traces(hoverinfo='text', hoverlabel=dict(font_size=16, font_color="#1A1F63"))
    st.plotly_chart(fig, key=6)

    # calculate the absolute variation per clas_gen between 2023 and 2024
    g = (muni.pivot_table(index='Concepto', columns='Año', values='Valor_pc', aggfunc='sum')
         .assign(var=lambda x: x[2024] - x[2023])
         .sort_values(by='var', ascending=False))

    vari = (g[2024].sum() - g[2023].sum()) / g[2023].sum()
    g['vari'] = (g['var'] * vari) / g['var'].sum()

    val_y = list(g['vari'].values)
    val_y = [round(x * 100, 2) for x in val_y]
    val_y.insert(0, 100)
    val_y.append(0)
    # waterfall con las variaciones (total 2023, variaciones, total 2024) 

    val_x = list(g.index)
    val_x.insert(0, "Total 2023")
    val_x.append("Total 2024")

    fig = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = ["relative", "relative", "relative", "relative", "relative", "relative", "relative", "total"],
        x = val_x,
        textposition = "outside",
        y = val_y,
        connector = {"line":{"color":"#FFE9C5"}},
        decreasing = {"marker":{"color":"#D8841C", "line":{"color":"#D8841C"}}},
        increasing = {"marker":{"color":"#0FB7B3", "line":{"color":"#0FB7B3"}}},
        totals = {"marker":{"color":"#1A1F63", "line":{"color":"#1A1F63"}}}  
    ))

    fig.update_layout(
            title = "Cambio en el ingreso por SGP - 2023 a 2024",
            showlegend = False
    )

    st.plotly_chart(fig, key=61)

# Tab 3: SGR
with tab3:
    st.header("SGR")

    # donut de asignaciones generales

    fig = px.treemap(sgr, path=[px.Constant("SGR"), 'Concepto', 'Subconcepto', 'Subsubconcepto'], values='Valor',
                     color_discrete_sequence=["#1A1F63","#2F399B" ,"#D9D9ED",])

    st.plotly_chart(fig, key=612)

    # barras de asignación directa por categoría en departamentos

    t = sgr[sgr['Tipo entidad'] == 'Departamento'].groupby(['Categoría', 'Subconcepto'])['Valor'].sum().reset_index()

    fig = px.bar(t, x='Categoría', y='Valor', color='Subconcepto', title='SGR por categoría y subconcepto en Departamentos',
                 color_discrete_sequence=["#1A1F63","#2F399B" ,"#D9D9ED",])
    st.plotly_chart(fig, key=6111)
    # barras de asignación directa por categoría en municipios

    t = sgr[sgr['Tipo entidad'] == 'Municipio'].groupby(['Categoría', 'Subconcepto'])['Valor'].sum().reset_index()

    fig = px.bar(t, x='Categoría', y='Valor', color='Subconcepto', title='SGR por categoría y subconcepto en Departamentos',
                 color_discrete_sequence=["#1A1F63","#2F399B" ,"#D9D9ED",])
    st.plotly_chart(fig, key=613)

    # seleccionar departamento
    deps = sgr[sgr['Tipo entidad'] == 'Departamento']['Departamento'].unique()
    depto = st.selectbox("Selecciona un departamento", options=deps)
    fil_dep = sgr[sgr['Departamento'] == depto]

    # donut chart con la asignación del departamento
    a = fil_dep.groupby('Subconcepto')['Valor'].sum().reset_index()
    a = a.sort_values(by='Valor', ascending=False)
    fig = go.Figure(data=[go.Pie(labels=a['Subconcepto'], values=a['Valor'],
                                    hole=0.7,
                                    textinfo='percent',
                                    marker=dict(colors=["#D9D9ED", "#2F399B", '#1A1F63', "#2635bf"]),
                                    pull=[0.1, 0.1, 0.1, 0.1],
                                    hoverinfo='label+percent',
                                    showlegend=True)])
    fig.update_layout(title_text=f"Presupuesto SGR 2024 - {depto}",
                      title_font=dict(size=20, color="#1A1F63"),
                      font=dict(size=14, color="#1A1F63"),
                        paper_bgcolor="#FFE9C5",
                        plot_bgcolor="#FFE9C5",
                        legend=dict(title_text="Subconcepto", title_font=dict(size=16, color="#1A1F63")))
    # reduce margins in plot
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    # change hover info: only show name and increase the size of the hover text
    fig.update_traces(hoverinfo='label+percent', hoverlabel=dict(font_size=16, font_color="#1A1F63"), showlegend=False)
    st.plotly_chart(fig, key=62)

    # seleccionar municipio

    muns = fil_dep[fil_dep['Tipo entidad'] == 'Municipio']['Entidad'].unique()
    mun = st.selectbox("Selecciona un municipio", options=muns)
    fil_mun = fil_dep[fil_dep['Entidad'] == mun]
    # donut chart con la asignación del municipio
    a = fil_mun.groupby('Subconcepto')['Valor'].sum().reset_index()
    a = a.sort_values(by='Valor', ascending=False)
    fig = go.Figure(data=[go.Pie(labels=a['Subconcepto'], values=a['Valor'],
                                    hole=0.7,
                                    textinfo='percent',
                                    marker=dict(colors=["#D9D9ED", "#2F399B", '#1A1F63', "#2635bf"]),
                                    pull=[0.1, 0.1, 0.1, 0.1],
                                    hoverinfo='label+percent',
                                    showlegend=True)])
    fig.update_layout(title_text=f"Presupuesto SGR 2024 - {depto} - {mun}",
                      title_font=dict(size=20, color="#1A1F63"),
                      font=dict(size=14, color="#1A1F63"),
                        paper_bgcolor="#FFE9C5",
                        plot_bgcolor="#FFE9C5",
                        legend=dict(title_text="Subconcepto", title_font=dict(size=16, color="#1A1F63")))
    # reduce margins in plot
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    # change hover info: only show name and increase the size of the hover text
    fig.update_traces(hoverinfo='label+percent', hoverlabel=dict(font_size=16, font_color="#1A1F63"), showlegend=False)
    st.plotly_chart(fig, key=621)
