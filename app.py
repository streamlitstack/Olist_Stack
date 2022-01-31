#Carregar bibliotecas

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model
import plotly.express as px
import acessando_blob_storage as abs
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#configuração da janela
st.set_page_config(
    page_title = 'Olist Store analytics',
    page_icon = '📊',
    layout = 'wide'
)

#Carregar Bases
abs.download_blob('presentation', 'final_model.pkl', 'final_model.pkl') # Modelo de Classificação
abs.download_blob('presentation', 'sellers_in_out4.csv', 'sellers_in_out4.csv') # Tabela de entradas e saídas 
abs.download_blob('presentation', 'df_cluster.csv', 'df_cluster.csv') # Tabela Cluster 
abs.download_blob('presentation', 'tb_base.csv', 'tb_base.csv') # Tabela que alimenta modelo
abs.download_blob('presentation', 'Logo.jpeg', 'Logo.jpeg') # Banner do App
abs.download_blob('presentation', 'retention1.csv', 'retention1.csv') # Banner do App

var_model = "final_model"

var_dataset_modelo = "tb_base.csv"
var_sellers_in_out= "sellers_in_out4.csv"
var_cluster="df_cluster.csv"
var_retention="retention1.csv"

#carregando o modelo treinado.
model = load_model(var_model)


#carregando o conjunto de dados.
#dataset = pd.read_csv(var_dataset)
dataset_modelo = pd.read_csv(var_dataset_modelo)
dataset_modelo = dataset_modelo.drop('target', axis=1)
dataset_modelo= dataset_modelo.drop(dataset_modelo.columns[0], axis=1)


dataset_cluster= pd.read_csv(var_cluster)
dataset_sellers=pd.read_csv(var_sellers_in_out)
dataset_retention=pd.read_csv(var_retention, index_col=0)


st.image("Logo.jpeg", width=120)

# título
st.title("Olist Analytics")

# subtítulo
st.markdown("Este é um Data App utilizado para exibir a solução de Machine Learning para Olist Analytics.")

# imprime o conjunto de dados usado
#st.dataframe(dataset_modelo.head())

# Gráfico entradas e saídas de Sellers-------------------------------------------------------------------------------
fig = px.line(dataset_sellers, x="data_pedido", y=["sellers_in", "sellers_out"], title='Entrada e Saída de Sellers na Olist', labels={"value": "Qtde de Sellers","data_pedido": "Mês","variable": "Sellers Status"})

fig.update_layout(title_text='Entrada e Saída de Sellers na Olist', title_x=0.5, title_font_size=25, 
legend=dict(
        x=0.9,  # value must be between 0 to 1.
        y=.1,   # value must be between 0 to 1.
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black")))

st.plotly_chart(fig, use_container_width=True)
st.markdown("""---""")
# histograma das saídas dos sellers-------------------------------------------------------------------------------
fig2 = px.histogram(dataset_modelo, x="dias_atividade", labels={"count": "Qtde de Sellers","dias_atividade": "Tempo de permanência (dias)"})

fig2.update_layout(title_text='Tempo de Permanência dos Sellers na Olist', title_x=0.5, title_font_size=25, 
legend=dict(
        x=0.9,  # value must be between 0 to 1.
        y=.1,   # value must be between 0 to 1.
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black")))

st.plotly_chart(fig2, use_container_width=True)
st.markdown("""---""")
# Analise de Retenção dos sellers-------------------------------------------------------------------------------



fig6, ax = plt.subplots(figsize=(20,10))
sns.set(font_scale=1.4)
sns.heatmap(dataset_retention, ax=ax, annot=True, fmt = '.0%', vmin= 0.0, vmax=0.5, cmap='summer_r', annot_kws={"size": 14} )
plt.style.use('fivethirtyeight')
plt.ylabel('Cohort Group', fontsize = 15) # y-axis label with fontsize 15
plt.xlabel('Cohort Period', fontsize = 15) # y-axis label with fontsize 15
plt.title('Cohort Analysis (%) - Retention Rates', fontsize=20)
st.pyplot(fig6)
st.markdown("""---""")
# Clusterização dos Sellers-------------------------------------------------------------------------------

fig4= px.scatter_3d(dataset_cluster,x='media_produtos_por_pedido',y='media_valor_pedido_sem_frete',z='dias_atividade',color='cluster')
fig4.update_layout(title_text='Clusterização dos Sellers', title_x=0.5, title_font_size=25) 
st.plotly_chart(fig4, use_container_width=True)

#st.subheader("Defina os atributos do empregado para predição de turnover")

st.markdown("""---""")

# inserindo um botão na tela
btn_predict = st.button("Realizar Classificação")

if btn_predict:
    #data_teste = pd.DataFrame(dataset_modelo)
    

    #imprime os dados de teste    
    #print(data_teste)

    #realiza a predição
    result = predict_model(model, data=dataset_modelo, raw_score=True)
    #pegando só coluna cluster da tabela df_cluster 
    clusters = dataset_cluster.filter(like='cluster')
    #concatenando com dataframe da predição
    df_final = pd.concat([result, clusters], axis=1)



    st.write(df_final)

    @st.cache
    def convert_df(dataset):
        return dataset.to_csv().encode('utf-8')


    csv = convert_df(df_final)

    st.download_button(
        "Press to Download",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
    )

    df_grafico_final = pd.DataFrame(df_final.groupby(['cluster', 'Label'])['id_vendedor'].count()).reset_index()

    fig7 = px.bar(df_grafico_final, x="cluster", y="id_vendedor", color="Label", hover_data=['cluster'], barmode = 'stack')
    st.plotly_chart(fig7, use_container_width=True)


    df_final_hist=df_final.loc[df_final['Label']==1]
    df_hist_cluster0=df_final_hist.loc[df_final_hist['cluster']==0]
    df_hist_cluster1=df_final_hist.loc[df_final_hist['cluster']==1]
    df_hist_cluster2=df_final_hist.loc[df_final_hist['cluster']==2]

    
    fig10=plt.figure(figsize=(10,6))
    plt.subplot(3,1,1);sns.histplot(df_hist_cluster0['Score_1'])
    plt.subplot(3,1,2);sns.histplot(df_hist_cluster1['Score_1'])
    plt.subplot(3,1,3);sns.histplot(df_hist_cluster2['Score_1'])
    st.pyplot(fig10)