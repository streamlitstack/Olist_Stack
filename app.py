#Carregar bibliotecas

import pandas as pd
import streamlit as st
#import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model
import plotly.express as px
import acessando_blob_storage as abs
import seaborn as sns

#configuração da janela
st.set_page_config(
    page_title = 'Olist Store analytics',
    page_icon = '📊',
    layout = 'wide'
)

#Carregar Bases
abs.download_blob('presentation', 'final_model.pkl', 'final_model.pkl') # Modelo de Classificação
#abs.download_blob('presentation', 'tb_final.csv', 'tb_final.csv') # Tabela Matriz
abs.download_blob('presentation', 'sellers_in_out.csv', 'sellers_in_out.csv') # Tabela de entradas e saídas 
abs.download_blob('presentation', 'df_cluster.csv', 'df_cluster.csv') # Tabela Cluster 
abs.download_blob('presentation', 'tb_base.csv', 'tb_base.csv') # Tabela que alimenta modelo
abs.download_blob('presentation', 'banner.jpeg', 'banner.jpeg') # Banner do App

var_model = "final_model"
#var_dataset = "tb_final.csv"
var_dataset_modelo = "tb_base.csv"
var_sellers_in_out= "sellers_in_out.csv"
var_cluster="df_cluster.csv"

#carregando o modelo treinado.
model = load_model(var_model)


#carregando o conjunto de dados.
#dataset = pd.read_csv(var_dataset)
dataset_modelo = pd.read_csv(var_dataset_modelo)
dataset_cluster= pd.read_csv(var_cluster)


print (dataset_modelo.head())

#st.image("banner.jpeg", width=100)
st.image("banner.jpeg")

# título
st.title("Olist Analytics")

# subtítulo
st.markdown("Este é um Data App utilizado para exibir a solução de Machine Learning para Olist Analytics.")

# imprime o conjunto de dados usado
st.dataframe(dataset_modelo.head())


#st.line_chart(data=var_sellers_in_out, use_container_width=True)


#fig= px.scatter_3d(dataset_cluster,x='media_produtos_por_pedido',y='media_valor_pedido_sem_frete',z='dias_atividade',color='cluster')
#st.plotly_chart(fig, use_container_width=True)

st.subheader("Defina os atributos do empregado para predição de turnover")

# inserindo um botão na tela
btn_predict = st.button("Realizar Classificação")

if btn_predict:
    data_teste = pd.DataFrame(dataset_modelo)
    




    
    #imprime os dados de teste    
    print(data_teste)

    #realiza a predição
    result = predict_model(model, data=data_teste)
    #pegando só coluna cluster da tabela df_cluster 
    clusters = dataset_cluster.filter(like='cluster')
    #concatenando com dataframe da predição
    df_final = pd.concat([clusters, result], axis=1)



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