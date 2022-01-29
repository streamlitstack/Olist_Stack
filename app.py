#carregando as bibliotecas
import pandas as pd
import streamlit as st
#from minio import Minio
#import joblib
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model
import plotly.express as px

#configuração da janela
st.set_page_config(
    page_title = 'Olist Store analytics',
    page_icon = '📊',
    layout = 'wide'
)

var_model = "model"
var_dataset = "dataset.csv"
var_dataset_modelo = "dataset_modelo.csv"

#carregando o modelo treinado.
model = load_model(var_model)


#carregando o conjunto de dados.
dataset = pd.read_csv(var_dataset)
dataset_modelo = pd.read_csv(var_dataset_modelo)

print (dataset.head())

st.image("lg_app_olist-min.jpeg", width=100)

# título
st.title("Olist Analytics")

# subtítulo
st.markdown("Este é um Data App utilizado para exibir a solução de Machine Learning para Olist Analytics.")

# imprime o conjunto de dados usado
st.dataframe(dataset.head())

fig= px.scatter_3d(dataset,x='dias_na_base',y='products_by_orders',z='ticket_medio',color='inativo')

st.plotly_chart(fig, use_container_width=True)

st.subheader("Defina os atributos do empregado para predição de turnover")

# inserindo um botão na tela
btn_predict = st.button("Realizar Classificação")

if btn_predict:
    data_teste = pd.DataFrame(dataset_modelo)
    




    
    #imprime os dados de teste    
    print(data_teste)

    #realiza a predição
    result = predict_model(model, data=data_teste)
    
    st.write(result)

    @st.cache
    def convert_df(dataset):
        return dataset.to_csv().encode('utf-8')


    csv = convert_df(result)

    st.download_button(
        "Press to Download",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
    )


