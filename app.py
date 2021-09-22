import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import joblib as jb
import datetime
from sklearn.preprocessing import LabelEncoder

#função para carregar o dataset
@st.cache
def get_data():
    df = pd.read_csv("/Users/Issamu Umeda/Documents/GitHub/Property Maintenance Blight Ticket Fines/csv/train.csv", encoding='ISO-8859-1', low_memory=False)
    df = df[(df['city'].str.lower() == 'detroit') & (~df['compliance_detail'].str.contains('not responsible')) & (~df['compliance_detail'].str.contains('compliant by no fine'))].set_index('ticket_id')
    df['owed_amount'] = df['judgment_amount'] - df['discount_amount']
    df = pd.concat([df, df['ticket_issued_date'].str.extract(r'(?P<ticket_issued_year>\d{4})-(?P<ticket_issued_month>\d{2})')], axis=1)
    df['ticket_issued_semester'] = np.where(df.ticket_issued_month.astype('int') <= 6, 1, 2)
    return df


def transform(le, data_list):
    """
    This will transform the data_list to id list where the new values get assigned to Unknown class
    :param data_list:
    :return:
    """
    new_data_list = list(data_list)
    for unique_item in np.unique(data_list):
        if unique_item not in le.classes_:
            new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

    return le.transform(new_data_list)


#função para treinar o modelo
def get_models():
    rf = jb.load('/Users/Issamu Umeda/Documents/GitHub/Property Maintenance Blight Ticket Fines/model/rf_blight_ticket.pk;.z')
    return rf

#criando um dataframe
data = get_data()

#obtendo os modelos
model = get_models()
le_disposition = LabelEncoder().fit(list(data['disposition']) + ['Unknown'])

#título
st.title("Data App - Predicting Probability of Paying Blight Ticket on Time")

#subtítulo
st.markdown("This is a Data App used to show the Machine Learning solution for the Detroit Blight Ticket payment problem")

#verificando o dataset
st.subheader("Selecting the attributes")

#atributos para serem exibidos por padrão
defaultcols = ['disposition', 'ticket_issued_date', 'owed_amount']

#definindo atributos a partir do multiselect
cols = st.multiselect("Attributes", data.columns.tolist(), default=defaultcols)

#exibindo os top 10 registros do dataframe
st.dataframe(data[cols].head(10))

st.subheader("Distribution of blight tickets by sum of all fines and fees")

#definindo a faixa de valores
faixa_valores = st.slider("Owed Amount", float(data.owed_amount.min()), 1000., (0.0, 350.0))

#filtrando os dados
dados = data[data['owed_amount'].between(left=faixa_valores[0], right=faixa_valores[1])]

#plot a distribuição dos dados
f = px.histogram(dados, x="owed_amount", nbins=100, title="Distribution of Owed Amount")
f.update_xaxes(title="Owed Amount")
f.update_yaxes(title="Total People")
st.plotly_chart(f)

st.sidebar.subheader("Define the person attributes to prediction")

#mapeando dados do usuário para cada atributo
disposition = st.sidebar.selectbox('Judgement Type', data.disposition.unique())

#transformando o dado de entrada em valor binário
disposition = transform(le_disposition, pd.Series(disposition))[0]

issued_date = st.sidebar.date_input("Ticket Issued Date", value=datetime.date.today())
datee = datetime.datetime.strptime(str(issued_date), "%Y-%m-%d")
issued_month = datee.month
issued_semester = np.where(issued_month <= 6, 1, 2)

fine = st.sidebar.number_input("Fine Amount", value=data.fine_amount.mean())
admin = st.sidebar.selectbox("Admin Fee", (0, 20))
state = st.sidebar.selectbox("State Fee", (0, 10))
late = st.sidebar.selectbox("Late Fee", (0, fine*0.1))
discount = st.sidebar.number_input("Discount Amount", value=0)
owed_amount = fine + admin + state + late - discount
st.sidebar.text("Owed Amount")
st.sidebar.text(owed_amount)

#inserindo um botão na tela
btn_predict = st.sidebar.button("Predict")

#verifica se o botão foi acionado
if btn_predict:
    result = model.predict_proba([[owed_amount, issued_semester, disposition]])
    st.subheader("The chance of this person paying the blight ticket on time is:")
    result = str(round(result[0][1]*100,2)) + '%'
    st.write(result)
