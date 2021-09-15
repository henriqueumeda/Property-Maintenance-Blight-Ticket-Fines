import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

#função para carregar o dataset
@st.cache
def get_data():
    df = pd.read_csv("/Users/Issamu Umeda/Documents/GitHub/Property Maintenance Blight Ticket Fines/csv/train.csv", encoding='ISO-8859-1', low_memory=False)
    df = df[(df['city'].str.lower() == 'detroit') & (~df['compliance_detail'].str.contains('not responsible')) & (~df['compliance_detail'].str.contains('compliant by no fine'))].set_index('ticket_id')
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
def train_model():
    data = get_data()
    columns = ['disposition', 'violation_code', 'judgment_amount']
    X = data[columns]
    y = data['compliance']
    X['disposition'] = transform(le_disposition, X['disposition'])
    X['violation_code'] = transform(le_violation, X['violation_code'])
    rf = RandomForestClassifier(bootstrap= True, max_depth=10, max_features='sqrt', min_samples_leaf=1, min_samples_split=5, n_estimators=10)
    rf.fit(X, y)
    return rf


#criando um dataframe
data = get_data()

#treinando o modelo
columns = ['disposition', 'violation_code', 'judgment_amount']
X = data[columns]
y = data['compliance']
le_disposition = LabelEncoder().fit(list(X['disposition']) + ['Unknown'])
le_violation = LabelEncoder().fit(list(X['violation_code']) + ['Unknown'])
model = train_model()

#título
st.title("Data App - Predicting Probability of Paying Blight Ticket on Time")

#subtítulo
st.markdown("This is a Data App used to show the Machine Learning solution for the Detroit Blight Ticket payment problem")

#verificando o dataset
st.subheader("Selecting the attributes")

#atributos para serem exibidos por padrão
defaultcols = ['disposition', 'violation_code', 'violation_description', 'judgment_amount']

#definindo atributos a partir do multiselect
cols = st.multiselect("Attributes", data.columns.tolist(), default=defaultcols)

#exibindo os top 10 registros do dataframe
st.dataframe(data[cols].head(10))

st.subheader("Distribution of blight tickets by sum of all fines and fees")

#definindo a faixa de valores
faixa_valores = st.slider("Judgment Amount", float(data.judgment_amount.min()), 1000., (0.0, 350.0))

#filtrando os dados
dados = data[data['judgment_amount'].between(left=faixa_valores[0], right=faixa_valores[1])]

#plot a distribuição dos dados
f = px.histogram(dados, x="judgment_amount", nbins=100, title="Distribution of Judgment Amount")
f.update_xaxes(title="Judgment Amount")
f.update_yaxes(title="Total People")
st.plotly_chart(f)

st.sidebar.subheader("Define the person attributes to prediction")

#mapeando dados do usuário para cada atributo
disposition = st.sidebar.selectbox('Judgement Type', data.disposition.unique())

#transformando o dado de entrada em valor binário
disposition = transform(le_disposition, pd.Series(disposition))[0]

violation_desc = st.sidebar.selectbox('Violation', sorted(data.violation_description.unique()))

violation = st.sidebar.selectbox('Violation', data.loc[data['violation_description'] == violation_desc, 'violation_code'].unique())

#transformando o dado de entrada em valor binário
violation = transform(le_violation, pd.Series(violation))[0]

fine = st.sidebar.number_input("Fine Amount", value=data.fine_amount.mean())
admin = st.sidebar.number_input("Admin Fee", value=0)
state = st.sidebar.number_input("State Fee", value=0)
late = st.sidebar.number_input("Late Fee", value=0)
judgment_amount = fine + admin + state + late
st.sidebar.text("Judgment Amount")
st.sidebar.text(judgment_amount)

#inserindo um botão na tela
btn_predict = st.sidebar.button("Predict")

#verifica se o botão foi acionado
if btn_predict:
    result = model.predict_proba([[disposition, violation, judgment_amount]])
    st.subheader("The chance of this person paying the blight ticket on time is:")
    result = str(round(result[0][1]*100,2)) + '%'
    st.write(result)
