import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import joblib as jb
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

#função para carregar o dataset
@st.cache
def get_data():
    df = pd.read_csv("csv/train.csv", encoding='ISO-8859-1', low_memory=False)
    df = df[(df['city'].str.lower() == 'detroit') & (~df['compliance_detail'].str.contains('not responsible')) & (~df['compliance_detail'].str.contains('compliant by no fine'))].set_index('ticket_id')
    df['owed_amount'] = df['judgment_amount'] - df['discount_amount']
    df = pd.concat([df, df['ticket_issued_date'].str.extract(r'(?P<ticket_issued_year>\d{4})-(?P<ticket_issued_month>\d{2})')], axis=1)
    df['ticket_issued_semester'] = np.where(df.ticket_issued_month.astype('int') <= 6, 1, 2)

    bands_dict = {}
    band_columns = ['owed_amount']
    for column in band_columns:
        band_name = column + '_band'
        if column == 'owed_amount':
            df[band_name] = pd.cut(df[column].drop(df[df[column] > 400].index), 5)
        else:
            df[band_name] = pd.cut(df[column], 5)
        bands = df.groupby(band_name).agg({'compliance': len})
        bands_dict[column] = bands
        df[band_name] = df[band_name].astype(str)
        for number, band in enumerate(bands.index):
            bands_dict[column].iloc[number] = number
            df.loc[df[band_name] == str(band), band_name] = number
        if column == 'owed_amount':
            df[band_name] = df[band_name].replace('nan', 4)
        df[band_name] = df[band_name].astype(int)
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
    rf = jb.load('model/rf_blight_ticket.pk;.z')
    scaler = jb.load('model/scaler_blight_ticket.pk;.z')
    le_disposition = jb.load('model/le_disposition_blight_ticket.pk;.z')
    le_violation_code = jb.load('model/le_violation_code_blight_ticket.pk;.z')
    return rf, scaler, le_disposition, le_violation_code


def add_value_labels(ax, orientation, xspace=0, yspace=0, percentage=False):
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        if orientation == 'v':
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
            label = rect.get_height()
        elif orientation == 'h':
            y_value = rect.get_y() + rect.get_height() / 4
            x_value = rect.get_width()
            label = rect.get_width()

        # Vertical alignment for positive values
        va = 'bottom'

        # Define the label format
        if percentage==True:
            label = "{:.1%}".format(label)
        else:
            label = "{:,.0f}".format(label)

        # Create annotation
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(xspace, yspace),
            textcoords="offset points",
            ha='center',
            va=va)


def plot_percent_category(df, field, xvalues, compliance=False, xticks=None, legend_loc='best', xspace=0, yspace=0,
                          legend_pos=None):
    total = []
    non_compliant = []
    compliant = []

    total_values = df[field].value_counts().sort_index()
    non_compliant_values = df[df['compliance'] == 0][field].value_counts()
    compliant_values = df[df['compliance'] == 1][field].value_counts()

    if xticks == None:
        xticks = xvalues

    for value in xvalues:
        if value in total_values.index:
            total.append(total_values[value])
        else:
            total.append(0)

        if value in non_compliant_values.index:
            non_compliant.append(non_compliant_values[value])
        else:
            non_compliant.append(0)

        if value in compliant_values.index:
            compliant.append(compliant_values[value])
        else:
            compliant.append(0)

    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(111)
    w = 1
    folds = np.linspace(5, len(xvalues) * 5, len(xvalues))

    if compliance == False:
        title = 'Customer percentage distribution by {} and compliance'.format(field)
        ax.bar([element - w / 2 for element in folds],
               [number if sum(total) == 0 else number / sum(total) for number in non_compliant], width=w)
        ax.bar([element + w / 2 for element in folds],
               [number if sum(total) == 0 else number / sum(total) for number in compliant], width=w)
    else:
        title = 'Compliance percentage by {}'.format(field)
        ax.bar([element - w / 2 for element in folds],
               [number if total == 0 else number / total for number, total in zip(non_compliant, total)], width=w)
        ax.bar([element + w / 2 for element in folds],
               [number if total == 0 else number / total for number, total in zip(compliant, total)], width=w)

    add_value_labels(ax, orientation='v', yspace=5, percentage=True, xspace=xspace)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.xticks(folds, xticks)
    plt.title(title, fontweight='bold', size=15)
    legend = ['Non-compliant', 'Compliant']
    if legend_pos == None:
        plt.legend(legend, framealpha=0, loc=legend_loc)
    else:
        plt.legend(legend, framealpha=0, bbox_to_anchor=legend_pos)
    return fig


#criando um dataframe
data = get_data()

#obtendo os modelos
model, scaler, le_disposition, le_violation_code = get_models()

#título
st.title("Data App - Predicting Probability of Paying Blight Ticket on Time")

#subtítulo
st.markdown("This is a Data App used to show the Machine Learning solution for the Detroit Blight Ticket payment problem")

#verificando o dataset
st.subheader("Selecting the attributes")

#atributos para serem exibidos por padrão
defaultcols = ['disposition', 'ticket_issued_date', 'violation_code', 'violation_description', 'owed_amount']

#definindo atributos a partir do multiselect
cols = st.multiselect("Attributes", data.columns.tolist(), default=defaultcols)

#exibindo os top 10 registros do dataframe
st.dataframe(data[cols].head(10))

st.subheader("Distribution of blight tickets by sum of all fines and fees")

#definindo a faixa de valores
faixa_valores = st.slider("Owed Amount", float(data.owed_amount.min()), float(data.owed_amount.max()), (0.0, 400.0))

#filtrando os dados
dados = data[data['owed_amount'].between(left=faixa_valores[0], right=faixa_valores[1])]

#plot a distribuição dos dados
f = px.histogram(dados, x="owed_amount", nbins=100, title="Distribution of Owed Amount")
f.update_xaxes(title="Owed Amount")
f.update_yaxes(title="Total People")
st.plotly_chart(f)

#cria gráfico de taxa de compliance para valores de owed_amount
st.subheader("Compliance ratio by owed amount bands")
owed_amount_ticks = ['x <= 113.6', '113.6 < x <= 175.2', '175.2 < x <= 236.8', '236.8 < x <= 298.4', 'x > 298.4']
order = list(np.linspace(0,4,5))
fig_owed_amount = plot_percent_category(data, 'owed_amount_band', order, xticks=owed_amount_ticks, compliance=True, legend_loc=9)
st.pyplot(fig_owed_amount)

#cria gráfico de taxa de compliance por disposition
st.subheader("Compliance ratio by disposition type")
disposition_ticks = list(data['disposition'].value_counts().index)
fig_disposition = plot_percent_category(data, 'disposition', disposition_ticks, xticks=disposition_ticks, compliance=True)
st.pyplot(fig_disposition)

#cria gráfico de taxa de compliance por ticket_issued_semester
st.subheader("Compliance ratio by ticket issued semester")
order = [1, 2]
fig_semester = plot_percent_category(data, 'ticket_issued_semester', order, xticks=order, compliance=True)
st.pyplot(fig_semester)

st.sidebar.subheader("Define the person attributes to prediction")

#mapeando dados do usuário para cada atributo
disposition = st.sidebar.selectbox('Judgement Type', data.disposition.unique())


violation_description = st.sidebar.selectbox('Violation', sorted(data.violation_description.unique()))
data_violation_code = data[data['violation_description'] == violation_description].violation_code.unique()
violation_code = st.sidebar.selectbox('Violation Code', data_violation_code)
violation_code = transform(le_violation_code, pd.Series(violation_code))[0]

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
    result = model.predict_proba([[owed_amount, issued_semester, violation_code, disposition]])
    st.subheader("The chance of this person paying the blight ticket on time is:")
    st.sidebar.subheader("The chance of this person paying the blight ticket on time is:")
    result = str(round(result[0][1]*100,2)) + '%'
    st.write(result)
    st.sidebar.write(result)
