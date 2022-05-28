import streamlit as st
import pandas as pd
import numpy as np
# import awesome_streamlit as ast
# from fbprophet import Prophet
# from fbprophet.diagnostics import performance_metrics
# from fbprophet.diagnostics import cross_validation
# from fbprophet.plot import plot_cross_validation_metric
import pickle
import base64


st.title('Time Series Forecasting Using Streamlit')

@st.cache
def get_data():
    df = pd.read_csv( 'train.csv', engine = 'python')
    df['date'] = pd.to_datetime(df[['Day', 'Month', 'Year']], format='%Y-%m-%d')
    df2 = pd.DataFrame(columns = ['ds', 'y'])
    df2['ds'] = df['date']
    df2['y'] = df['Sales']
    return df, df2

df, df2 = get_data()
st.write(df.head())
st.write("SELECT FORECAST PERIOD")

window_selection_c = st.sidebar.container() # create an empty container in the sidebar
window_selection_c.markdown("## Insights") # add a title to the sidebar container
sub_columns = window_selection_c.columns(2) #Split the container into two columns for start and end date


# Years = np.array([ 2015, 2014, 2013])  # TODO : include all stocks
store = window_selection_c.selectbox("Store_id", df['Store'].unique())
holiday = window_selection_c.selectbox("isHoliday", df['StateHoliday'].unique())
# state_holiday = window_selection_c.selectbox("select years", df['StateHoliday'].unique())
date = window_selection_c.date_input('enter a date you want to forcast')
promo = window_selection_c.selectbox("isPromo", df['Promo'].unique())



# infile = open('*.pkl','rb')
# model = pickle.load(infile)
# # define the model
# model = Prophet()
# # fit the model
# model.fit(df2)
