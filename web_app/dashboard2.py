import streamlit as st
import pandas as pd
import numpy as np
# df = pd.DataFrame(
#     np.random.randn(50, 20),
#     columns=('col %d' % i for i in range(20)))


 
@st.cache
def get_data():
    train_store['date'] = pd.to_datetime(train_store[['Day', 'Month', 'Year']], format='%Y%m%d')
    return pd.read_csv( '../data/train_store.csv', engine = 'python')

df = get_data()
window_selection_c = st.sidebar.container() # create an empty container in the sidebar
window_selection_c.markdown("## Insights") # add a title to the sidebar container
sub_columns = window_selection_c.columns(2) #Split the container into two columns for start and end date


# Years = np.array([ 2015, 2014, 2013])  # TODO : include all stocks
years = window_selection_c.selectbox("select years", df['Year'].unique())
years = window_selection_c.selectbox("select years", df['Day'].unique())
years = window_selection_c.selectbox("select years", df['StateHoliday'].unique())
years = window_selection_c.selectbox("select years", df['Year'].unique())

store = window_selection_c.selectbox("select stores", df['Store'].unique())

chart_width = st.expander(label="chart width").slider("", 1000, 2800, 1400)


# makes = df['Store']
# years = df['Year']
# days = df['Day']
# engines = df['engine']
# components = df['components']
# make_choice = st.sidebar.selectbox('Select your vehicle:', makes)
# st.sidebar.selectbox('', years)
# st.sidebar.selectbox('', makes)
# st.sidebar.selectbox('', days)

st.write(df.head())