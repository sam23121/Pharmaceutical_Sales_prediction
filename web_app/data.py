import streamlit as st
import pandas as pd 

def write():
    
    with st.spinner("Loading Data ..."):
        st.title('Data description  ')
        # na_value=['',' ','nan','Nan','NaN','na', '<Na>']
        train = pd.read_csv('../web_app/train.csv', engine='python') #na_values=na_value)
        store = pd.read_csv('./store.csv', engine= 'python')#na_values=na_value)
        full_train = pd.merge(left = train, right = store, how = 'inner', left_on = 'Store', right_on = 'Store')
        full_train = full_train.set_index('Store')
        st.write(full_train.sample(20))