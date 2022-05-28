import streamlit as st
import pandas as pd 
import os

path = os.path.dirname(__file__)
my_file = path+'/train.csv'
my_file2 = path+'/store.csv'
my_file3 = path+'/test.csv'
def write():
    
    with st.spinner("Loading Data ..."):
        st.title('Data description  ')
        # na_value=['',' ','nan','Nan','NaN','na', '<Na>']
        train = pd.read_csv(my_file, engine='python') #na_values=na_value)
        store = pd.read_csv(my_file2, engine= 'python')#na_values=na_value)
        full_train = pd.merge(left = train, right = store, how = 'inner', left_on = 'Store', right_on = 'Store')
        full_train = full_train.set_index('Store')
        st.write(full_train.sample(20))