import streamlit as st



def write():
   
    with st.spinner("Loading Home ..."):
        st.title('Rossmann Pharmaceuticals')
        #st.image('../assets/ross.jpg', use_column_width=True)
        st.write(
            """
           A streamlit app for buliding time series forcasting to predict sales
                """
        )