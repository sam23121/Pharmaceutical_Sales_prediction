# import the child scripts
import streamlit as st
import awesome_streamlit as ast
import home
import data 
import plots
import pred
import dashboard

ast.core.services.other.set_logging_format()

# create the pages
PAGES = {
    "Home": home,
    "Data":data,
    "Data visualisations": plots,
    "Predictions": pred,
    "predictions": dashboard
}


# render the pages
def main():
   
    st.sidebar.title("Salse Prediction")
    selection = st.sidebar.selectbox("Select", list(PAGES.keys()))

    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        ast.shared.components.write_page(page)
    if selection =="Home":
        st.sidebar.title("INFORMATION")
        st.sidebar.info(
        """
        This App is created for Rosemann pharmaceutical company to 
        view predictions on sales across their stores 
        """
    )
    elif selection=="Predictions":
        st.sidebar.title("")


if __name__ == "__main__":
    main()