import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# To fill missing values

# To Split our train data
from sklearn.model_selection import train_test_split
# df = pd.DataFrame(
#     np.random.randn(50, 20),
#     columns=('col %d' % i for i in range(20)))


 
@st.cache
def get_data():
    df = pd.read_csv( '../data/train_store.csv', engine = 'python')
    df['date'] = pd.to_datetime(df[['Day', 'Month', 'Year']], format='%Y%m%d')
    return df

def pre_processing(df):
    #droping the auction id since it has no value for the train
    try:
        df.drop('Unnamed: 0', axis=1, inplace=True) 
    except:
        pass

    # numr_col = pre.get_numerical_columns(df) 
    # categorical_column = pre.get_categorical_columns(df)
    numerical_column = df.select_dtypes(exclude="object").columns.tolist()
    categorical_column = df.select_dtypes(include="object").columns.tolist()

    # Get column names have less than 10 more than 2 unique values
    to_one_hot_encoding = [col for col in categorical_column if df[col].nunique() <= 10 and df[col].nunique() > 2]
    one_hot_encoded_columns = pd.get_dummies(df[to_one_hot_encoding])
    df = pd.concat([df, one_hot_encoded_columns], axis=1)

    # Get Categorical Column names thoose are not in "to_one_hot_encoding"
    # to_label_encoding = [col for col in categorical_column if not col in to_one_hot_encoding]
    # le = LabelEncoder()
    # df[to_label_encoding] = df[to_label_encoding].apply(le.fit_transform)

    # df.drop(['date', 'browser'], axis=1, inplace=True)
    df.drop(categorical_column, axis=1, inplace=True)
    X = df.drop(['Customers', 'Sales', 'SalePerCustomer'], axis = 1) 
    col_name = X.columns.tolist()
    y=np.log(df.Sales)


    # y = df['brand_awareness']
    # X = df.drop(["brand_awareness"], axis=1)

    return X, y, col_name

df = get_data()


window_selection_c = st.sidebar.container() # create an empty container in the sidebar
window_selection_c.markdown("## Insights") # add a title to the sidebar container
sub_columns = window_selection_c.columns(2) #Split the container into two columns for start and end date


# Years = np.array([ 2015, 2014, 2013])  # TODO : include all stocks
# years = window_selection_c.selectbox("Store_id", df['Store '].unique())
store = window_selection_c.selectbox("select stores", df['Store'].unique())
Dayofweek = window_selection_c.selectbox("select day of week", df['DayOfWeek'].unique())
open = window_selection_c.selectbox("open", df['Open'].unique())
promo = window_selection_c.selectbox("isPromo", df['Promo'].unique())
holiday = window_selection_c.selectbox("isHoliday", df['StateHoliday'].unique())
sch = window_selection_c.selectbox("isschoolday", df['SchoolDay'].unique())
stort = window_selection_c.selectbox("isPromo", df['StoreType'].unique())
CompetitionDistance = window_selection_c.selectbox("isPromo", df['CompetitionDistance'].unique())
CompetitionOpenSinceMonth = window_selection_c.selectbox("isPromo", df['CompetitionOpenSinceMonth'].unique())
CompetitionOpenSinceYear = window_selection_c.selectbox("isPromo", df['CompetitionOpenSinceYear'].unique())
Promo2 = window_selection_c.selectbox("isPromo", df['Promo2'].unique())
Promo2SinceWeek = window_selection_c.selectbox("isPromo", df['Promo2SinceWeek'].unique())
Promo2SinceYear = window_selection_c.selectbox("isPromo", df['PromoInterval'].unique())







# state_holiday = window_selection_c.selectbox("select years", df['StateHoliday'].unique())
# date = window_selection_c.date_input('enter a date you want to forcast')
# promo = window_selection_c.selectbox("isPromo", df['Promo'].unique())


# date = window_selection_c.date_input('enter a date you want to forcast')
# promo = window_selection_c.selectbox("isPromo", df['Promo'].unique())

# store = window_selection_c.selectbox("select stores", df['Store'].unique())
# date = window_selection_c.date_input('enter a date you want to forcast')
# promo = window_selection_c.selectbox("isPromo", df['Promo'].unique())

# store = window_selection_c.selectbox("select stores", df['Store'].unique())

# chart_width = st.expander(label="chart width").slider("", 1000, 2800, 1400)






X, y, col_name = pre_processing(df)
y_test, y_train, X_test, X_train = train_test_split(y, X, test_size=0.75, shuffle=False)
model_pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('model',RandomForestRegressor(n_estimators = 10, max_depth=5))])
model_pipeline.fit(X_train, y_train)
# model_pipeline.predict(test)
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