import numpy as np
from numpy import concatenate
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
# from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error 
from pandas import DataFrame
from pandas import concat
from datetime import datetime
import pickle
from math import sqrt

import mlflow.keras

mlflow.keras.autolog()

train_store = pd.read_csv(r"C:\Users\sam\Desktop\pharma\data\train_store.csv")
# test_store = pd.read_csv("C:\Users\sam\Desktop\pharma\data\train_store.csv")

train_store['date'] = pd.to_datetime(train_store[['Day', 'Month', 'Year']], format='%Y%m%d')
train_store['date'] = train_store.date.dt.strftime('%Y-%m-%d')
train_store2 = train_store.set_index(['date'])


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    return rmse, mae, mse

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

    
    df.drop(categorical_column, axis=1, inplace=True)
    X = df.drop(['Customers', 'Sales', 'SalePerCustomer'], axis = 1) 
    col_name = df.columns.tolist()
    y=df.Sales
    frames = [X, y]
    result = pd.concat(frames, axis=1, join='inner')


    return result, col_name


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg


X, col_name = pre_processing(train_store2)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)
scaled_features_df = pd.DataFrame(scaled_features, index=X.index, columns=X.columns)

values = scaled_features_df.values
data = series_to_supervised(values, 1, 1)

# split into train and test sets
values = data.values
n_train_days = 365 * 1015
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

history = model.fit(train_X, train_y, epochs=10, batch_size=64, validation_data=(test_X, test_y), verbose=2, shuffle=False)


date = datetime.now()
dt_string = str(date.strftime("%d-%m-%Y-%H-%M-%S"))
pickle.dump(model, open('../models/{}.pkl'.format(dt_string), 'wb'))


# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)