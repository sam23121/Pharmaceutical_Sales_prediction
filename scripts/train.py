import imp
import os
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join('../scripts')))

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# To fill missing values
from sklearn.impute import SimpleImputer

# To Split our train data
from sklearn.model_selection import train_test_split


# To Train our data
# from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn
import pickle

from logger import Logger
import logging
from ml import Ml
from preprocess import Preprocess
# from plot import Plot


# logging.basicConfig(level=logging.WARN)
# logger = logging.getLogger(__name__)

# from ml import ML
# ml = Ml()
# pre = Preprocess()
# pt = Plot()

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    return rmse, mae, mse

def pre_processing(df):
    #droping the auction id since it has no value for the train
    df.drop('Unnamed: 0', axis=1, inplace=True) 

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
    df.drop(['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'], axis=1, inplace=True)
    X = df.drop(['Customers', 'Sales', 'SalePerCustomer'], axis = 1) 
    col_name = X.columns.tolist()
    y=np.log(df.Sales)


    # y = df['brand_awareness']
    # X = df.drop(["brand_awareness"], axis=1)

    return X, y, col_name

# def feature_importance:
#     importance = model_pipeline.named_steps["model"].feature_importances_
#     # summarize feature importance
#     for i,v in enumerate(importance):
# 	print(col_name[i], ', Score: %.5f' % (v))
#     # plot feature importance
#     plt.bar([x for x in range(len(importance))], importance)
#     plt.show()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # np.random.seed(40)

    # pd.set_option('max_column', None)
    df = pd.read_csv( r'C:\Users\sam\Desktop\pharma\data\train_store.csv', engine = 'python')

    X, y, col_name = pre_processing(df)
    
    axis_fs = 18 #fontsize
    title_fs = 22 #fontsize
    # sns.set(style="whitegrid")
    
    y_test, y_train, X_test, X_train = train_test_split(y, X, test_size=0.80, shuffle=False)
    # print ("Training and testing split was successful.")

    with mlflow.start_run():

        # creating a pipeline
        model_pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('model', RandomForestRegressor(n_estimators = 10, max_depth=5))])
        model_pipeline.fit(X_train, y_train)
        # lr = LogisticRegression()
        # lr.fit(X_train, y_train)
        # train_score = model_pipeline.score(X_train, y_train)
        # test_score = model_pipeline.score(X_train_test, y_train_test)

        

        predicted_qualities = model_pipeline.predict(X_test)
        # acc_sco = accuracy_score(y_test, predicted_qualities)


        (rmse, mae, mse) = eval_metrics(y_test, predicted_qualities)

        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  mse: %s" % mse)
        with open("metrics.txt", 'w') as outfile:
            outfile.write("Root mean squared error: %2.1f%%\n" % rmse)
            outfile.write("mean apsolute error: %2.1f%%\n" % mae)
            outfile.write("mean squared error: %2.1f%%\n" % mse)


        # mlflow.log_metric("train_score", train_score)
        # mlflow.log_metric("acc_sco", train_score)
        # mlflow.log_metric("test_score", test_score)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        # mlflow.sklearn.log_model(lr, "model")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model_pipeline, "model", registered_model_name="RandomRegressorTime")
        else:
            mlflow.sklearn.log_model(model_pipeline, "model")

        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

        # y_pred = model_pipeline.predict(X_test) + np.random.normal(0,0.25,len(y_test))
        # y_jitter = y_test + np.random.normal(0,0.25,len(y_test))
        # res_df = pd.DataFrame(list(zip(y_jitter,y_pred)), columns = ["true","pred"])

        # ax = sns.scatterplot(x="true", y="pred",data=res_df)
        # ax.set_aspect('equal')
        # ax.set_xlabel('True predictions',fontsize = axis_fs) 
        # ax.set_ylabel('Predicted predictions', fontsize = axis_fs)#ylabel
        # ax.set_title('Residuals', fontsize = title_fs)

        # # Make it pretty- square aspect ratio
        # ax.plot([1, 10], [1, 10], 'black', linewidth=1)
        # plt.ylim((2.5,8.5))
        # plt.xlim((2.5,8.5))

        # plt.tight_layout()
        # plt.savefig("residuals_for_logesticregression.png",dpi=120)

    # with mlflow.start_run():
    #     model_pipeline = Pipeline(steps=[('scaler', MinMaxScaler()), ('model', DecisionTreeClassifier(criterion = 'entropy'))])
    #     model_pipeline.fit(X_train, y_train)
    #     # lr = LogisticRegression()
    #     # lr.fit(X_train, y_train)
    #     train_score = model_pipeline.score(X_train, y_train)
    #     test_score = model_pipeline.score(X_test, y_test)

    #     with open("metrics2.txt", 'w') as outfile:
    #         outfile.write("Training variance explained: %2.1f%%\n" % train_score)
    #         outfile.write("Test variance explained: %2.1f%%\n" % test_score)

    #     predicted_qualities = model_pipeline.predict(y_pred)
    #     acc_sco = accuracy_score(y_test, predicted_qualities)


    #     (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

    #     print("  RMSE: %s" % rmse)
    #     print("  MAE: %s" % mae)
    #     print("  R2: %s" % r2)

    #     mlflow.log_param("cretrion", 'entropy')
    #     mlflow.log_metric("train_score", train_score)
    #     mlflow.log_metric("acc_sco", train_score)
    #     mlflow.log_metric("test_score", test_score)
    #     mlflow.log_metric("rmse", rmse)
    #     mlflow.log_metric("r2", r2)
    #     mlflow.log_metric("mae", mae)
    #     # mlflow.sklearn.log_model(lr, "model")

    #     tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    #     # Model registry does not work with file store
    #     if tracking_url_type_store != "file":

    #         # Register the model
    #         # There are other ways to use the Model Registry, which depends on the use case,
    #         # please refer to the doc for more information:
    #         # https://mlflow.org/docs/latest/model-registry.html#api-workflow
    #         mlflow.sklearn.log_model(model_pipeline, "model", registered_model_name="RandomRegressorModel")
    #     else:
    #         mlflow.sklearn.log_model(model_pipeline, "model")

    #     print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

    #     y_pred = model_pipeline.predict(X_test) + np.random.normal(0,0.25,len(y_test))
    #     y_jitter = y_test + np.random.normal(0,0.25,len(y_test))
    #     res_df = pd.DataFrame(list(zip(y_jitter,y_pred)), columns = ["true","pred"])

    #     ax = sns.scatterplot(x="true", y="pred",data=res_df)
    #     ax.set_aspect('equal')
    #     ax.set_xlabel('True predictions',fontsize = axis_fs) 
    #     ax.set_ylabel('Predicted predictions', fontsize = axis_fs)#ylabel
    #     ax.set_title('Residuals', fontsize = title_fs)

    #     # Make it pretty- square aspect ratio
    #     ax.plot([1, 10], [1, 10], 'black', linewidth=1)
    #     plt.ylim((2.5,8.5))
    #     plt.xlim((2.5,8.5))

    #     plt.tight_layout()
    #     plt.savefig("residuals_for_decisiontree.png",dpi=120)

        

    # with mlflow.start_run():

    #     model_pipeline = Pipeline(steps=[('scaler', MinMaxScaler()), ('model', RandomForestClassifier(n_estimators = 10, criterion = 'entropy'))])
    #     model_pipeline.fit(X_train, y_train)
    #     # lr = LogisticRegression()
    #     # lr.fit(X_train, y_train)
    #     train_score = model_pipeline.score(X_train, y_train)
    #     test_score = model_pipeline.score(X_test, y_test)

    #     with open("metrics3.txt", 'w') as outfile:
    #         outfile.write("Training variance explained: %2.1f%%\n" % train_score)
    #         outfile.write("Test variance explained: %2.1f%%\n" % test_score)

    #     predicted_qualities = model_pipeline.predict(X_val)
    #     acc_sco = accuracy_score(y_val, predicted_qualities)


    #     (rmse, mae, r2) = eval_metrics(y_val, predicted_qualities)

    #     print("  RMSE: %s" % rmse)
    #     print("  MAE: %s" % mae)
    #     print("  R2: %s" % r2)

    #     mlflow.log_param("n_estimators", 10)
    #     mlflow.log_param("cretrion", 'entropy')
    #     mlflow.log_metric("train_score", train_score)
    #     mlflow.log_metric("acc_sco", train_score)
    #     mlflow.log_metric("test_score", test_score)
    #     mlflow.log_metric("rmse", rmse)
    #     mlflow.log_metric("r2", r2)
    #     mlflow.log_metric("mae", mae)
    #     # mlflow.sklearn.log_model(lr, "model")

    #     tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    #     # Model registry does not work with file store
    #     if tracking_url_type_store != "file":

    #         # Register the model
    #         # There are other ways to use the Model Registry, which depends on the use case,
    #         # please refer to the doc for more information:
    #         # https://mlflow.org/docs/latest/model-registry.html#api-workflow
    #         mlflow.sklearn.log_model(model_pipeline, "model", registered_model_name="randomforestAB")
    #     else:
    #         mlflow.sklearn.log_model(model_pipeline, "model")

    #     print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

    #     y_pred = model_pipeline.predict(X_test) + np.random.normal(0,0.25,len(y_test))
    #     y_jitter = y_test + np.random.normal(0,0.25,len(y_test))
    #     res_df = pd.DataFrame(list(zip(y_jitter,y_pred)), columns = ["true","pred"])

    #     ax = sns.scatterplot(x="true", y="pred",data=res_df)
    #     ax.set_aspect('equal')
    #     ax.set_xlabel('True predictions',fontsize = axis_fs) 
    #     ax.set_ylabel('Predicted predictions', fontsize = axis_fs)#ylabel
    #     ax.set_title('Residuals', fontsize = title_fs)

    #     # Make it pretty- square aspect ratio
    #     ax.plot([1, 10], [1, 10], 'black', linewidth=1)
    #     plt.ylim((2.5,8.5))
    #     plt.xlim((2.5,8.5))

    #     plt.tight_layout()
    #     plt.savefig("residuals_for_randomforest.png",dpi=120) 
