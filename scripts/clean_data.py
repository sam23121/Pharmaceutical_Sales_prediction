import pandas as pd
import numpy as np

class CleanData:
    def __init__(self):
        pass
    #Function to drop columns with zero values
    def drop_rows(self, df,col1,col2):
        df_new= df.drop(df[(df[col1] == 0) & (df[col2] == 0)].index)
        return df_new
    # Calculating skewness of each column 
    def calculating_skewness(self,df):
        df.skew(axis='index', skipna=True)
   #Function to drop columns missing descriptive data that is unpredictable
    def drop_missing(self,df,col):
        df_dropped= df.dropna(subset=[col])
        return df_dropped.shape
        
    # Functions to backward fill and forward fill columns  
    def fix_missing_ffill(self,df, col):
        df[col] = df[col].fillna(method='ffill',axis = 1)
        return df[col]


    def fix_missing_bfill(self,df, col):
        df[col] = df[col].fillna(method='bfill')
        return df[col]
    # Functions to fill missing columns with mean, median, mode
    #using median
    def fix_missing_median(self,df, col):
        df_Med=df[col].fillna(df[col].median(), inplace=True)
        return df_Med
  
    # Using mean
    def fix_missing_mean(self,df, col):
        df_mean=df[col].fillna(int(df[col].mean()), inplace=True)
        return df_mean
  
    # Using mode
    def fix_missing_mode(self,df, col):
        df_mode=df[col].fillna(int(df[col].mode()), inplace=True)
        return df_mode
    # By Interpolation
    def fix_missing_interpolation_fffil(self,df):
        df_clean = df_dropped.interpolate(method='ffill')
        return df_clean
    def fix_missing_interpolation_bffil(self,df):  
        df_clean = df_dropped.interpolate(method='bfill')
        return df_clean

    
    def detect_outliers_zscore(data):
        outliers = []
        thres = 3
        mean = np.mean(data)
        std = np.std(data)
        # print(mean, std)
        for i in data:
            z_score = (i-mean)/std
            if (np.abs(z_score) > thres):
                outliers.append(i)
        return outliers# Driver code

