import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file

# For visualization
import matplotlib.pyplot as plt

def normalization(x):
    result = (x-x.min())/(x.max()-x.min())
    return result

def standardization(x):
    result = (x-x.mean())/x.var()
    return result

def dataset_preprocess(csv_file_path):
    pd.set_option('display.max_rows', None)
    df_ts = pd.read_csv(csv_file_path)
    drop_list = ['CustomerId']
    df_ts = df_ts.drop(drop_list, axis=1)

    non_normalization_list = ['CreditLevel']
    countries = df_ts["Geography"].unique()
    geo = pd.get_dummies(df_ts["Geography"])
    df_ts.drop(["Geography"], axis=1, inplace=True)
    df_ts = pd.concat([df_ts, geo], axis=1)  # , join="inner"
    df_creditlevel = df_ts.pop('CreditLevel')  # remove column of label and store it in df1
    df_ts['CreditLevel'] = df_creditlevel

    columns = list(df_ts.columns)
    for column in columns:
        if column not in non_normalization_list:
            df_ts[column] = normalization(df_ts[column])

    for column in columns:
        if column not in non_normalization_list:
            df_ts[column] = standardization(df_ts[column])

    return df_ts


