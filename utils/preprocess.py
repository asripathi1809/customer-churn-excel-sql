import pandas as pd
import joblib

def preprocess_input(df):
    model_columns = joblib.load('models/model_columns.pkl')

    df = pd.get_dummies(df)
    df = df.reindex(columns=model_columns, fill_value=0)

    return df
