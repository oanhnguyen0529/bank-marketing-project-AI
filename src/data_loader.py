import pandas as pd

def load_raw_data(path="data/raw/bank-additional-full.csv"):
    df = pd.read_csv(path, sep=';')
    return df

def get_feature_info():
    return {
        "numeric": ["age","duration","campaign","pdays","previous",
                    "emp.var.rate","cons.price.idx","cons.conf.idx",
                    "euribor3m","nr.employed"],
        "categorical": ["job","marital","education","default",
                        "housing","loan","contact","month",
                        "day_of_week","poutcome"],
        "target": "y"
    }
