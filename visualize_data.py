import pandas as pd

df = pd.read_pickle("datasets/ph2_dataset.pkl")
print(df.head())
print(df["Clinical Diagnosis"].value_counts())