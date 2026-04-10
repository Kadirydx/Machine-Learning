import pandas as pd

# load the dataset from local environment
df = pd.read_csv("data/cbis-ddsm-r-dataset.csv")

print(df.head())
print(df.info())
print(df.shape)
print(df.columns())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())




