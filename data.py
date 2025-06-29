import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo

data = fetch_ucirepo(id=17)
x = data.data.features
y = data.data.targets.replace({'B': 0, 'M': 1})


df = x.copy()
df["Diagnosis"] = y

df.to_csv("data/veri.csv", index=False)
# df_loaded = pd.read_csv("data/veri.csv")


# | Etiket | Açılımı       | Anlamı                             |
# | ------ | ------------- | ---------------------------------- |
# | **B**0  | **Benign**    | **İyi huylu tümör** (kanser değil) |
# | **M**1  | **Malignant** | **Kötü huylu tümör** (kanserli)    |
