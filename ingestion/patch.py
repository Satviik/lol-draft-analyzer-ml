import pandas as pd

df = pd.read_csv("data/processed/draft_dataset.csv")
print(df["patch"].value_counts().head(15))
