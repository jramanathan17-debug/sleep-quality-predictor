import pandas as pd

data = pd.read_csv("dataset/sleep_data.csv")

print("Dataset loaded successfully!")
print("\nColumns:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)
print("\nQuality of Sleep values:")
print(sorted(data["Quality of Sleep"].unique()))