import pandas as pd

# Load the synthetic dataset
df = pd.read_csv("synthetic_energy_data.csv", parse_dates=["timestamp"])

print(df.head())
print(df['anomaly'].value_counts())  # to check class balance
