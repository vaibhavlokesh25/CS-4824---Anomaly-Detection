import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Load the synthetic dataset
df = pd.read_csv("synthetic_energy_data.csv", parse_dates=["timestamp"])

# Select numeric columns for anomaly detection
features = ['temperature', 'humidity', 'occupancy', 'energy']

# Compute Z-scores
z_scores = np.abs(zscore(df[features]))

# Set threshold (e.g., z > 3 indicates anomaly)
threshold = 3
anomaly_flags = (z_scores > threshold).any(axis=1)

# Add to DataFrame
df['statistical_anomaly'] = anomaly_flags.astype(int)

# Print anomaly stats
print("Total detected anomalies:", df['statistical_anomaly'].sum())

# Plot energy consumption with anomalies
plt.figure(figsize=(12, 5))
plt.plot(df['timestamp'], df['energy'], label='Energy Usage')
plt.scatter(df[df['statistical_anomaly'] == 1]['timestamp'],
            df[df['statistical_anomaly'] == 1]['energy'],
            color='red', label='Anomaly')
plt.xlabel("Timestamp")
plt.ylabel("Energy Usage")
plt.title("Statistical Anomaly Detection on Energy Usage")
plt.legend()
plt.tight_layout()
plt.show()
