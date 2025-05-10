import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load and preprocess the data
df = pd.read_csv("synthetic_energy_data.csv")
energy = df["energy"].values.reshape(-1, 1)

scaler = MinMaxScaler()
energy_scaled = scaler.fit_transform(energy)

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window_size = 24
X, y = create_sequences(energy_scaled, window_size)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Split
train_size = int(len(X_tensor) * 0.8)
X_test = X_tensor[train_size:]
y_test = y_tensor[train_size:]

# Define model again (must match structure from training)
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Load model
model = LSTMModel()
model.load_state_dict(torch.load("lstm_model.pth"))
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(X_test).numpy()
    targets = y_test.numpy()

# Inverse transform to original scale
predicted_energy = scaler.inverse_transform(predictions)
actual_energy = scaler.inverse_transform(targets)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(actual_energy, label="Actual")
plt.plot(predicted_energy, label="Predicted")
plt.title("LSTM Energy Usage Prediction")
plt.xlabel("Time Step")
plt.ylabel("Energy Usage")
plt.legend()
plt.tight_layout()
plt.savefig("LSTM_Predictions.png")
plt.show()

