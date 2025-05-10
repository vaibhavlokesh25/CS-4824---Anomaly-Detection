import torch
import torch.nn as nn
import numpy as np
from lstm_preprocessing import train_loader, test_loader

# Define LSTM model
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
        out = self.fc(out[:, -1, :])  # last time step
        return out

# Training loop
def train_model(model, train_loader, num_epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# Evaluation
def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        predictions = []
        targets = []
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            predictions.append(output.numpy())
            targets.append(y_batch.numpy())
        
        predictions = np.concatenate(predictions).flatten()
        targets = np.concatenate(targets).flatten()
        mse = np.mean((predictions - targets) ** 2)
        print(f"Test MSE: {mse:.4f}")

# Main
if __name__ == "__main__":
    model = LSTMModel()
    train_model(model, train_loader, num_epochs=20, lr=0.001)
    evaluate_model(model, test_loader)
    torch.save(model.state_dict(), "lstm_model.pth")

