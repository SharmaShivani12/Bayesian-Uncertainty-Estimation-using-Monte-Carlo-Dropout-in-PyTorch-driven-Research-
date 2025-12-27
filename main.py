import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Simple Bayesian Dropout Network Example

class BayesianNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.dropout = nn.Dropout(p=0.3)  # dropout enables uncertainty
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # active during inference for Monte Carlo sampling
        return torch.sigmoid(self.fc2(x))

# Generate random dataset
X = torch.randn(500, 20)
y = (torch.rand(500) > 0.5).float().unsqueeze(1)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = BayesianNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training loop
for epoch in range(5):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
print("Training Completed")

# Bayesian inference via Monte Carlo sampling for uncertainty
def predict_with_uncertainty(x, n_samples=20):
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(x).numpy())
    return np.mean(preds, axis=0), np.std(preds, axis=0)

x_sample = X[0:1]
mean, std = predict_with_uncertainty(x_sample)

print("Prediction Mean:", mean)
print("Uncertainty (Std):", std)
