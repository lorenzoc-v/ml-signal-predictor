import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data from .mat file
filename = 'RNN_ML_Ready_Data.mat'
with h5py.File(filename, 'r') as f:
    X_ref = f['data_train']['X'][0][0]
    Y_ref = f['data_train']['Y'][0][0]
    X = f[X_ref][:]
    Y = f[Y_ref][:].ravel()

# Smooth signals
def smooth_signal(signal, window_size=10):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

X_smooth = np.zeros_like(X)
for i in range(X.shape[1]):
    X_smooth[:, i] = smooth_signal(X[:, i])

# Normalize inputs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_smooth)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define Feedforward Neural Network
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = FeedforwardNN(X_train.shape[1])
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# Evaluate
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor).numpy()

# Plot predictions vs actual
plt.figure(figsize=(14, 5))
plt.plot(y_test[:500], label='Actual', color='green', linestyle='--')
plt.plot(y_pred_test[:500], label='Predicted', color='blue')
plt.title("Feedforward NN: Actual vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel("Output Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
