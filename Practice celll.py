import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
filename = 'RNN_ML_Ready_Data.mat'
with h5py.File(filename, 'r') as f:
    X_ref = f['data_train']['X'][0][0]
    Y_ref = f['data_train']['Y'][0][0]
    X = f[X_ref][:]
    Y = f[Y_ref][:].ravel()  # Flatten Y

# Plot Y target signal
plt.plot(Y, label='Target Y')
plt.title("Y over Time")
plt.xlabel("Time Step")
plt.ylabel("Target Value")
plt.grid(True)
plt.legend()
plt.show()

# Plot all 9 features over full length
plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.plot(X[:, i])
    plt.title(f'Feature {i}')
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True)
plt.tight_layout()
plt.show()

# Smoothing Function
def smooth_signal(signal, window_size=10):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

# Apply smoothing
X_smooth = np.zeros_like(X)
for i in range(9):
    X_smooth[:, i] = smooth_signal(X[:, i])

# Plot smoothed features
plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.plot(X_smooth[:, i])
    plt.title(f'Smoothed Feature {i}')
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True)
plt.tight_layout()
plt.show()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Print accuracy (R^2 score)
print("Accuracy (R^2):", model.score(X_test, y_test))

# Predict
y_pred = model.predict(X_test)

# Plot full Actual vs Predicted
plt.figure(figsize=(14, 5))
plt.plot(y_test, label='Actual', color='green', linestyle='--', linewidth=1.5)
plt.plot(y_pred, label='Predicted', color='blue', alpha=0.8)
plt.title("Actual vs Predicted Output (Full Set)")
plt.xlabel("Sample Index")
plt.ylabel("Output Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
