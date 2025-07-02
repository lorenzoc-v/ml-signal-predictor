# Time-Series Signal Prediction using ML

This project demonstrates how machine learning can be used to analyze and predict time-series sensor data.

📊 **Overview**  
The goal of this project is to take a dataset of smoothed signal features and train both a **Feedforward Neural Network (FNN)** and **Linear Regression** model to predict a target output variable.

🛠 **Key Features**
- Loads and processes `.mat` time-series data using `scipy.io.loadmat`
- Visualizes patterns in high-dimensional sensor data
- Trains and evaluates two models: a custom FNN and baseline linear regression
- Compares prediction performance visually and numerically

📁 **Files in This Repo**
| File | Description |
|------|-------------|
| `FNN.py` | Core feedforward neural network model |
| `importingdata.py` | Script to load `.mat` data and print structure |
| `notebook.ipynb` | Jupyter notebook for training, plotting, and analysis |
| `Practice celll.py` | First practice script for exploring data |
| `Second Practice Cell.py` | Second script for testing model logic |
| `README.md` | This file — project explanation |
| `.gitignore` | Prevents large `.mat` files from being uploaded |

📦 **How to Restore the `.mat` File**
GitHub doesn’t allow uploads over 100MB, so the `.mat` dataset (`RNN_ML_Ready_Data.mat`) was removed. To recreate it:

```python
from scipy.io import savemat
import numpy as np

data = {
    'X': np.random.rand(100, 10),  # Feature data
    'Y': np.random.rand(100, 1)    # Target values
}
savemat('RNN_ML_Ready_Data.mat', data)
