# Time-Series Signal Prediction using ML

This project loads and processes `.mat` data containing 9 time-series features. It performs signal smoothing, feature visualization, and trains both a linear regression and a feedforward neural network (FNN) to predict target output values.

## 📁 Features
- `.mat` loader using h5py
- 9-feature signal smoothing using moving average
- Visualization with matplotlib subplots
- Linear regression (scikit-learn)
- Feedforward neural net (PyTorch)
- Actual vs Predicted graph comparison

## 🔧 Technologies
- Python
- NumPy, h5py, matplotlib
- scikit-learn
- PyTorch

## 📈 Results
![Prediction Plot](plots/predictions.png)

## 💡 Run It Yourself
1. Clone this repo
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Run:  
   `python main.py`

> Dataset not included due to size. DM me for access or swap in your own `.mat` time-series.
