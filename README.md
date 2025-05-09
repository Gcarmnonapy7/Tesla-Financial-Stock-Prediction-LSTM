# Time Series Forecasting with PyTorch

This project focuses on forecasting financial time series data using PyTorch. It uses a sliding window approach to transform the data and prepares it for training deep learning models such as LSTM or GRU.

## Features

- Data normalization with `StandardScaler`
- Windowing technique for time series sequences
- Train-test split (80/20)
- Tensor conversion for model input
- GPU-ready with `.to(device)`

## Technologies Used

- Python
- Pandas
- NumPy
- PyTorch
- Scikit-learn
