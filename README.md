# Stock Price Prediction Using Machine Learning

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.0+-orange.svg" alt="TensorFlow 2.0+"/>
  <img src="https://img.shields.io/badge/Keras-2.0+-red.svg" alt="Keras 2.0+"/>
  <img src="https://img.shields.io/badge/Pandas-1.0+-green.svg" alt="Pandas 1.0+"/>
  <img src="https://img.shields.io/badge/NumPy-1.19+-yellow.svg" alt="NumPy 1.19+"/>
  <img src="https://img.shields.io/badge/Matplotlib-3.3+-purple.svg" alt="Matplotlib 3.3+"/>
</div>

## 📊 Project Overview

This project implements a deep learning-based stock price prediction model using Long Short-Term Memory (LSTM) networks. The system analyzes historical stock data to forecast future price movements, providing both short-term predictions and longer-term forecasts.

Developed as part of the Future Intern Machine Learning Internship program, this project demonstrates advanced time series analysis techniques applied to financial market data.

<div align="center">
  <img src="visualizations/AAPL Stock Price Prediction Graph.png" alt="Stock Price Prediction" width="800px"/>
</div>

## 📋 Table of Contents

- [Features](#-features)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Code Structure](#-code-structure)
- [Future Improvements](#-future-improvements)

## ✨ Features

- **Comprehensive Data Analysis**: Exploratory data analysis of historical stock prices with technical indicators
- **Feature Engineering**: Creation of relevant financial features and technical indicators
- **Deep Learning Model**: Implementation of LSTM-based neural network for time series forecasting
- **Interactive Visualizations**: Beautiful visualizations of predicted vs. actual stock prices
- **Performance Metrics**: Rigorous evaluation using RMSE, MAE, and R² score
- **Future Price Forecasting**: 30-day price forecasts with confidence intervals
- **Error Analysis**: Detailed analysis of prediction errors and model performance

## 📊 Dataset

The dataset contains historical daily prices for stocks trading on NASDAQ, retrieved from Yahoo Finance. Each CSV file includes:

- **Date**: Trading date
- **Open**: Opening price
- **High**: Maximum daily price
- **Low**: Minimum daily price
- **Close**: Closing price (adjusted for splits)
- **Adj Close**: Adjusted close price (adjusted for dividends and splits)
- **Volume**: Number of shares traded

The project uses data up to April 1, 2020, organized into folders for ETFs and stocks.

## 🧠 Model Architecture

The project implements a sequential LSTM neural network with the following architecture:

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, 60, 50)            10400     
_________________________________________________________________
dropout (Dropout)            (None, 60, 50)            0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 50)                20200     
_________________________________________________________________
dropout_1 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense (Dense)                (None, 25)                1275      
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 26        
=================================================================
Total params: 31,901
Trainable params: 31,901
Non-trainable params: 0
_________________________________________________________________
```

Key components:
- Two LSTM layers with 50 units each
- Dropout layers (0.2) to prevent overfitting
- Dense layers for final predictions
- Adam optimizer with MSE loss function

## 📈 Results

The model was evaluated on Apple (AAPL) stock data with impressive results:

| Metric | Value |
|--------|-------|
| Test RMSE | $6.81 |
| Test MAE | $3.65 |
| Test R² | $0.98 |
| Correlation | 0.98 |

### Prediction Visualization

<div align="center">
  <img src="visualizations/AAPL vs Predicted.png" alt="Prediction vs Actual Prices" width="800px"/>
</div>

The model accurately captures most of the price trends and movements, with minor deviations during high-volatility periods.

### Forecast Visualization

<div align="center">
  <img src="visualizations/AAPL 30 day Stock Price Forecast.png" alt="30-Day Price Forecast" width="800px"/>
</div>

The 30-day forecast demonstrates the model's ability to project future price movements based on historical patterns.

### Error Analysis

<div align="center">
  <img src="visualizations/AAPL Prediction Errorr over Time.png" alt="Prediction Error Analysis" width="800px"/>
</div>

Error analysis shows the model's predictions are generally within a narrow range of the actual prices, with larger errors corresponding to unexpected market events.

## 🛠️ Installation

```bash
# Clone this repository
git clone https://github.com/Redoy0/FUTURE_ML_02.git
cd FUTURE_ML_02

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## 🚀 Usage

To run the stock prediction model:

```python
# Import the main module
from stock_prediction import main

# Run the prediction on default stock (AAPL)
model, df, prediction_df, forecast_df = main()

# Or specify a different stock ticker
from stock_prediction import load_and_preprocess_stock, prepare_lstm_data, build_lstm_model, visualize_predictions

# Load data for a specific stock
ticker = 'MSFT'
df = load_and_preprocess_stock(ticker)

# Continue with model training and prediction
# ...
```

## 📁 Code Structure

```
.
├── code
|   ├── stock_prediction.ipynb       # Main module with prediction functionality
├── data/                     # Dataset directory
│   ├── stocks/               # Individual stock CSV files
│   ├── etfs/                 # Individual ETF CSV files
│   └── symbols_valid_meta.csv # Metadata for all symbols
├── visualizations/                   # Visualization outputs
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## 🔮 Future Improvements

- Add sentiment analysis from news and social media
- Implement ensemble methods combining multiple prediction algorithms
- Incorporate additional technical indicators
- Apply attention mechanisms to improve LSTM performance
- Develop a web interface for real-time predictions

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- **Future Intern** for providing this challenging and educational project opportunity
- **Yahoo Finance** for the historical stock data
- All open-source libraries that made this project possible

---

<div align="center">
  <p>Developed by Md. Sabbir Ahamed as part of the Future Intern Machine Learning Internship</p>
</div>
