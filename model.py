# train_model.py
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import joblib
import numpy as np

# Function to fetch historical stock data
def get_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

# Function to preprocess data and train the model
def model_train(features, target):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train Linear Regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Train ARIMA model
    arima_model = ARIMA(y_train, order=(5,1,0))
    arima_model_fit = arima_model.fit()

    # Train LSTM model
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[1])))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_reshaped, y_train, epochs=25, verbose=0)

    return linear_model, arima_model_fit, lstm_model, scaler

# Function to fetch historical stock data, preprocess data, and train the models
def train_stock_prediction_model(ticker, start_date, end_date):
    # Fetch historical stock data
    df = get_stock_data(ticker, start_date, end_date)

    # Feature engineering (using just the 'Close' price as a feature for simplicity)
    df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    # Select features and target variable
    features = df[['Day', 'Month', 'Year']]
    target = df['Close']

    # Train the models
    linear_model, arima_model, lstm_model, scaler = model_train(features, target)

    # Save the trained models
    joblib.dump(linear_model, 'linear_model.joblib')
    arima_model.save('arima_model.pkl')
    lstm_model.save('lstm_model.h5')
    joblib.dump(scaler, 'scaler.joblib')

if __name__ == "__main__":
    ticker_symbol = "AAPL"  # Change to the desired stock symbol
    start_date = "2022-01-01"
    end_date = "2023-01-01"

    train_stock_prediction_model(ticker_symbol, start_date, end_date)

