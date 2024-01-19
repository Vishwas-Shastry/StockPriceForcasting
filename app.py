
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import yfinance as yf
from keras.models import load_model


# Function to fetch historical stock data
def get_historical_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

ticker_symbol = "AAPL"  
start_date = "2020-01-01"
end_date = "2023-01-01"
historical_data = get_historical_data(ticker_symbol, start_date, end_date)

# Load the trained models and scaler
linear_model = joblib.load('linear_model.joblib')
arima_model = joblib.load('arima_model.pkl')  
lstm_model = load_model('lstm_model.h5')  
scaler = joblib.load('scaler.joblib')

# Streamlit UI
st.title("Stock Price Prediction")

# User input for prediction
selected_date = st.date_input("Select Date", datetime(2022, 1, 15))
selected_date_datetime = pd.to_datetime(selected_date)
day = selected_date.day
month = selected_date.month
year = selected_date.year

# Radio buttons for model selection
selected_model = st.radio("Select Model", ['LR', 'ARIMA', 'LSTM', 'ALL'])

day = selected_date.day
month = selected_date.month
year = selected_date.year

#predictions
input_data = pd.DataFrame({'Day': [day], 'Month': [month], 'Year': [year]})
# Predict using Linear Regression model
linear_prediction = linear_model.predict(input_data)[0]

# Predict using ARIMA model
arima_input = pd.Series([linear_prediction], name='predicted_close_price')
arima_prediction_value = arima_model.forecast(steps=1, exog=arima_input)
arima_prediction = arima_prediction_value.values[0]

# Predict using LSTM model
scaled_input = scaler.transform(input_data)
lstm_input = scaled_input.reshape((scaled_input.shape[0], 1, scaled_input.shape[1]))
lstm_prediction = lstm_model.predict(lstm_input)[0][0]
# Modify this line to provide the correct shape for inverse_transform
lstm_prediction = scaler.inverse_transform([[0, 0, lstm_prediction]])[0][2]
lstm_prediction = lstm_prediction / 10


def plot_data(selected_date, prediction, historical_data):
    fig, ax = plt.subplots()
    ax.plot(historical_data.index, historical_data['Close'], label='Historical Data')
    ax.scatter(selected_date, prediction, color='r', label='Predicted Value', zorder=5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.set_title('Stock Data and Prediction')
    ax.legend()


    # Save the figure to BytesIO buffer
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    # Display the plot in Streamlit
    st.image(buffer, caption='Stock Data and Prediction Plot', use_column_width=True)

# Make prediction button
if st.button("Make Prediction"):
    input_data = pd.DataFrame({'Day': [day], 'Month': [month], 'Year': [year]})
    if selected_model == 'LR':
        # Make predictions for Linear Regression model
        st.success(f"Linear Regression Predicted Close Price: {linear_prediction:.2f}")
        plot_data(selected_date, linear_prediction, historical_data)

    elif selected_model == 'ARIMA':
        # Make predictions for ARIMA model
        st.success(f"ARIMA Predicted Close Price: {arima_prediction:.2f}")
        plot_data(selected_date, arima_prediction, historical_data)

    elif selected_model == 'LSTM':
        # Make predictions for LSTM model
        st.success(f"LSTM Predicted Close Price: {lstm_prediction:.2f}")
        plot_data(selected_date, lstm_prediction, historical_data)

    elif selected_model == 'ALL':
        # Make predictions for all models and Aggregate predictions
        final_prediction = np.mean([linear_prediction, arima_prediction, lstm_prediction])
        st.success(f"Aggregated Predicted Close Price: {final_prediction:.2f}")
        plot_data(selected_date, final_prediction, historical_data)
