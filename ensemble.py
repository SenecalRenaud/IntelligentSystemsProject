
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf
import pmdarima as pm
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, RNN, GRU, InputLayer, SimpleRNN
from sklearn.ensemble import RandomForestRegressor

from stock_prediction import scaled_data

def main():
    ensembleTesting("AAPL", '2015-01-01', '2020-01-01')

if __name__ == "__main__":
    main()

def ensembleTesting(ticker, startDate=None, endDate=None, testToTrainRatio=0.2, storeDataLocally=False, splitByDate=True, scale=False):
    if isinstance(ticker, str):
        # load data from yahoo finance library
        data = yf.download(ticker, startDate, endDate, progress=False)
    elif isinstance(ticker, pd.DataFrame):
        # if data already loaded
        data = ticker
    else:
        raise TypeError("ticker is not a String or DataFrame")

    result = {}

    if storeDataLocally and not isinstance(ticker, pd.DataFrame):
        if startDate is not None and endDate is not None:
            data.to_csv(ticker+':'+startDate+"to"+endDate+".csv")
            result["dataFileName"] = ticker + ':' + startDate + "to" + endDate + ".csv"
        else:
            data.to_csv(ticker+".csv")
            result["dataFileName"] = ticker + ".csv"

    # lookback_days is the number of days to look in the past to predict the next day
    # lookup_days is the number of days to predict with n lookback days
    # feature_columns are the columns that are included when downloading data from yfinance
    lookback_days = 50
    feature_columns = ['adjclose', 'volume', 'open', 'high', 'low']

    # add "future" column in the data to represent the next days' closing price
    scaled_data['future'] = scaled_data['adjclose'].shift(-lookback_days)

    #unscaled data that will be used for ARIMA/SARIMA/Random Forest
    unscaled_data = data.copy()

    if scale:
        column_scaler = {}
        # scale the data from 0 to 1
        for column in feature_columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            data[column] = scaler.fit_transform(np.expand_dims(data[column].values, axis=1))
            column_scaler[column] = scaler
        result["scalers"] = column_scaler
        # storing scaled columns in returned dict

    # remove NaNs
    # inplace true to modify the data directly
    data.dropna(inplace=True)
    unscaled_data.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=lookback_days)
    # double ended queue with lookback days as max length

    for entry, target in zip(data[feature_columns].values, data['future'].values):
        sequences.append(entry)
        if len(sequences) == lookback_days:
            sequence_data.append([np.array(sequences), target])
    # sequence_data is now filled with the data from the past 50 days and the 51st day price

    x_train = []
    y_train = []
    unscaled_x_train = []
    unscaled_y_train = []

    # transferring the training data into x_train and y_train
    for seq, target in scaled_data:
        x_train.append(seq)
        y_train.append(target)

    # transferring the training data into unscaled_x_train and unscaled_y_train
    for seq, target in unscaled_data:
         unscaled_x_train.append(seq)
         unscaled_y_train.append(target)

    # Convert into an array
    x_train, y_train = np.array(x_train), np.array(y_train)
    unscaled_x_train, unscaled_y_train = np.array(unscaled_x_train), np.array(unscaled_y_train)

    # split the dataset into training & testing sets by date or randomly depending on arguments
    # also taking into account the testToTrainRatio
    if splitByDate:
        train_samples = int((1 - testToTrainRatio) * len(x_train))
        x_train = x_train[:train_samples]
        y_train = y_train[:train_samples]
        x_test = x_train[train_samples:]
        y_test = y_train[train_samples:]

        unscaled_x_train = unscaled_x_train[:train_samples]
        unscaled_y_train = unscaled_y_train[:train_samples]
        unscaled_x_test = unscaled_x_train[train_samples:]
        unscaled_y_test = unscaled_y_train[train_samples:]
    else:
        x_train, x_test, y_train, y_test = train_test_split(data, test_size=(1-testToTrainRatio))
        unscaled_x_train, unscaled_x_test, unscaled_y_train, unscaled_y_test = train_test_split(unscaled_data, test_size=(1 - testToTrainRatio))

    x_train = x_train[:, :, :len(feature_columns)].astype(np.float32)
    x_test = x_test[:, :, :len(feature_columns)].astype(np.float32)
    #reformat x_train and x_test into floating point numbers

    # ------------------------------------------------------------------------------
    # Building and training the model
    # ------------------------------------------------------------------------------

    # ------------------------------------
    # SARIMA
    # ------------------------------------

    #Automatically choose the appropriate SARIMA parameters with the data
    #SARIMA data does not need to be scaled
    #Seasonal is set to true because SARIMA is seasonal
    sarima_prediction = pm.auto_arima(unscaled_y_train, seasonal=True, trace=True,
                                 error_action='ignore',  # don't want to know if an order does not work
                                 suppress_warnings=True,  # don't want convergence warnings
                                 stepwise=True, m=12)  # set to stepwise

    # ------------------------------------
    # ARIMA
    # ------------------------------------

    # Automatically choose the appropriate ARIMA parameters with the data
    # ARIMA data does not need to be scaled
    #Seasonal is set to false because ARIMA is NOT seasonal
    arima_prediction = pm.auto_arima(unscaled_y_train, seasonal=False, trace=True,
                                 error_action='ignore',  # don't want to know if an order does not work
                                 suppress_warnings=True,  # don't want convergence warnings
                                 stepwise=True, m=12)  # set to stepwise


    # ------------------------------------
    # LSTM
    # ------------------------------------

    lstm = Sequential()

    # Adding layers to the model
    lstm.add(LSTM(units=50, return_sequences=True, batch_input_shape=(None, lookback_days, len(feature_columns))))
    lstm.add(Dropout(0.2))
    lstm.add(LSTM(units=50, return_sequences=True))
    lstm.add(Dropout(0.2))
    lstm.add(LSTM(units=50))
    lstm.add(Dropout(0.2))

    lstm.add(Dense(units=1))

    lstm.compile(optimizer='adam', loss='mean_squared_error')

    lstm.fit(x_train, y_train, epochs=25, batch_size=32)

    lstm_prediction = lstm.predict(x_test)

    # ------------------------------------
    # GRU
    # ------------------------------------

    gru = Sequential()

    # Adding layers to the model
    gru.add(GRU(units=50, return_sequences=True, batch_input_shape=(None, lookback_days, len(feature_columns))))
    gru.add(Dropout(0.2))
    gru.add(GRU(units=50, return_sequences=True))
    gru.add(Dropout(0.2))
    gru.add(GRU(units=50))
    gru.add(Dropout(0.2))

    gru.add(Dense(units=1))

    gru.compile(optimizer='adam', loss='mean_squared_error')

    gru.fit(x_train, y_train, epochs=25, batch_size=32)

    gru_prediction = gru.predict(x_test)

    # ------------------------------------
    # Random Forest
    # ------------------------------------

    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(unscaled_x_train, unscaled_y_train)
    rf_prediction = rf.predict(unscaled_x_test)

    # ------------------------------------
    # RNN
    # ------------------------------------

    rnn = Sequential()

    # Adding layers to the model
    rnn.add(SimpleRNN(units=50, return_sequences=True, batch_input_shape=(None, lookback_days, len(feature_columns))))
    rnn.add(Dropout(0.2))
    rnn.add(SimpleRNN(units=50, return_sequences=True))
    rnn.add(Dropout(0.2))
    rnn.add(SimpleRNN(units=50))
    rnn.add(Dropout(0.2))

    rnn.add(Dense(units=1))

    rnn.compile(optimizer='adam', loss='mean_squared_error')

    rnn.fit(x_train, y_train, epochs=25, batch_size=32)

    rnn_prediction = rnn.predict(x_test)


    # ------------------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------------------

    if scale:
        y_test = np.squeeze(result["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        lstm_prediction = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(lstm_prediction))
    # Reverse the scaling

    #simple ensemble method of averaging
    ensemble = np.add(rf_prediction, lstm_prediction)/2

    result["predicted_prices"] = ensemble.copy()
    result["actual_prices"] = y_test.copy()

    plt.plot(unscaled_y_test, color="black", label=f"Actual Price")
    plt.plot(ensemble, color="green", label=f"Predicted Price")
    plt.title(f" Share Price")
    plt.xlabel("Time")
    plt.ylabel(f" Share Price")
    plt.legend()
    plt.show()

    return result