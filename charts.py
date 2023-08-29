import finplot as fplt
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    #candlestickChart("AAPL", '2020-01-02', '2022-12-31', "5d")
    #Possible intervals by the yfinance library: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    #This determines the number of days shown by each candlestick

    boxplotChart("AAPL", '2020-01-02', '2022-12-31', "1wk")

def candlestickChart(ticker, startDate=None, endDate=None, numberOfDaysInterval="1d"):
    data = yf.download(ticker, startDate, endDate, interval=numberOfDaysInterval)
    fplt.candlestick_ochl(data[['Open', 'Close', 'High', 'Low']])
    fplt.show()

def boxplotChart(ticker, startDate=None, endDate=None, numberOfDaysInterval="1d"):
    data = yf.download(ticker, startDate, endDate, interval=numberOfDaysInterval)

    data = data[['Open', 'Close', 'High', 'Low']]
    data = np.array(data)

    dimensions = data.shape
    rows = dimensions[0]
    columns = dimensions[1]

    data = data.reshape(columns, rows)

    figure = plt.figure(figsize=(15, 8))
    plt.boxplot(data)
    plt.show()


if __name__ == "__main__":
    main()