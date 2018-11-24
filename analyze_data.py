import os
import pandas as pd
import numpy as np
from DataTools.pickle import save_to_pickle, load_from_pickle

import matplotlib.pyplot as plt
import seaborn as sns

if __name__=='__main__':
    series = load_from_pickle('data','data_369.pkl')
    vals = series.values

    # remove zeros before ts data
    idx = np.argmax(vals > 0)
    vals = vals[idx:]

    # smooth with moving average filter
    mva = pd.rolling_mean(vals, 500)

    # choose raw or pre-smoothed data
    data = vals

    # split dataset
    split_idx = 80000
    window = 150
    train = data[:split_idx]
    test = data[split_idx:]
    test_window = test[:window]

    # statsmodels autoregression
    run_ar = False
    if run_ar:
        from statsmodels.tsa.ar_model import AR
        from sklearn.metrics import mean_squared_error

        model = AR(train)
        model_fit = model.fit()
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)

        y_hat = model_fit.predict(start=len(train), end=len(train)+len(test_window)-1, dynamic=False)
        error = mean_squared_error(test_window, y_hat)

        plt.plot(test, 'k', label='test data')
        plt.plot(test_window, 'gray', label='test window')
        plt.plot(y_hat, 'r', label='predictions')
        plt.legend()
        plt.title('statsmodels ar_model on power use timeseries')
        plt.show()

    # arima model AutoRegressive Integrated Moving Average
    run_arima = True
    if run_arima:
        from statsmodels.tsa.arima_model import ARIMA

        arima = ARIMA(train, order=(5,1,5))
        arima_fit = arima.fit(disp=0)
        print(arima_fit.summary())

        # plot residuals, check for systematic bias
        residuals = arima_fit.resid

        fig, ax = plt.subplots(1,2)
        ax[0].plot(residuals)
        ax[0].set_title('residuals')
        sns.kdeplot(residuals, ax=ax[1])
        ax[1].set_title('residuals histogram')
        ax[1].axvline(np.mean(residuals), 0, 10, color='r',label='mean')
        ax[1].legend()
        plt.show()

        #print(residuals.describe()) this only works if 'residuals' is a dataframe

        y_hat = arima_fit.predict(start=len(train), end=len(train)+len(test_window)-1, dynamic=False)

        plt.plot(test, 'k', label='test data')
        plt.plot(test_window, 'gray', label='test window')
        plt.plot(y_hat, 'r', label='predictions')
        plt.legend()
        plt.title('statsmodels ar_model on power use timeseries')
        plt.show()
