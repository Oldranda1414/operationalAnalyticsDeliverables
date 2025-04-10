import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pmdarima as pm

def preprocessing(data, seasonalm):
    # log transform
    log = np.log(data)

    # diff transform
    logdiff = np.array([log[i] - log[i - 1] for i in range(1, len(log))])

    # diffm transform
    logdiffm = np.array([logdiff[i] - logdiff[i - seasonalm] for i in range(1, len(logdiff))])
    return log, logdiff, logdiffm


def reconstruct_fore(logdiff, log, forecast, seasonalm):
    # Undo seasonal differencing
    reclogdiff = [forecast[i] + logdiff[-seasonalm + i % seasonalm] for i in range(len(forecast))]

    # Undo regular differencing
    reclog = [log[-1] + reclogdiff[0]] + [0] * (len(reclogdiff) - 1)
    for i in range(1, len(reclogdiff)):
        reclog[i] = reclog[i - 1] + reclogdiff[i]

    # Undo log transform
    return np.exp(reclog)

# Load the dataset
df = pd.read_csv("../../datasets/M3C_monthly.csv")
rawdata = df.iloc[490,6:].values.astype(float)
train,test = rawdata[:-12], rawdata[-12:]

m = 12

logdata, logdiffdata, logdiffmdata = preprocessing(train, m)

# forecast

# Arma prediction and forecast

model = ARIMA(logdiffmdata, order=(2, 0, 1))
model_fit = model.fit()
# make prediction (in-sample or out-of-sample)
# armaPred = model_fit.predict(0, len(logdiff))
# make forecast (out of sample)
armaFore = model_fit.forecast(len(test))

# Arima prediction and forecast

model = ARIMA(logdiffmdata, order=(2, 1, 1))
model_fit = model.fit()
# make prediction (in-sample or out-of-sample)
# armaPred = model_fit.predict(0, len(logdiff))
# make forecast (out of sample)
arimaFore = model_fit.forecast(len(test))

# reconstruct

armaRecFore = reconstruct_fore(logdiffdata, logdata, armaFore, m)
arimaRecFore = reconstruct_fore(logdiffdata, logdata, arimaFore, m)

# Sarima model
model = pm.auto_arima(train, start_p=1, start_q=1,
test='adf', max_p=3, max_q=3, m=4,
start_P=0, seasonal=True,
d=None, D=1, trace=True,
error_action='ignore',
suppress_warnings=True,
stepwise=True) # stepwise=False full grid
print(model.summary())
morder = model.order # p,d,q
mseasorder = model.seasonal_order # P,D,Q,m
fitted = model.fit(train)
sarimaFore = fitted.predict(n_periods=12) # forecast
# sarimaPred = fitted.predict_in_sample()



# plot

plt.plot(rawdata[:-12], label="train")
plt.plot(range(len(rawdata)-12,len(rawdata)),armaRecFore,label="arma prediction")
plt.plot(range(len(rawdata)-12,len(rawdata)),arimaRecFore,label="arima prediction")
plt.plot(range(len(train), len(train) + len(sarimaFore)), sarimaFore,label="sarima prediction")
plt.plot(range(len(rawdata)-12,len(rawdata)),rawdata[-12:], label="test")
plt.title("M3 series"),plt.xlabel("time"),plt.ylabel("value")
plt.legend()
plt.show()