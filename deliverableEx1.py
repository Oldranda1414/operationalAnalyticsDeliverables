import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

def rec(missingdf):
    result = []
    result.append((missingdf.fillna(method="ffill"), "ffill"))
    result.append((missingdf.fillna(method="bfill"), "bfill"))
    result.append((missingdf.fillna(missingdf.mean()), "mean"))
    result.append((missingdf.interpolate(), "interpolate"))
    return result

def plot_rec(recs, title):
    plt.figure(figsize=(10, 5))
    for recunstruct in recs:
        plt.plot(recunstruct[0], label=recunstruct[1])
    plt.legend()
    plt.title(title)
    plt.show()

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual)) # MAPE
    me = np.mean(forecast - actual) # ME
    mae = np.mean(np.abs(forecast - actual)) # MAE
    mpe = np.mean((forecast - actual)/actual) # MPE
    rmse = np.mean((forecast - actual)**2)**.5 # RMSE
    corr = np.corrcoef(forecast, actual)[0,1] # correlation coeff
    return({'mape':mape, 'me':me, 'mae': mae, 'mpe': mpe, 'rmse':rmse,
            'corr':corr})

missing6 = pd.read_csv("../../datasets/missing6BoxJenkins.csv", usecols=["Passengers"])
missing4 = pd.read_csv("../../datasets/missing4BoxJenkins.csv", usecols=["Passengers"])
missing3 = pd.read_csv("../../datasets/missing3BoxJenkins.csv", usecols=["Passengers"])

df_path = '../../datasets/BoxJenkins.csv'
df = pd.read_csv(df_path)

missing6Rec = rec(missing6)
missing4Rec = rec(missing4)
missing3Rec = rec(missing3)

# plot_rec(missing6Rec, "rec with 6 missing values")
# plot_rec(missing4Rec, "rec with 4 missing values")
# plot_rec(missing3Rec, "rec with 3 missing values")


training = df.iloc[:-24]
validation = df.iloc[-24:-12]
forecast = df.iloc[-12:]

dflog = np.log(training["Passengers"].to_numpy())

diff1 = [dflog[i] - dflog[i - 1] for i in range(1, len(dflog))]

plt.figure(figsize=(10, 5))
sm.graphics.tsa.plot_acf(np.array(diff1))
plt.show()
m = 12

diffm = [diff1[i] - diff1[i - m] for i in range(m, len(diff1))]

adf_result = adfuller(diffm)
# results (ADF statistic and p-value)
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')

shapiro_test = stats.shapiro(diffm)
print(f"Shapiro-Wilk Test: Statistic={shapiro_test.statistic}, pvalue={shapiro_test.pvalue}")

x = np.arange(len(diffm))
y = diffm
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.figure(figsize=(10, 5))
plt.plot(y)
plt.plot(x,p(x),"r--")
plt.show()

validation_range = range(len(training) , len(training) + 12)

pred = np.polyval(z, x) # the linear model
valid = np.polyval(z, validation_range) # extension

# reconstrunct diff1
recValid1 = np.append(diff1[-m:], valid)
for i in range(m, len(recValid1)):
    recValid1[i] += recValid1[i - m]
# reconstruct dflog
recValidLog = np.append([dflog[-1]] ,recValid1[-12:])
for i in range(1, 12):
    recValidLog[i] += recValidLog[i - 1]
# reconstruct original
recValid = np.exp(recValidLog[:-1])

print(forecast_accuracy(recValid, validation["Passengers"]))

forecast_range = range(len(training) + 12, len(training) + 24)

pred = np.polyval(z, x) # the linear model
forec = np.polyval(z, forecast_range) # extension

# reconstrunct diff1
recFore1 = np.append(recValid1[-m:], forec)
for i in range(m, len(recFore1)):
    recFore1[i] += recFore1[i - m]
# reconstruct dflog
recForeLog = np.append([recValidLog[-2]] ,recFore1[-12:])
for i in range(1, 12):
    recForeLog[i] += recForeLog[i - 1]
# reconstruct original
recFore = np.exp(recForeLog[:-1])

print(forecast_accuracy(recFore, forecast["Passengers"]))

plt.figure(figsize=(10, 5))
plt.plot(validation_range, recValid, label="predicted")
plt.plot(forecast_range, recFore, label="forecast")
plt.plot(range(len(df)), df["Passengers"].to_numpy(), label="original")
plt.show()
