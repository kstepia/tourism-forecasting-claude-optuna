# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: tf-cert
#     language: python
#     name: python3
# ---

# %%
# !pip list

# %%
# !pip install pandas

# %%
# !pip install matplotlib

# %%
# !pip install statsmodels

# %%
# !pip install -U scikit-learn

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf

# %%
# Load the tourism time series data
df = pd.read_csv('/Users/kdy/Downloads/python-test/toursim_data.csv', header=0, index_col=0, parse_dates=True)

# %%
# Visualize the data
plt.plot(df)
plt.show()

# %%
# Remove trend and seasonality using differencing
df_diff = df.diff().dropna()

# %%
# Check for stationarity
result = adfuller(df_diff)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
if result[1] > 0.05:
    print('Data is not stationary')
else:
    print('Data is stationary')

# %%
# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_diff)

# %%
# Create train and test sets
train_size = int(len(scaled_data) * 0.7)
test_size = len(scaled_data) - train_size
train, test = scaled_data[0:train_size,:], scaled_data[train_size:len(scaled_data),:]

# %%
# Convert data into sequences for LSTM model
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data)-sequence_length-1):
        X.append(data[i:(i+sequence_length), 0])
        y.append(data[i+sequence_length, 0])
    return np.array(X), np.array(y)

# %%
sequence_length = 10
trainX, trainY = create_sequences(train, sequence_length)
testX, testY = create_sequences(test, sequence_length)

# %%
# Build LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=72, verbose=2)

# %%
# Evaluate the model
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# %%
# Inverse scale the predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# %%
# Plot the predictions
plt.plot(df_diff.index[sequence_length+1:train_size+1], trainY[0])
plt.plot(df_diff.index[sequence_length+1:train_size+1], trainPredict[:,0])
plt.plot(df_diff.index[train_size+sequence_length+1:len(scaled_data)-1], testY[0])
plt.plot(df_diff.index[train_size+sequence_length+1:len(scaled_data)-1], testPredict[:,0])
plt.show()
