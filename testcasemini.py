import numpy as np # arrays
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

#load data
company='FB'

start= dt.datetime(2018, 1,1)
end= dt.datetime(2022,1,1)
df= data.DataReader(company,'yahoo', start,end)

#prepare data
scaler= MinMaxScaler(feature_range=(0,1))
scaled_data= scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days= 100

x_train= []
y_train= []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x,0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train= np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

#build the model

model= Sequential()

model.add(LSTM(units= 50, return_sequences=True, input_shape= x_train.shape[1],1))
model.add(Dropout(0.2))
model.add(LSTM(units= 50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units= 50))
model.add(Dropout(0.2))
model.add(Dense(units=1))# prediction of next closing price

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs= 25, batch_size=32)

#how well will model performa based on data we know past data
#test accuracy on existing data

#load test data
test_start= dt.datetime(2020,1,1)
test_end= dt.datetime.now()

test_data= data.DataReader(company, 'yahoo', test_start, test_end)
actual_prices= test_data['Close'].values

total_dataset= pd.concat((data['Close'], test_data['Close']), axis=0)

