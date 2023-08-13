import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import tensorflow

#Getting our datasets
start= '2012-01-01'
end='2022-11-08'
df= data.DataReader('AAPL','yahoo', start,end)
df= df.reset_index()
df.drop(['Date', 'Adj Close'], axis=1)



#split into train and test
print(df.shape)
data_training= pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
print("trainin",data_training.shape)
print(data_training.head())
print(data_training.tail())
print("testing",data_testing.shape)
print(data_testing.head())
print(data_testing.tail())

#sklearn training
scaler= MinMaxScaler(feature_range=(0,1))
data_training_array= scaler.fit_transform(data_training)
print("Data training fit transform",data_training_array)
#xtrain,y train
x_train= []
y_train= []
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i]) #starts from (100 -100=0, i)
    y_train.append(data_training_array[i,0])
#There are some rules to give input data to the LSTM model, so we reshape the data accordingly
x_train, y_train= np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print("Xtrain is",x_train.shape)

#Machine Learning model
model1= Sequential()
model1.add(LSTM(units= 70,  return_sequences= True, input_shape= (x_train.shape[1],1))) #xtrain in 1 index and 1 is 'close' column
model1.add(LSTM(units= 64, return_sequences= False))
model1.add(Dense(25))
model1.add(Dense(1))
print(model1.summary())

model1.compile(optimizer= 'adam', loss= 'mean_squared_error')
model1.fit(x_train, y_train,batch_size=10, epochs=20)
model1.save('keras_model1_mini.h5')

#past 100 days is used to predict next day's closing price which is stored in data_testing
past_100_days= data_training.tail(100) # the last 100 days stock price value is taken from the training data
final_df= past_100_days.append(data_testing, ignore_index= True) # The past 100 days value is appended with the remaining 30% data.



#scale down testing
input_data= scaler.fit_transform(final_df)
print("final df after scale down",input_data)
print(input_data)

#testing
x_test= []
y_test= []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100 : i])
    y_test.append(input_data[i,0])
x_test, y_test= np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
y_test= y_test.reshape(-1,1)
print("xtest shape", x_test)
print(x_test)
print("Ytest shape", y_test)
print(y_test)

#making prediction and predicting vs testing

y_predicted= model1.predict(x_test)
y_predicted= scaler.inverse_transform(y_predicted)

y_test= scaler.inverse_transform(y_test)
print("My predicted values \n",y_predicted)
print("My original values \n", y_test)

# Plotting the Predictions vs Original value graph
plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label= 'Original Price')
plt.plot(y_predicted, 'r', label= 'Predicted Price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()


