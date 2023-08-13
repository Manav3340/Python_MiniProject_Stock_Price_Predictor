import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.models import load_model
import streamlit as st
import tensorflow



# Getting our datasets from yahoo finance
start= '2012-01-01'
end= '2022-11-19'
st.title("Stock Price Predictor")
user_input= st.text_input('Enter Stock Ticker', 'AAPL')
df= data.DataReader(user_input,'yahoo', start,end)
print(df.head())
print(df.tail())

#Describe data
st.subheader("Data from 2012- 2022")
st.write(df.describe())

#visualizations: Graph between closing price and time
st.subheader("Closing Price vs Time chart")
st.subheader(user_input)
fig= plt.figure(figsize= (12,6))
plt.plot(df.Close)
st.pyplot(fig)


#split data again
data_training= pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

# scale down dataset values into values ranging between 0 and 1
scaler= MinMaxScaler(feature_range=(0,1))
data_training_array= scaler.fit_transform(data_training)




#load model
model= load_model('keras_model1_mini.h5')
#past 100 days is used to predict next day's closing price which is stored in data_testing
past_100_days= data_training.tail(100) # the last 100 days stock price value is taken from the training data
final_df= past_100_days.append(data_testing, ignore_index= True)
input_data= scaler.fit_transform(final_df)
# Preparing testing data
x_test= []
y_test= []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100 : i])
    y_test.append(input_data[i,0])

x_test, y_test= np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
y_test= y_test.reshape(-1,1)
print(x_test)
print(y_test)
#making prediction and predicting vs testing
y_predicted= model.predict(x_test)
# Scaling values back to their original values
y_predicted= scaler.inverse_transform(y_predicted)
y_test= scaler.inverse_transform(y_test)
print("My predicted values \n",y_predicted)
print("My original values \n", y_test)

#Plotting graph between Predictions and Original data
st.subheader("Predictions vs Original graph")
st.subheader(user_input)

#mean absolute percentage error if less than 5 Percent it means its a good model
# accuracy % = 100% - mape
mape = np.mean(np.abs((y_test- y_predicted)/y_test)) *100
accuracy_percentage= 100- mape
st.subheader("The accuracy of the predictions made by the model are ")
st.subheader(accuracy_percentage)
print(mape)

#predict next day

#Plotting the graph
fig2= plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label= 'Original Price')
plt.plot(y_predicted, 'r', label= 'Predicted Price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)