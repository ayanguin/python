from bs4 import BeautifulSoup
import requests
import csv;
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential # INITIALISE THE RNN
from keras.layers import Dense # CREATE THE OUTPUT LAYER OF RNN
from keras.layers import LSTM
import matplotlib.pyplot as plt

#extracting the train_data from the web (8-AUG-2019 to 30-DEC-2019)
url = "https://in.finance.yahoo.com/quote/GOOGL/history?period1=1423440000&period2=1577750400&interval=1d&filter=history&frequency=1d"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
s_table = soup.find_all('div', class_="Pb(10px) Ovx(a) W(100%)")
s_table = s_table[0]
for row in s_table.find_all('tr'):
    for col in row.find_all('td'):
        col.text
#save the extracted data in a csv file
with open ('google_train.csv','w',newline='') as r:
    writer = csv.writer(r)
    writer.writerow(['Date','Open','High','Low','Close','Adj. close','Volume'])
    for row in s_table.find_all('tr'):
        s_lidt = []
        for col in row.find_all('td'):
            s_lidt.append(col.text)
        writer.writerow(s_lidt)
        

#extracting the test_data from the web. (2-JAN-2020 to 30-JAN-2020)
url = "https://in.finance.yahoo.com/quote/GOOGL/history?period1=1577836800&period2=1580428800&interval=1d&filter=history&frequency=1d"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
s_table = soup.find_all('div', class_="Pb(10px) Ovx(a) W(100%)")
s_table = s_table[0]
for row in s_table.find_all('tr'):
    for col in row.find_all('td'):
        col.text
#save the extracted data in a csv file
with open ('google_test.csv','w',newline='') as r:
    writer = csv.writer(r)
    writer.writerow(['Date','Open','High','Low','Close','Adj. close','Volume'])
    for row in s_table.find_all('tr'):
        s_lidt = []
        for col in row.find_all('td'):
            s_lidt.append(col.text)
        writer.writerow(s_lidt)
        
        
#store and read the csv file in a variable named DATASET.
d_train = pd.read_csv("google_train.csv")
train = pd.read_csv("google_train.csv")
d_train = d_train.replace(",","", regex=True) #to remove all commas from the dataset.

#iloc is used to extract the 1st column of the train dataset. 
dataset_train_old = d_train.iloc[:,1:2].values 

# FEATURE SCALING
sc = MinMaxScaler()
# it will help us to understand and visualise the values more clearly.
dataset_train = sc.fit_transform(dataset_train_old)

# Getting INPUT and OUTPUTS for LSTM model
X_train = dataset_train[0:98] #stock price at time T (INPUT)
Y_train = dataset_train[1:99] #stock price at time T+1 (OUTPUT)

# RESHAPEING. reshaping the dataset into 2D to 3D .
# as we have input at time T and output at time T+1.
# (T+1)-T = 1
X_train = np.reshape(X_train, (98, 1, 1))


# BUILDING THE RNN.

# initialise RNN
re = Sequential()

# adding the input layer and LSTM layer
re.add(LSTM(units = 50, activation = 'tanh', input_shape = (None,1)))

# adding the output layer and LSTM layer
re.add(Dense(units = 1))

# Compile the RNN
re.compile(optimizer  = 'adam', loss = 'mean_squared_error')

# Fit the RNN to the training set
re.fit(X_train, Y_train, batch_size = 10, epochs = 55)

# getting the test csv file.
test = pd.read_csv("google_test.csv")
dataset_test = test.replace(",","", regex=True) # to remove all commas from the dataset.
m = int(input("enter the starting day of the month [1 to 21]: "))
n = int(input("enter the end day of the mpnth [1 to 21]: "))
real_stock_price = dataset_test.iloc[m:n,1:2].values 
real_stock_price = np.float64(real_stock_price) # conversion from array of string to array of float64

# getting the predicted stock price
inputs = real_stock_price
inputs = sc.transform(inputs) 
inpu = np.reshape(inputs, ((n-m), 1, 1)) 
predicted_stock_price = re.predict(inpu) 
predicted_stock = sc.inverse_transform(predicted_stock_price) 

# visualise the result 
plt.plot(real_stock_price, color = 'red', label = 'REAL GOOGLE STOCK PRICE')
plt.plot(predicted_stock, color = 'blue', label = 'PREDICTED GOOGLE STOCK PRICE') 
plt.title('GOOGLE STOCK PRICE PREDICTION')
plt.xlabel('number of days')
plt.ylabel('google stock price')
plt.legend()
plt.show()


