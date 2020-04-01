
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

#extracting the train_data from the web (max 6 months)
url = input("Enter the URL from YAHOO FINANCE: \t")
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
        

#extracting the test_data from the web.
url = input("enter the URL from YAHOO FINANCE: \t")
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
d_train = d_train.replace("-","0", regex=True)
d_train = d_train.drop_duplicates('Date')
print("\npress  1 for OPEN column\npress  2 for HIGH column\npress  3 for LOW column\npress  4 for CLOSE column\npress  5 for ADJ.CLOSE column\n")
v = int(input("\nenter the column number upon which you want to predict "))

#iloc is used to extract the 1st column of the train dataset. 
dataset_train_old = d_train.iloc[:,v:(v+1)].values 

# FEATURE SCALING
sc = MinMaxScaler()
# it will help us to understand and visualise the values more clearly.
dataset_train = sc.fit_transform(dataset_train_old)
z = len(dataset_train)-2
# Getting INPUT and OUTPUTS for LSTM model
x_train = dataset_train[0:(z)] #stock price at time T (INPUT)
Y_train = dataset_train[1:(z+1)] #stock price at time T+1 (OUTPUT)

# RESHAPEING. reshaping the dataset into 2D to 3D .
# as we have input at time T and output at time T+1.
# (T+1)-T = 1
X_train = np.reshape(x_train, (z, 1, 1))


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
print(f"\nenter the starting day of the month 1 to {len(dataset_test)}")
m = int(input(""))
print(f"enter the ending day of the month 1 to {len(dataset_test)}")
n = int(input(""))
real_stock_price = dataset_test.iloc[m:n,v:(v+1)].values 
real_stock_price = np.float64(real_stock_price) # conversion from array of string to array of float64

# getting the predicted stock price
inputs = real_stock_price
inputs = sc.transform(inputs) 
inpu = np.reshape(inputs, ((n-m), 1, 1)) 
predicted_stock_price = re.predict(inpu) 
predicted_stock = sc.inverse_transform(predicted_stock_price) 

# visualise the result 
plt.plot(real_stock_price, color = 'red', label = 'REAL STOCK PRICE')
plt.plot(predicted_stock, color = 'blue', label = 'PREDICTED STOCK PRICE') 
plt.title('STOCK PRICE PREDICTION')
plt.xlabel('number of days')
plt.ylabel('stock price in USD')
plt.legend()
plt.show()

er = 0
erro = []
for i, (r, p) in enumerate(zip(real_stock_price[0:len(real_stock_price)-1], predicted_stock[0:len(predicted_stock)-1])):
    erro.append(abs((float(r[0]) - float(p[0])) * 100) / float(r[0]))
     
fmt = '%-20s%-20s%-20s%s'
print(fmt % ('', 'real_stock', 'predicted_stock', 'error(%)'))
for i, (r, p, e) in enumerate(zip(real_stock_price[0:len(real_stock_price)-1], predicted_stock[0:len(predicted_stock)-1], erro)):
    print(fmt % (i, "%.2f" % float(r[0]), "%.2f" % float(p[0]), "%.2f" % float(e)))
    er += (abs((float(r[0]) - float(p[0])) * 100) / float(r[0]))
    
    
print("\naccuracy is: ",(100 - round(er / (n-m),2)),"%")

