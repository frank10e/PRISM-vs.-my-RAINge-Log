import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# read data
data = pd.read_csv('data.csv')

# ensure the date column is in datetime format and sort by date
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')

# select features and target
features = ['tempHigh', 'tempLow', 'tempAvg', 'windspeedHigh', 'windspeedLow', 'windspeedAvg', 'windgustHigh',
            'windgustLow', 'windgustAvg', 'dewptHigh', 'dewptLow', 'dewptAvg', 'windchillHigh', 'windchillLow',
            'windchillAvg', 'heatindexHigh', 'heatindexLow', 'heatindexAvg', 'pressureMax', 'pressureMin',
            'pressureTrend', 'precipRate']
target = ['precipTotal']

# normalization
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# split data
train_data = data.iloc[:-60]
test_data = data.iloc[-60:]

# create time series sequences
sequence_length = 8

def create_sequences(data, target, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length].flatten()
        y = target[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(train_data[features].values, train_data[target].values, sequence_length)
X_test, y_test = create_sequences(test_data[features].values, test_data[target].values, sequence_length)

# use random forest to train the model
model = RandomForestRegressor(n_estimators=200,max_depth=10, random_state=49)
model.fit(X_train, y_train)

# predict
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# calculate MSE
train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
print(f'Train MSE: {train_mse}')
print(f'Test MSE: {test_mse}')

# transform the dates
train_dates = data['date'].iloc[sequence_length:len(train_predictions)+sequence_length]
test_dates = data['date'].iloc[-len(test_predictions):]

# plot the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(train_dates, y_train, label='Actual')
plt.plot(train_dates, train_predictions, label='Predicted')
plt.title('Training Set')
plt.xlabel('Date')
plt.ylabel('PrecipTotal')
plt.xticks(rotation=45)
plt.legend()

# plot the results
plt.subplot(1, 2, 2)
plt.plot(test_dates, y_test, label='Actual')
plt.plot(test_dates, test_predictions, label='Predicted')
plt.title('Testing Set')
plt.xlabel('Date')
plt.ylabel('PrecipTotal')
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig('random_forest.png')
plt.show()
