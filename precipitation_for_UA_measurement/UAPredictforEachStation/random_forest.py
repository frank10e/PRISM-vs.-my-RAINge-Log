import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# read data
df = pd.read_csv('data.csv')
stations = df['STATION'].unique()

# Define a function to create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        max_precip = np.max(data[i:i+seq_length])
        min_precip = np.min(data[i:i+seq_length])
        avg_precip = np.average(data[i:i+seq_length])
        x = [max_precip,min_precip,avg_precip]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# use RandomForestRegressor to predict the precipitation
for station in stations:
    print(f'Station = {station}')
    data = pd.read_csv(f'{station}/data.csv')
    data=data.fillna(0)


    # split the data into training and testing sets
    train_data = data.iloc[:-50]
    test_data = data.iloc[-50:]

    # sequence length
    sequence_length = 3

    X_train, y_train = create_sequences(train_data['precip'].values, sequence_length)
    X_test, y_test = create_sequences(test_data['precip'].values, sequence_length)


    try:
        # model
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # predictions
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        # MSE
        train_mse = mean_squared_error(y_train, train_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)
        print(f'Train MSE: {train_mse}')
        print(f'Test MSE: {test_mse}')

        # Plot for the training and testing sets
        train_dates = data['date'].iloc[sequence_length:len(train_predictions) + sequence_length]
        test_dates = data['date'].iloc[-len(test_predictions):]

        # Plot the training and testing sets
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_dates, y_train, label='Actual')
        plt.plot(train_dates, train_predictions, label='Predicted')
        plt.title('Training Set')
        plt.xlabel('Date')
        plt.ylabel('PrecipTotal')
        plt.xticks([train_dates.iloc[0], train_dates.iloc[len(train_dates) // 2], train_dates.iloc[-1]], rotation=45)
        plt.legend()

        # Plot the testing set
        plt.subplot(1, 2, 2)
        plt.plot(test_dates, y_test, label='Actual')
        plt.plot(test_dates, test_predictions, label='Predicted')
        plt.title('Testing Set')
        plt.xlabel('Date')
        plt.ylabel('PrecipTotal')
        plt.xticks([test_dates.iloc[0], test_dates.iloc[len(test_dates) // 2], test_dates.iloc[-1]], rotation=45)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{station}/random_forest.png')
    except:
        continue
    # plt.show()
