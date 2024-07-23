import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

# read data
data = pd.read_csv('data.csv')
features = ['tempHigh', 'tempLow', 'tempAvg', 'windspeedHigh', 'windspeedLow', 'windspeedAvg', 'windgustHigh',
            'windgustLow', 'windgustAvg', 'dewptHigh', 'dewptLow', 'dewptAvg', 'windchillHigh', 'windchillLow',
            'windchillAvg', 'heatindexHigh', 'heatindexLow', 'heatindexAvg', 'pressureMax', 'pressureMin',
            'pressureTrend', 'precipRate']
target = ['precipTotal']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Normalization
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
        x = data[i:i+seq_length]
        y = target[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(train_data[features].values, train_data[target].values, sequence_length)
X_test, y_test = create_sequences(test_data[features].values, test_data[target].values, sequence_length)

# transform to tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# create dataloader
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = WeatherDataset(X_train, y_train)
test_dataset = WeatherDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


import torch.nn as nn

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_size = len(features)
hidden_size = 50
num_layers = 2
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# train model
num_epochs = 100

# Train the model
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# test model
model.eval()
with torch.no_grad():
    test_loss = 0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss}')


import matplotlib.pyplot as plt

# get train set prediction
model.eval()
train_predictions = []
train_targets = []

with torch.no_grad():
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        train_predictions.extend(outputs.cpu().numpy())
        train_targets.extend(y_batch.cpu().numpy())

# gey test set prediction
test_predictions = []
test_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        test_predictions.extend(outputs.cpu().numpy())
        test_targets.extend(y_batch.cpu().numpy())

# transform to numpy array
train_predictions = np.array(train_predictions)
train_targets = np.array(train_targets)
test_predictions = np.array(test_predictions)
test_targets = np.array(test_targets)


# transform to original scale
train_dates = data['date'].iloc[sequence_length:len(train_targets)+sequence_length]
test_dates = data['date'].iloc[-len(test_targets):]

# plot result
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(train_dates, train_targets, label='Actual')
plt.plot(train_dates, train_predictions, label='Predicted')
plt.title('Training Set')
plt.xlabel('Date')
plt.ylabel('PrecipTotal')
plt.xticks([train_dates.iloc[0],train_dates.iloc[100],train_dates.iloc[200],train_dates.iloc[-1]])
plt.legend()

# plot test set
plt.subplot(1, 2, 2)
plt.plot(test_dates, test_targets, label='Actual')
plt.plot(test_dates, test_predictions, label='Predicted')
plt.title('Testing Set')
plt.xlabel('Date')
plt.ylabel('PrecipTotal')
plt.xticks([test_dates.iloc[0],test_dates.iloc[25],test_dates.iloc[-1]])
plt.legend()

plt.title('Lstm Result')
plt.savefig('lstm.png')

