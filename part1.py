import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Loads the data
train_data = pd.read_csv('midtermProject-part1-TRAIN.csv')
test_data = pd.read_csv('midtermProject-part1-TEST.csv')

# Extracts features and target
X_train = train_data.drop(columns=['ANGLE-ACC-ARM'])
y_train = train_data['ANGLE-ACC-ARM']

X_test = test_data.drop(columns=['ANGLE-ACC-ARM'])
y_test = test_data['ANGLE-ACC-ARM']

# Normalizes data (min-max scaling)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Converts data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)  

X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)  

# Creates DataLoader for training and testing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Defining neural network architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# Defining hyperparameters
input_size = 32  # Number of features
hidden_size1 = 4  # Number of nodes in the first hidden layer
hidden_size2 = 3  # Number of nodes in the second hidden layer
output_size = 1  # Since it's a regression task

learning_rate = 0.01
momentum = 0.9  
batch_size = 64
num_epochs = 100  

# Initializing the model, loss function, and optimizer
model = MLP(input_size, hidden_size1, hidden_size2, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)




# Training loop
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Print intermediate results
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluating test and train performance
model.eval()
with torch.no_grad():
    predictions = []
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predictions.append(outputs.numpy())

model.eval()
with torch.no_grad():
    predictions2 = []
    for inputs, targets in train_loader:
        outputs = model(inputs)
        predictions2.append(outputs.numpy())

predictions = np.concatenate(predictions)
test_mae = mean_absolute_error(y_test, predictions)
test_mse = mean_squared_error(y_test, predictions)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, predictions)

predictions2 = np.concatenate(predictions2)
train_mae = mean_absolute_error(y_train, predictions2)
train_mse = mean_squared_error(y_train, predictions2)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, predictions2)




# Saving the performance scores and model details to a text file
with open('part1_report.txt', 'w') as f:
    # Training results
    f.write(f"Tahsin Berk Oztekin - 21070001035\n")
    f.write(f"Used Libraries:\n")
    f.write(f"pandas: Data manipulation and organization.\n")
    f.write(f"scikit-learn: Data splitting, scaling, and metrics.\n")
    f.write(f"numpy: Numerical operations and array handling.\n")
    f.write(f"torch: Neural network creation.\n")
    f.write(f"torch.nn: Neural network layers.\n")
    f.write(f"torch.optim: Optimizers for training.\n")
    f.write(f"torch.utils.data: Data handling for PyTorch.\n\n")
    f.write(f"Train results:\n")
    f.write(f"MAE: {train_mae:.4f}\n")
    f.write(f"MSE: {train_mse:.4f}\n")
    f.write(f"RMSE: {train_rmse:.4f}\n")
    f.write(f"R2: {train_r2:.4f}\n\n")

    # Test results
    f.write("Test results:\n")
    f.write(f"MAE: {test_mae:.4f}\n")
    f.write(f"MSE: {test_mse:.4f}\n")
    f.write(f"RMSE: {test_rmse:.4f}\n")
    f.write(f"R2: {test_r2:.4f}\n\n")

    # Model details
    f.write("Model details:\n")
    f.write(f"Number of hidden layers: 2\n")  
    f.write(f"Number of nodes in each hidden layer: {hidden_size1}, {hidden_size2}\n")
    f.write(f"Activation function in each layer: ReLU\n")  
    f.write(f"Learning rate: {learning_rate}\n")
    f.write(f"Momentum value: {momentum}\n")  
    f.write(f"Gradient method: SGD\n")  
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Number of epochs: {num_epochs}\n\n")
    f.write(f"Final values of the weights and biases:\n")
    for name, parameter in model.named_parameters():
        f.write (f"{name}\n")
        f.write (f"{parameter}\n")



