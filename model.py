import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(577)

def label_encode(df, cols):
    feature_mappings = {}
    df_feature = pd.DataFrame()
    for column in cols:
        categories = df[column].unique()
        label_mapping = {}
        encoded_labels = []
        label_count = 1

        for category in categories:
            label_mapping[category] = label_count
            label_count += 1

        # print(column)
        # print(df[column].isna().sum())
        # print(label_mapping)
        for value in df[column]:
            encoded_labels.append(label_mapping[value])

        feature_mappings[column] = label_mapping
        df_feature[column] = encoded_labels

    return feature_mappings, df_feature

def train_embedding(df, feature_cols, label_cols):
    data = {}
    encoded_data = pd.DataFrame()
    data, encoded_data = label_encode(df, feature_cols)

    labels = {}
    encoded_labels = pd.DataFrame()
    labels, encoded_labels = label_encode(df, label_cols)

    for column in label_cols:
        encoded_data[column] = encoded_labels[column]

    return encoded_data, data, labels

def test_embedding(df, features, labels):
    new_data = pd.DataFrame()
    for column in features.keys():
        encoded_labels = []
        for value in df[column]:
            if features[column].get(value) is not None:
                encoded_labels.append(features[column][value])
            else:
                features[column][value] = len(features[column])+1
                encoded_labels.append(features[column][value])
            # print(value)
            # print(features[column])
        new_data[column] = encoded_labels

    for column in labels.keys():
        encoded_labels = []
        for value in df[column]:
            if labels[column].get(value) is not None:
                encoded_labels.append(labels[column][value])
            else:
                labels[column][value] = len(labels[column])+1
                encoded_labels.append(labels[column][value])
            # print(value)
            # print(labels[column])
        new_data[column] = encoded_labels

    return new_data
                    

    
# Custom Dataset class
class TrafficDataset(Dataset):
    def __init__(self, df, feature_cols, label_cols):
        """
        Args:
            dataframe (DataFrame): Pandas DataFrame containing the data.
            feature_cols (list): List of column names to be used as features.
            label_cols (list): List of column names to be used as labels.
        """
        self.features = df[feature_cols]
        self.labels = df[label_cols]
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.tensor(self.features.iloc[idx].values, dtype=torch.float)
        label = torch.tensor(self.labels.iloc[idx].values, dtype=torch.float)

        # print(feature)
        return feature, label



class TrafficCNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(TrafficCNN, self).__init__()
        self.conv1 = nn.Conv1d(num_features, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(32 * 1, 32)  # Adjust the input features to match your data
        self.fc2 = nn.Linear(32, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Add sequence of convolutional and max pooling layers
        x = F.relu(self.conv1(x))
        # print("After conv1:", x.size())
        x = F.relu(self.conv2(x))
        # print("After conv2:", x.size())
        x = x.view(-1, 32 * 1)  # Flatten the tensor
        # print("After view:", x.size())
        x = F.relu(self.fc1(x))
        # print("After fc1:", x.size())
        x = F.relu(self.fc2(x))
        # print("After fc2:", x.size())
        x = self.fc3(x)
        # print("After fc3:", x.size())
        return x

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate the RNN
        out, _ = self.rnn(x, h0)
        
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out


def train_model(train_loader, valid_loader, model, criterion, optimizer, num_epochs=25):
    for epoch in range(1):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for i, (feature, label) in enumerate(train_loader):
            # Zero the parameter gradients
            feature, label = feature.to(device), label.to(device)
            feature = feature.unsqueeze(2)

            optimizer.zero_grad()
            outputs = model(feature)
            print(outputs)
            loss = criterion(outputs, label)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if i % 5 == 4:    # Print every 100 mini-batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

    # Validation loss
    correct = 0
    total = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for feature, label in valid_loader:
            feature, label = feature.to(device), label.to(device)
            feature = feature.unsqueeze(2)
            outputs = model(feature)
            print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            print(predicted)
            correct += (predicted == label).sum().item()

    print(f'Validation Accuracy: {100 * correct / total:.2f}%')

    print('Finished Training')
        




# Assuming 'data' is a dictionary of pandas DataFrames for 'train_set', 'test_set', and 'valid_set'
feature_cols = ['TRAFFIC_CONTROL_DEVICE', 'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'ROADWAY_SURFACE_COND', 'POSTED_SPEED_LIMIT', 'TRAFFICWAY_TYPE']
label_cols = ['CRASH_TYPE', 'DAMAGE', 'MOST_SEVERE_INJURY', 'INJURIES_TOTAL']

# Define the path to the data folder
data_folder = "data/"

# Read the train_set.csv file
train_df = pd.read_csv(data_folder + "train_set.csv")

# Read the test_set.csv file
test_df = pd.read_csv(data_folder + "test_set.csv")

# Read the valid_set.csv file
valid_df = pd.read_csv(data_folder + "valid_set.csv")

train_data, features, labels = train_embedding(train_df, feature_cols, label_cols)
test_data = test_embedding(test_df, features, labels)
valid_data = test_embedding(valid_df, features, labels)


# Create datasets
train_dataset = TrafficDataset(train_data, feature_cols, label_cols)
test_dataset = TrafficDataset(test_data, feature_cols, label_cols)
valid_dataset = TrafficDataset(valid_data, feature_cols, label_cols)


# Create dataloaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Number of features after one-hot encoding
num_features = train_dataset.features.shape[1] # You need to calculate this based on your one-hot encoded data
# # Number of classes for the classification task
num_classes = train_dataset.labels.shape[1] # You need to define this based on your label data

    

# Instantiate the model

model = TrafficCNN(num_features, num_classes).to(device)

# input_size = 6  # Number of features
# hidden_size = 128  # Number of features in the hidden state
# num_layers = 2  # Number of stacked RNN layers
# num_classes = 4  # Number of output classes

# # Initialize the model
# model = RNNClassifier(input_size, hidden_size, num_layers, num_classes).cuda()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_model(train_loader, valid_loader, model, criterion, optimizer, num_epochs=25)