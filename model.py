import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import pandas as pd
import numpy as np

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

        # print(encoded_df)

        # # One-hot encode the features and labels
        # self.features = pd.get_dummies(df[feature_cols], columns=feature_cols).values
        # # print(self.features)
        # self.labels = pd.get_dummies(df[label_cols], columns=label_cols).values
        
        # # Convert to torch tensors
        # self.features = torch.tensor(self.features, dtype=torch.float32)
        # self.labels = torch.tensor(self.labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]



class TrafficCNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(TrafficCNN, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        # Assuming num_features is the number of total dummy features after one-hot encoding
        self.embedding = nn.Embedding(num_embeddings=num_features, embedding_dim=10)
        
        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=10, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * (num_features // 2), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Handle missing values in data
        # Assuming missing values have been encoded as a specific number, e.g., -1
        x = torch.where(x == -1, torch.zeros_like(x), x)
        
        # Embedding layer for one-hot encoded data
        x = self.embedding(x.long()).transpose(1, 2)
        
        # CNN layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 32 * (self.num_features // 2))
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(train_loader, valid_loader, model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for i, (features, labels) in enumerate(train_loader, 0):
            # Zero the parameter gradients
            optimizer.zero_grad()
            print(features)
            # Forward pass
            outputs = model(features)
            print(outputs.shape)
            print(labels.shape)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
        
        # Validation loss
        valid_loss = 0.0
        valid_steps = 0
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for features, labels in valid_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                valid_steps += 1
        print(f'Epoch {epoch + 1}, Validation Loss: {valid_loss / valid_steps:.4f}')

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

# train_data = {}
# encoded_train_data = pd.DataFrame()
# train_data, encoded_train_data = label_encode(train_df, feature_cols)

# train_labels = {}
# encoded_train_labels = pd.DataFrame()
# train_labels, encoded_train_labels = label_encode(train_df, label_cols)

train_data = {}
encoded_train_data = pd.DataFrame()
train_data, encoded_train_data = label_encode(test_df, feature_cols)

train_labels = {}
encoded_train_labels = pd.DataFrame()
train_labels, encoded_train_labels = label_encode(test_df, label_cols)

train_data = {}
encoded_train_data = pd.DataFrame()
train_data, encoded_train_data = label_encode(valid_df, feature_cols)

train_labels = {}
encoded_train_labels = pd.DataFrame()
train_labels, encoded_train_labels = label_encode(valid_df, label_cols)

print(train_labels, encoded_train_labels)



# Create datasets
train_dataset = TrafficDataset(train_df, feature_cols, label_cols)
test_dataset = TrafficDataset(test_df, feature_cols, label_cols)
valid_dataset = TrafficDataset(valid_df, feature_cols, label_cols)

# print(train_dataset.features['TRAFFIC_CONTROL_DEVICE'])







# # Create dataloaders
# batch_size = 32
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# # Number of features after one-hot encoding
# num_features = train_dataset.features.shape[1] # You need to calculate this based on your one-hot encoded data
# # Number of classes for the classification task
# num_classes = train_dataset.labels.shape[1] # You need to define this based on your label data


# # Instantiate the model
# model = TrafficCNN(num_features, num_classes)

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train_model(train_loader, valid_loader, model, criterion, optimizer, num_epochs=25)