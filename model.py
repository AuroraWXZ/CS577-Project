import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(577)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "logistic":
            self.activation = torch.sigmoid

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

        
def encode_targets(df, label_encoders, target):
    return label_encoders[target].transform(df[[target]])



# Assuming 'data' is a dictionary of pandas DataFrames for 'train_set', 'test_set', and 'valid_set'
feature_cols = ['TRAFFIC_CONTROL_DEVICE', 'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'ROADWAY_SURFACE_COND', 'POSTED_SPEED_LIMIT', 'TRAFFICWAY_TYPE']
label_cols = ['CRASH_TYPE', 'DAMAGE', 'MOST_SEVERE_INJURY', 'INJURIES_TOTAL']


# Define the path to the data folder
data_folder = "data/"

# Read the train_set.csv file
train_df = pd.read_csv(data_folder + "train_set.csv")

# Read the test_set.csv file
test_df = pd.read_csv(data_folder + "test_set - results - JY.csv")

# Read the valid_set.csv file
valid_df = pd.read_csv(data_folder + "valid_set.csv")

df = pd.concat([train_df, test_df, valid_df])

data_encoder = OneHotEncoder(sparse=False)
data_encoder.fit(df[feature_cols])

train_data = data_encoder.transform(train_df[feature_cols])
test_data = data_encoder.transform(test_df[feature_cols])
valid_data = data_encoder.transform(valid_df[feature_cols])


label_encoders = {}

for label in label_cols:
    label_encoder = OneHotEncoder(sparse=False)
    label_encoder.fit(df[[label]])
    label_encoders[label] = label_encoder


train_labels = {label: encode_targets(train_df, label_encoders, label) for label in label_cols}
valid_labels = {label: encode_targets(valid_df, label_encoders, label) for label in label_cols}
test_labels = {label: encode_targets(test_df, label_encoders, label) for label in label_cols}

# Number of features after one-hot encoding
num_features = train_data.shape[1] # You need to calculate this based on your one-hot encoded data
# Number of classes for the classification task
num_classes = len(train_labels) # You need to define this based on your label data


# Instantiate the model

# print(train_labels['DAMAGE'].shape)

model_CT = MLP(num_features, 32, train_labels['CRASH_TYPE'].shape[1], "logistic").to(device)
model_D = MLP(num_features, 64, train_labels['DAMAGE'].shape[1], "logistic").to(device)
model_MSI= MLP(num_features, 32, train_labels['MOST_SEVERE_INJURY'].shape[1], "logistic").to(device)
model_IT = MLP(num_features, 32, train_labels['INJURIES_TOTAL'].shape[1], "relu").to(device)

# Define loss function and optimizer
criterion_CT = nn.CrossEntropyLoss()
criterion_D  = nn.CrossEntropyLoss()
criterion_MSI = nn.CrossEntropyLoss()
criterion_IT = nn.CrossEntropyLoss()
optimizer_CT = torch.optim.Adam(model_CT.parameters(), lr=0.001)
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=0.005)
optimizer_MSI = torch.optim.Adam(model_MSI.parameters(), lr=0.001)
optimizer_IT = torch.optim.Adam(model_IT.parameters(), lr=0.0001)

# for ba in [16, 32, 64, 128, 256]:
#     CT_t = []
#     D_t = []
#     MSI_t = []
#     IT_t = []

#     CT_v = []
#     D_v = []
#     MSI_v = []
#     IT_v = []

#     f1, axv1 = plt.subplots()
#     f2, axv2 = plt.subplots()
#     for lr in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
#         model_CT = MLP(num_features, ba, train_labels['CRASH_TYPE'].shape[1], "logistic").to(device)
#         model_D = MLP(num_features, ba, train_labels['DAMAGE'].shape[1], "logistic").to(device)
#         model_MSI= MLP(num_features, ba, train_labels['MOST_SEVERE_INJURY'].shape[1], "logistic").to(device)
#         model_IT = MLP(num_features, ba, train_labels['INJURIES_TOTAL'].shape[1], "relu").to(device)

#         # Define loss function and optimizer
#         criterion_CT = nn.CrossEntropyLoss()
#         criterion_D  = nn.CrossEntropyLoss()
#         criterion_MSI = nn.CrossEntropyLoss()
#         criterion_IT = nn.CrossEntropyLoss()
#         optimizer_CT = torch.optim.Adam(model_CT.parameters(), lr=lr)
#         optimizer_D = torch.optim.Adam(model_D.parameters(), lr=lr)
#         optimizer_MSI = torch.optim.Adam(model_MSI.parameters(), lr=lr)
#         optimizer_IT = torch.optim.Adam(model_IT.parameters(), lr=lr)


        # f = plt.figure("Train Loss")

        # CT = []
        # D = []
        # MSI = []
        # IT = []

for epoch in range(10):
    for i in range(train_data.shape[0]):
        optimizer_CT.zero_grad()
        optimizer_D.zero_grad()
        optimizer_MSI.zero_grad()
        optimizer_IT.zero_grad()

        line = torch.tensor(train_data[i], dtype=torch.float).to(device)
        outputs_CT = model_CT(line)
        outputs_D = model_D(line)
        outputs_MSI = model_MSI(line)
        outputs_IT = model_IT(line)

        # print(outputs_D.shape)

        loss_CT = criterion_CT(outputs_CT, torch.tensor(train_labels['CRASH_TYPE'][i], dtype=torch.float).to(device))
        loss_D = criterion_D(outputs_D, torch.tensor(train_labels['DAMAGE'][i], dtype=torch.float).to(device))
        loss_MSI = criterion_MSI(outputs_MSI, torch.tensor(train_labels['MOST_SEVERE_INJURY'][i], dtype=torch.float).to(device))
        loss_IT = criterion_IT(outputs_IT, torch.tensor(train_labels['INJURIES_TOTAL'][i], dtype=torch.float).to(device))

        loss_CT.backward()
        loss_D.backward()
        loss_MSI.backward()
        loss_IT.backward()

        optimizer_CT.step()
        optimizer_D.step()
        optimizer_MSI.step()
        optimizer_IT.step()

        # if (i+1) % 50 == 0:
            # print(f'Epoch [{epoch+1}/10], Loss: {loss_CT.item()}')
            # print(f'Epoch [{epoch+1}/10], Loss: {loss_D.item()}')
            # print(f'Epoch [{epoch+1}/10], Loss: {loss_MSI.item()}')
            # print(f'Epoch [{epoch+1}/10], Loss: {loss_IT.item()}')
            # print("--------------- Line ---------------")
            # CT.append(loss_CT.item())
            # D.append(loss_D.item())
            # MSI.append(loss_MSI.item())
            # IT.append(loss_IT.item())

        # plt.plot(np.arange(len(CT)), CT)
        # plt.plot(np.arange(len(D)), D)
        # plt.plot(np.arange(len(MSI)), MSI, label = "MOST_SEVERE_INJURY")
        # plt.plot(np.arange(len(IT)), IT, label = "INJURIES_TOTAL")

        # plt.title("Development of loss during training")
        # plt.xlabel("Number of iterations")
        # plt.ylabel("Loss")
        # plt.legend(loc = "upper")
        # name = "graphs/classification_loss.png"
        # plt.savefig(name)

model_CT.eval()
model_D.eval()
model_MSI.eval()
model_IT.eval()

total_loss = [0, 0, 0, 0]
correct = [0, 0, 0, 0]

with torch.no_grad():
    for i in range(valid_data.shape[0]):
        line = torch.tensor(valid_data[i], dtype=torch.float).to(device)
        outputs_CT = model_CT(line)
        outputs_D = model_D(line)
        outputs_MSI = model_MSI(line)
        outputs_IT = model_IT(line)

        label_CT = torch.tensor(valid_labels['CRASH_TYPE'][i], dtype=torch.float).to(device)
        loss_CT = criterion_CT(outputs_CT, label_CT)

        label_D = torch.tensor(valid_labels['DAMAGE'][i], dtype=torch.float).to(device)
        loss_D = criterion_D(outputs_D, label_D)

        label_MSI = torch.tensor(valid_labels['MOST_SEVERE_INJURY'][i], dtype=torch.float).to(device)
        loss_MSI = criterion_MSI(outputs_MSI, label_MSI)

        label_IT = torch.tensor(valid_labels['INJURIES_TOTAL'][i], dtype=torch.float).to(device)
        loss_IT = criterion_IT(outputs_IT, label_IT)

        total_loss[0] += loss_CT.item()
        total_loss[1] += loss_D.item()
        total_loss[2] += loss_MSI.item()
        total_loss[3] += loss_IT.item()

        predicted_CT = torch.sigmoid(outputs_CT)
        predicted_D = torch.sigmoid(outputs_D) 
        predicted_MSI = torch.sigmoid(outputs_MSI)
        predicted_IT = torch.sigmoid(outputs_IT)

        # print(torch.argmax(label_CT, dim=0))

        correct[0] += (torch.argmax(predicted_CT, dim=0) == torch.argmax(label_CT, dim=0)).sum().item()
        correct[1] += (torch.argmax(predicted_D, dim=0) == torch.argmax(label_D, dim=0)).sum().item()
        correct[2] += (torch.argmax(predicted_MSI, dim=0) == torch.argmax(label_MSI, dim=0)).sum().item()
        correct[3] += (torch.argmax(predicted_IT, dim=0) == torch.argmax(label_IT, dim=0)).sum().item()

avg_loss = [tl / valid_data.shape[0] for tl in total_loss]
accuracy = [c / valid_data.shape[0] for c in correct]

        # CT_v.append(accuracy[0])
        # D_v.append(accuracy[1])
        # MSI_v.append(accuracy[2])
        # IT_v.append(accuracy[3])

print("Validation Loss and Accuracy: ")
print(avg_loss)
print(accuracy)

total_loss = [0, 0, 0, 0]
correct = [0, 0, 0, 0]

with torch.no_grad():
    for i in range(test_data.shape[0]):
        line = torch.tensor(test_data[i], dtype=torch.float).to(device)
        outputs_CT = model_CT(line)
        outputs_D = model_D(line)
        outputs_MSI = model_MSI(line)
        outputs_IT = model_IT(line)

        label_CT = torch.tensor(test_labels['CRASH_TYPE'][i], dtype=torch.float).to(device)
        loss_CT = criterion_CT(outputs_CT, label_CT)

        label_D = torch.tensor(test_labels['DAMAGE'][i], dtype=torch.float).to(device)
        loss_D = criterion_D(outputs_D, label_D)

        label_MSI = torch.tensor(test_labels['MOST_SEVERE_INJURY'][i], dtype=torch.float).to(device)
        loss_MSI = criterion_MSI(outputs_MSI, label_MSI)

        label_IT = torch.tensor(test_labels['INJURIES_TOTAL'][i], dtype=torch.float).to(device)
        loss_IT = criterion_IT(outputs_IT, label_IT)

        total_loss[0] += loss_CT.item()
        total_loss[1] += loss_D.item()
        total_loss[2] += loss_MSI.item()
        total_loss[3] += loss_IT.item()

        predicted_CT = torch.sigmoid(outputs_CT)
        predicted_D = torch.sigmoid(outputs_D) 
        predicted_MSI = torch.sigmoid(outputs_MSI)
        predicted_IT = torch.sigmoid(outputs_IT)

        # print(torch.argmax(label_CT, dim=0))

        correct[0] += (torch.argmax(predicted_CT, dim=0) == torch.argmax(label_CT, dim=0)).sum().item()
        correct[1] += (torch.argmax(predicted_D, dim=0) == torch.argmax(label_D, dim=0)).sum().item()
        correct[2] += (torch.argmax(predicted_MSI, dim=0) == torch.argmax(label_MSI, dim=0)).sum().item()
        correct[3] += (torch.argmax(predicted_IT, dim=0) == torch.argmax(label_IT, dim=0)).sum().item()

    avg_loss = [tl / test_data.shape[0] for tl in total_loss]
    accuracy = [c / test_data.shape[0] for c in correct]

            # CT_t.append(accuracy[0])
            # D_t.append(accuracy[1])
            # MSI_t.append(accuracy[2])
            # IT_t.append(accuracy[3])

print("Test Loss and Accuracy: ")
print(avg_loss)
print(accuracy)
            # axv.set_title("Validation Accuracy for different Learning Rate with Iteration")

    # axv1.scatter(np.arange(len(CT_v)), CT_v, label="CT_v")
    # axv1.plot(np.arange(len(CT_v)), CT_v)
    # axv1.scatter(np.arange(len(D_v)), D_v, label="D_v")
    # axv1.plot(np.arange(len(D_v)), D_v)
    # axv1.scatter(np.arange(len(MSI_v)), MSI_v, label="MSI_v")
    # axv1.plot(np.arange(len(MSI_v)), MSI_v)
    # axv1.scatter(np.arange(len(IT_v)), IT_v, label="IT_v")
    # axv1.plot(np.arange(len(IT_v)), IT_v)
    # axv1.set_xticklabels([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05])

    # axv2.scatter(np.arange(len(CT_t)), CT_t, label="CT_t")
    # axv2.plot(np.arange(len(CT_t)), CT_t)
    # axv2.scatter(np.arange(len(D_t)), D_t, label="D_t")
    # axv2.plot(np.arange(len(D_t)), D_t)
    # axv2.scatter(np.arange(len(MSI_t)), MSI_t, label="MSI_t")
    # axv2.plot(np.arange(len(MSI_t)), MSI_t)
    # axv2.scatter(np.arange(len(IT_t)), IT_t, label="IT_t")
    # axv2.plot(np.arange(len(IT_t)), IT_t)
    # axv2.set_xticklabels([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05])

    # axv1.set(xlabel = "Learning Rate", ylabel = "Validation Accuracy")
    # name = "graphs/validation_" + str(lr) + "_" + str(ba) + ".png"
    # f1.legend(loc = "lower right")
    # f1.savefig(name)

    # axv2.set(xlabel = "Learning Rate", ylabel = "Test Accuracy")
    # name = "graphs/test_" + str(lr) + "_" + str(ba) + ".png"
    # f2.legend(loc = "lower right")
    # f2.savefig(name)