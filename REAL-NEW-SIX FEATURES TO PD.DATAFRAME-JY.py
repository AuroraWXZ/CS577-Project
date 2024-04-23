import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load data
train_set = pd.read_csv('train_set.csv', encoding='ISO-8859-1')
valid_set = pd.read_csv('valid_set.csv', encoding='ISO-8859-1')
test_set = pd.read_csv('test_set.csv', encoding='ISO-8859-1')

# Define weather conditions and map to numeric labels
LIGHTING_CONDITION = {
    "DARKNESS, LIGHTED ROAD": 0, "DAYLIGHT": 1, "UNKNOWN": 2, 
    "DARKNESS": 3, "DUSK": 4, "DAWN": 5
}
train_set['LIGHTING_label'] = train_set['LIGHTING_CONDITION'].replace(LIGHTING_CONDITION)
valid_set['LIGHTING_label'] = valid_set['LIGHTING_CONDITION'].replace(LIGHTING_CONDITION)

# Dataset preparation for training and validation
class WeatherConditionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        narrative = str(self.data.iloc[index]['narratives'])
        narrative = " ".join(narrative.split())
        inputs = self.tokenizer.encode_plus(
            narrative,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'targets': torch.tensor(self.data.iloc[index]['LIGHTING_label'], dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create data loaders for training and validation
train_data = WeatherConditionDataset(train_set, tokenizer, max_len=256)
valid_data = WeatherConditionDataset(valid_set, tokenizer, max_len=256)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False)

# Model setup
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(LIGHTING_CONDITION))
model.to('cpu')

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    for data in train_loader:
        ids = data['ids'].to('cpu')
        mask = data['mask'].to('cpu')
        targets = data['targets'].to('cpu')

        model.zero_grad()
        outputs = model(ids, mask, labels=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f'Training Loss after epoch {epoch + 1}: {total_loss / len(train_loader)}')

# Validation loop
model.eval()
val_predictions, val_true = [], []
with torch.no_grad():
    for data in valid_loader:
        ids = data['ids'].to('cpu')
        mask = data['mask'].to('cpu')
        outputs = model(ids, mask)
        logits = outputs.logits
        predicted = torch.argmax(logits, dim=1)
        val_predictions.extend(predicted.tolist())
        val_true.extend(data['targets'].tolist())

val_accuracy = accuracy_score(val_true, val_predictions)
print(f'Validation Accuracy: {val_accuracy}')

# Adjusted class for test data handling without labels
class TestData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        narrative = str(self.data.iloc[index]['narratives'])
        narrative = " ".join(narrative.split())
        inputs = self.tokenizer.encode_plus(
            narrative,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)

# Prepare test data
test_data = TestData(test_set, tokenizer, max_len=256)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Generate predictions for test data
test_predictions = []
with torch.no_grad():
    for data in test_loader:
        ids = data['ids'].to('cpu')
        mask = data['mask'].to('cpu')
        outputs = model(ids, mask)
        logits = outputs.logits
        predicted = torch.argmax(logits, dim=1)
        test_predictions.extend(predicted.tolist())

# Map numeric labels back to weather conditions
test_conditions = [list(LIGHTING_CONDITION.keys())[list(LIGHTING_CONDITION.values()).index(pred)] for pred in test_predictions]

# Save predictions to CSV
test_set['Predicted_LIGHTING_CONDITION'] = test_conditions
test_set.to_csv('Predicted_LIGHTING_CONDITION.csv', index=False)


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load data
train_set = pd.read_csv('train_set.csv', encoding='ISO-8859-1')
valid_set = pd.read_csv('valid_set.csv', encoding='ISO-8859-1')
test_set = pd.read_csv('test_set.csv', encoding='ISO-8859-1')

# Define weather conditions and map to numeric labels
ROADWAY_SURFACE_COND = {
    "WET": 0, "DRY": 1, "UNKNOWN": 2, 
    "OTHER": 3
}
train_set['ROADWAT_labels'] = train_set['ROADWAY_SURFACE_COND'].replace(ROADWAY_SURFACE_COND)
valid_set['ROADWAT_labels'] = valid_set['ROADWAY_SURFACE_COND'].replace(ROADWAY_SURFACE_COND)

# Dataset preparation for training and validation
class WeatherConditionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        narrative = str(self.data.iloc[index]['narratives'])
        narrative = " ".join(narrative.split())
        inputs = self.tokenizer.encode_plus(
            narrative,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'targets': torch.tensor(self.data.iloc[index]['ROADWAT_labels'], dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create data loaders for training and validation
train_data = WeatherConditionDataset(train_set, tokenizer, max_len=256)
valid_data = WeatherConditionDataset(valid_set, tokenizer, max_len=256)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False)

# Model setup
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(ROADWAY_SURFACE_COND))
model.to('cpu')

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    for data in train_loader:
        ids = data['ids'].to('cpu')
        mask = data['mask'].to('cpu')
        targets = data['targets'].to('cpu')

        model.zero_grad()
        outputs = model(ids, mask, labels=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f'Training Loss after epoch {epoch + 1}: {total_loss / len(train_loader)}')

# Validation loop
model.eval()
val_predictions, val_true = [], []
with torch.no_grad():
    for data in valid_loader:
        ids = data['ids'].to('cpu')
        mask = data['mask'].to('cpu')
        outputs = model(ids, mask)
        logits = outputs.logits
        predicted = torch.argmax(logits, dim=1)
        val_predictions.extend(predicted.tolist())
        val_true.extend(data['targets'].tolist())

val_accuracy = accuracy_score(val_true, val_predictions)
print(f'Validation Accuracy: {val_accuracy}')

# Adjusted class for test data handling without labels
class TestData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        narrative = str(self.data.iloc[index]['narratives'])
        narrative = " ".join(narrative.split())
        inputs = self.tokenizer.encode_plus(
            narrative,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)

# Prepare test data
test_data = TestData(test_set, tokenizer, max_len=256)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Generate predictions for test data
test_predictions = []
with torch.no_grad():
    for data in test_loader:
        ids = data['ids'].to('cpu')
        mask = data['mask'].to('cpu')
        outputs = model(ids, mask)
        logits = outputs.logits
        predicted = torch.argmax(logits, dim=1)
        test_predictions.extend(predicted.tolist())

# Map numeric labels back to weather conditions
test_conditions = [list(ROADWAY_SURFACE_COND.keys())[list(ROADWAY_SURFACE_COND.values()).index(pred)] for pred in test_predictions]

# Save predictions to CSV
test_set['Predicted_ROADWAY_SURFACE_COND'] = test_conditions
test_set.to_csv('Predicted_ROADWAY_SURFACE_COND.csv', index=False)


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score

# Load data
def load_data(filepath):
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    return df

train_df = load_data('train_set.csv')
valid_df = load_data('valid_set.csv')
test_df = load_data('test_set.csv')

# Define the dictionary to map speed limits to numeric labels
POSTED_SPEED_LIMIT = {
    "0": 0, "5": 1, "10": 2, 
    "15": 3, "20": 4, "25": 5, 
    "30": 6, "35": 7, "40": 8, "45": 9, "50": 10,
    "55": 11, "60": 12, "65": 13, "70": 14, "75": 15, "80": 16
}

# Apply mapping
train_df['POSTED_SPEED_LIMIT_labels'] = train_df['POSTED_SPEED_LIMIT'].astype(str).map(POSTED_SPEED_LIMIT)
valid_df['POSTED_SPEED_LIMIT_labels'] = valid_df['POSTED_SPEED_LIMIT'].astype(str).map(POSTED_SPEED_LIMIT)
test_df['POSTED_SPEED_LIMIT_labels'] = test_df['POSTED_SPEED_LIMIT'].astype(str).map(POSTED_SPEED_LIMIT)

# Dataset class
class SpeedLimitDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=256):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        narrative = self.data.iloc[index]['narratives']
        label = self.data.iloc[index]['POSTED_SPEED_LIMIT_labels']
        encoding = self.tokenizer.encode_plus(
            narrative,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialize tokenizer and create datasets and loaders
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = SpeedLimitDataset(train_df, tokenizer)
valid_dataset = SpeedLimitDataset(valid_df, tokenizer)
test_dataset = SpeedLimitDataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Setup model and optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(POSTED_SPEED_LIMIT))
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training and validation
def train_and_evaluate(model, train_loader, valid_loader, optimizer):
    model.train()
    for epoch in range(5):  # Example: train for 3 epochs
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Training Loss: {total_loss / len(train_loader)}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        correct_preds = 0
        total = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_val_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                correct_preds += (preds == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        val_accuracy = correct_preds / total
        print(f"Validation Loss: {total_val_loss / len(valid_loader)}, Accuracy: {val_accuracy}")

train_and_evaluate(model, train_loader, valid_loader, optimizer)

# Testing and saving results
def test_and_save(model, loader, filepath):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            inputs = {
                'input_ids': batch['input_ids'].to(model.device),
                'attention_mask': batch['attention_mask'].to(model.device)
            }
            outputs = model(**inputs)
            predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
    test_df['Predicted_Speed_Limit'] = predictions
    test_df.to_csv(filepath, index=False)
    print(f"Predictions saved to {filepath}")

test_and_save(model, test_loader, 'Predicted_Speed_Limit.csv')


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score
from torch.optim import AdamW

# Load data
train_set = pd.read_csv('train_set.csv', encoding='ISO-8859-1')
valid_set = pd.read_csv('valid_set.csv', encoding='ISO-8859-1')
test_set = pd.read_csv('test_set.csv', encoding='ISO-8859-1')

# Define a dataset class
class TrafficNarrativesDataset(Dataset):
    def __init__(self, narratives, labels, tokenizer):
        self.narratives = narratives
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.narratives)

    def __getitem__(self, idx):
        narrative = str(self.narratives[idx])
        label = self.labels[idx]
        tokenized_data = self.tokenizer(narrative, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        input_ids = tokenized_data['input_ids'][0]
        attention_mask = tokenized_data['attention_mask'][0]
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

# Prepare the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Map labels to integers
label_dict = {"NO CONTROLS": 0, "NO PASSING": 1, "OTHER": 2, "OTHER WARNING SIGN": 3, "RAILROAD CROSSING GATE": 4, "STOP SIGN/FLASHER": 5, "TRAFFIC SIGNAL": 6, "UNKNOWN": 7, "YIELD": 8}
train_labels = train_set['TRAFFIC_CONTROL_DEVICE'].map(label_dict)
valid_labels = valid_set['TRAFFIC_CONTROL_DEVICE'].map(label_dict)

# Prepare datasets and dataloaders
train_dataset = TrafficNarrativesDataset(train_set['narratives'], train_labels, tokenizer)
valid_dataset = TrafficNarrativesDataset(valid_set['narratives'], valid_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

# Training function
def train(model, dataloader):
    model.train()
    total_loss, total_accuracy = 0, 0
    for batch in dataloader:
        batch = [b.to(device) for b in batch]
        inputs, masks, labels = batch
        model.zero_grad()
        outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1).flatten()
        total_accuracy += accuracy_score(labels.cpu(), preds.cpu())
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    return avg_loss, avg_accuracy

# Validation function
def validate(model, dataloader):
    model.eval()
    total_loss, total_accuracy = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            batch = [b.to(device) for b in batch]
            inputs, masks, labels = batch
            outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1).flatten()
            total_accuracy += accuracy_score(labels.cpu(), preds.cpu())
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    return avg_loss, avg_accuracy

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training and validation loop
for epoch in range(5):  # Number of epochs
    train_loss, train_accuracy = train(model, train_loader)
    valid_loss, valid_accuracy = validate(model, valid_loader)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss}, Train Accuracy = {train_accuracy}")
    print(f"Epoch {epoch+1}: Valid Loss = {valid_loss}, Valid Accuracy = {valid_accuracy}")

# Testing and saving results
test_narratives = test_set['narratives']
test_dataset = TrafficNarrativesDataset(test_narratives, [0]*len(test_narratives), tokenizer)  # Dummy labels
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

def predict(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, masks, _ = [b.to(device) for b in batch]
            outputs = model(input_ids=inputs, attention_mask=masks)
            preds = torch.argmax(outputs.logits, dim=1).flatten()
            predictions.extend(preds.cpu().numpy())
    return predictions

predicted_labels = predict(model, test_loader)
test_set['TRAFFIC_CONTROL_DEVICE'] = [list(label_dict.keys())[label] for label in predicted_labels]
test_set.to_csv('Predicted_TRAFFIC_CONTROL_DEVICE.csv', index=False)


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load data
train_set = pd.read_csv('train_set.csv', encoding='ISO-8859-1')
valid_set = pd.read_csv('valid_set.csv', encoding='ISO-8859-1')
test_set = pd.read_csv('test_set.csv', encoding='ISO-8859-1')

# Define weather conditions and map to numeric labels
TRAFFICWAY_TYPE = {
    "NOT DIVIDED": 0, "DIVIDED - W/MEDIAN (NOT RAISED)": 1, "ONE-WAY": 2, 
    "FOUR WAY": 3, "PARKING LOT": 4, "DIVIDED - W/MEDIAN BARRIER": 5, 
    "OTHER": 6, "T-INTERSECTION": 7, "ALLEY": 8, "UNKNOWN":9, "FIVE POINT, OR MORE":10, "CENTER TURN LANE":11, "UNKNOWN INTERSECTION TYPE":12, "DRIVEWAY":13, "NOT REPORTED":14, "RAMP":15 
}
train_set['TRAFFICWAY_TYPE_labels'] = train_set['TRAFFICWAY_TYPE'].replace(TRAFFICWAY_TYPE)
valid_set['TRAFFICWAY_TYPE_labels'] = valid_set['TRAFFICWAY_TYPE'].replace(TRAFFICWAY_TYPE)

# Dataset preparation for training and validation
class WeatherConditionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        narrative = str(self.data.iloc[index]['narratives'])
        narrative = " ".join(narrative.split())
        inputs = self.tokenizer.encode_plus(
            narrative,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'targets': torch.tensor(self.data.iloc[index]['TRAFFICWAY_TYPE_labels'], dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create data loaders for training and validation
train_data = WeatherConditionDataset(train_set, tokenizer, max_len=256)
valid_data = WeatherConditionDataset(valid_set, tokenizer, max_len=256)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False)

# Model setup
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(TRAFFICWAY_TYPE))
model.to('cpu')

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    for data in train_loader:
        ids = data['ids'].to('cpu')
        mask = data['mask'].to('cpu')
        targets = data['targets'].to('cpu')

        model.zero_grad()
        outputs = model(ids, mask, labels=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f'Training Loss after epoch {epoch + 1}: {total_loss / len(train_loader)}')

# Validation loop
model.eval()
val_predictions, val_true = [], []
with torch.no_grad():
    for data in valid_loader:
        ids = data['ids'].to('cpu')
        mask = data['mask'].to('cpu')
        outputs = model(ids, mask)
        logits = outputs.logits
        predicted = torch.argmax(logits, dim=1)
        val_predictions.extend(predicted.tolist())
        val_true.extend(data['targets'].tolist())

val_accuracy = accuracy_score(val_true, val_predictions)
print(f'Validation Accuracy: {val_accuracy}')

# Adjusted class for test data handling without labels
class TestData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        narrative = str(self.data.iloc[index]['narratives'])
        narrative = " ".join(narrative.split())
        inputs = self.tokenizer.encode_plus(
            narrative,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)

# Prepare test data
test_data = TestData(test_set, tokenizer, max_len=256)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Generate predictions for test data
test_predictions = []
with torch.no_grad():
    for data in test_loader:
        ids = data['ids'].to('cpu')
        mask = data['mask'].to('cpu')
        outputs = model(ids, mask)
        logits = outputs.logits
        predicted = torch.argmax(logits, dim=1)
        test_predictions.extend(predicted.tolist())

# Map numeric labels back to weather conditions
test_conditions = [list(TRAFFICWAY_TYPE.keys())[list(TRAFFICWAY_TYPE.values()).index(pred)] for pred in test_predictions]

# Save predictions to CSV
test_set['Predicted_TRAFFICWAY_TYPE'] = test_conditions
test_set.to_csv('Predicted_TRAFFICWAY_TYPE.csv', index=False)


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load data
train_set = pd.read_csv('train_set.csv', encoding='ISO-8859-1')
valid_set = pd.read_csv('valid_set.csv', encoding='ISO-8859-1')
test_set = pd.read_csv('test_set.csv', encoding='ISO-8859-1')

# Define weather conditions and map to numeric labels
weather_conditions = {
    "CLEAR": 0, "CLOUDY/OVERCAST": 1, "FOG/SMOKE/HAZE": 2, 
    "FREEZING RAIN/DRIZZLE": 3, "OTHER": 4, "RAIN": 5, 
    "SLEET/HAIL": 6, "SNOW": 7, "UNKNOWN": 8
}
train_set['weather_label'] = train_set['WEATHER_CONDITION'].replace(weather_conditions)
valid_set['weather_label'] = valid_set['WEATHER_CONDITION'].replace(weather_conditions)

# Dataset preparation for training and validation
class WeatherConditionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        narrative = str(self.data.iloc[index]['narratives'])
        narrative = " ".join(narrative.split())
        inputs = self.tokenizer.encode_plus(
            narrative,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'targets': torch.tensor(self.data.iloc[index]['weather_label'], dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create data loaders for training and validation
train_data = WeatherConditionDataset(train_set, tokenizer, max_len=256)
valid_data = WeatherConditionDataset(valid_set, tokenizer, max_len=256)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False)

# Model setup
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(weather_conditions))
model.to('cpu')

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    for data in train_loader:
        ids = data['ids'].to('cpu')
        mask = data['mask'].to('cpu')
        targets = data['targets'].to('cpu')

        model.zero_grad()
        outputs = model(ids, mask, labels=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f'Training Loss after epoch {epoch + 1}: {total_loss / len(train_loader)}')

# Validation loop
model.eval()
val_predictions, val_true = [], []
with torch.no_grad():
    for data in valid_loader:
        ids = data['ids'].to('cpu')
        mask = data['mask'].to('cpu')
        outputs = model(ids, mask)
        logits = outputs.logits
        predicted = torch.argmax(logits, dim=1)
        val_predictions.extend(predicted.tolist())
        val_true.extend(data['targets'].tolist())

val_accuracy = accuracy_score(val_true, val_predictions)
print(f'Validation Accuracy: {val_accuracy}')

# Adjusted class for test data handling without labels
class TestData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        narrative = str(self.data.iloc[index]['narratives'])
        narrative = " ".join(narrative.split())
        inputs = self.tokenizer.encode_plus(
            narrative,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)

# Prepare test data
test_data = TestData(test_set, tokenizer, max_len=256)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Generate predictions for test data
test_predictions = []
with torch.no_grad():
    for data in test_loader:
        ids = data['ids'].to('cpu')
        mask = data['mask'].to('cpu')
        outputs = model(ids, mask)
        logits = outputs.logits
        predicted = torch.argmax(logits, dim=1)
        test_predictions.extend(predicted.tolist())

# Map numeric labels back to weather conditions
test_conditions = [list(weather_conditions.keys())[list(weather_conditions.values()).index(pred)] for pred in test_predictions]

# Save predictions to CSV
test_set['Predicted_WEATHER_CONDITION'] = test_conditions
test_set.to_csv('Predicted_WEATHER_CONDITION.csv', index=False)


import pandas as pd

# Load each CSV file
lighting_df = pd.read_csv('Predicted_LIGHTING_CONDITION.csv')
roadway_df = pd.read_csv('Predicted_ROADWAY_SURFACE_COND.csv')
trafficway_df = pd.read_csv('Predicted_TRAFFICWAY_TYPE.csv')
weather_df = pd.read_csv('Predicted_WEATHER_CONDITION.csv')
speed_limit_df = pd.read_csv('Predicted_SPEED_LIMIT.csv')
traffic_control_df = pd.read_csv('Predicted_TRAFFIC_CONTROL_DEVICE.csv')

# Create the combined DataFrame with simplified column names
combined_df = pd.DataFrame({
    'LIGHTING_CONDITION': lighting_df['Predicted_LIGHTING_CONDITION'],
    'ROADWAY_SURFACE_COND': roadway_df['Predicted_ROADWAY_SURFACE_COND'],
    'TRAFFICWAY_TYPE': trafficway_df['Predicted_TRAFFICWAY_TYPE'],
    'WEATHER_CONDITION': weather_df['Predicted_WEATHER_CONDITION'],
    'POSTED_SPEED_LIMIT': speed_limit_df['Predicted_Speed_Limit'],
    'TRAFFIC_CONTROL_DEVICE': traffic_control_df['Predicted_Traffic_Control_Device']
})

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('Combined_Predictions.csv', index=False)
