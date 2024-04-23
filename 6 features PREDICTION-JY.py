import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score

# Function to load data
def load_data():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    train_set = pd.read_csv('train_set.csv', encoding='ISO-8859-1')
    valid_set = pd.read_csv('valid_set.csv', encoding='ISO-8859-1')
    test_set = pd.read_csv('test_set.csv', encoding='ISO-8859-1')
    return train_set, valid_set, test_set

# Define the custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, feature, label_dict, tokenizer, max_len):
        self.data = dataframe
        self.feature = feature
        self.label_dict = label_dict
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
            'targets': torch.tensor(self.data.iloc[index][self.feature], dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)

# Function to train and evaluate models for each feature
def train_and_evaluate(feature, label_dict, tokenizer, max_len, device):
    train_set, valid_set, test_set = load_data()
    train_set[feature] = train_set[feature].replace(label_dict)
    valid_set[feature] = valid_set[feature].replace(label_dict)

    # Prepare datasets
    train_data = CustomDataset(train_set, feature, label_dict, tokenizer, max_len)
    valid_data = CustomDataset(valid_set, feature, label_dict, tokenizer, max_len)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False)

    # Model setup
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_dict))
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Training loop
    for epoch in range(1):  # Reduced number of epochs for brevity
        model.train()
        for data in train_loader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            targets = data['targets'].to(device)

            optimizer.zero_grad()
            outputs = model(ids, mask, labels=targets)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Testing and prediction extraction
    test_data = CustomDataset(test_set, feature, label_dict, tokenizer, max_len)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            outputs = model(ids, mask)
            logits = outputs.logits
            predicted = torch.argmax(logits, dim=1)
            predictions.extend(predicted.tolist())

    return [list(label_dict.keys())[list(label_dict.values()).index(pred)] for pred in predictions]

# Main script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features = {
    "WEATHER_CONDITION": {
    "CLEAR": 0, "CLOUDY/OVERCAST": 1, "FOG/SMOKE/HAZE": 2, 
    "FREEZING RAIN/DRIZZLE": 3, "OTHER": 4, "RAIN": 5, 
    "SLEET/HAIL": 6, "SNOW": 7, "UNKNOWN": 8
},
    "TRAFFICWAY_TYPE": {
    "NOT DIVIDED": 0, "DIVIDED - W/MEDIAN (NOT RAISED)": 1, "ONE-WAY": 2, 
    "FOUR WAY": 3, "PARKING LOT": 4, "DIVIDED - W/MEDIAN BARRIER": 5, 
    "OTHER": 6, "T-INTERSECTION": 7, "ALLEY": 8, "UNKNOWN":9, "FIVE POINT, OR MORE":10, "CENTER TURN LANE":11, "UNKNOWN INTERSECTION TYPE":12, "DRIVEWAY":13, "NOT REPORTED":14, "RAMP":15 
},
    "ROADWAY_SURFACE_COND":  {
    "WET": 0, "DRY": 1, "UNKNOWN": 2, 
    "OTHER": 3
}
}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 256
new_data = pd.DataFrame()

# Process each feature
for feature, label_dict in features.items():
    new_data[feature] = train_and_evaluate(feature, label_dict, tokenizer, max_len, device)

# Save results to CSV
new_data.to_csv('feature_predictions.csv', index=False)
