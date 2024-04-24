import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertForSequenceClassification

from torch.utils.data import Dataset, DataLoader
from create_dataset import CarCrash
import argparse
from tqdm import tqdm

import wandb

classes = {'CRASH_TYPE': ['NO INJURY / DRIVE AWAY', 'INJURY AND / OR TOW DUE TO CRASH'],
           'DAMAGE': ['OVER $1,500', '$501 - $1,500', '$500 OR LESS'],
           'MOST_SEVERE_INJURY': ['NO INDICATION OF INJURY', 'INCAPACITATING INJURY', 'NONINCAPACITATING INJURY', 'REPORTED, NOT EVIDENT', 'FATAL'],
           'INJURIES_TOTAL': ['0', '1', '2', '3', '4']}

epochs = 10
batch_size = 8
learning_rate = 5e-5
class_type = 'CRASH_TYPE'

model_name = "bert-base-uncased"
# model_name = "textattack/bert-base-uncased-yelp-polarity"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(classes[class_type]))
nn.init.normal_(model.classifier.weight)

config = {'lr': learning_rate, 'epochs': epochs, 'batch_size': batch_size, 'class_type': class_type}
wandb.init(project='cs577_final_project', name=f'bert_baseline_{class_type}', config=config)

train_dataset = CarCrash(type='train', class_type=class_type)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

valid_dataset = CarCrash(type='valid', class_type=class_type)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

test_dataset = CarCrash(type='test', class_type=class_type)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training step
max_valid_acc = 0
for ep in range(epochs):
    avg_loss = 0
    cnt = 0
    for idx, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        inputs = tokenizer(x, return_tensors="pt", truncation=True, padding=True)
        labels = torch.nn.functional.one_hot(y[None, :], num_classes=len(classes[class_type])).squeeze(0).to(torch.float)
        loss = model(**inputs, labels=labels).loss
        
        avg_loss += loss
        cnt += 1
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = avg_loss / cnt
    
    # valid step
    acc = 0
    for idx, (x, y) in enumerate(valid_dataloader):
        inputs = tokenizer(x, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class_ids = torch.argmax(logits, dim=1)
        for i in range(len(y)):
            if y[i] == predicted_class_ids[i]:
                acc += 1
    acc = acc / len(valid_dataset)
    if acc >= max_valid_acc:
        max_valid_acc = acc
        torch.save(model.state_dict(), f'{class_type}_best_model.pth')
    print(f'Epoch {ep}: avg_loss = {avg_loss:.3f}, valid_acc = {acc:.3f}')
    wandb.log({'training_loss': avg_loss, 'valid_acc': acc})
    
acc = 0
model.load_state_dict(torch.load(f'{class_type}_best_model.pth'))
for idx, (x, y) in enumerate(test_dataloader):
    inputs = tokenizer(x, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_ids = torch.argmax(logits, dim=1)
    for i in range(len(y)):
        if y[i] == predicted_class_ids[i]:
            acc += 1
acc = acc / len(test_dataset)
print(f'test_acc = {acc:.3f}')
    
