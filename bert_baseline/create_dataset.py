import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class CarCrash(Dataset):
    def __init__(self, type='train', class_type='CRASH_TYPE'):
        super().__init__()
        
        self.col_names = ['TRAFFIC_CONTROL_DEVICE',
                          'WEATHER_CONDITION',
                          'LIGHTING_CONDITION',
                          'ALIGNMENT',
                          'ROADWAY_SURFACE_COND',
                          'POSTED_SPEED_LIMIT',
                          'TRAFFICWAY_TYPE',
                          'CRASH_TYPE',
                          'DAMAGE',
                          'MOST_SEVERE_INJURY',
                          'INJURIES_TOTAL',
                          'narratives']
        
        self.classes = {'CRASH_TYPE': ['NO INJURY / DRIVE AWAY', 'INJURY AND / OR TOW DUE TO CRASH'],
                   'DAMAGE': ['OVER $1,500', '$501 - $1,500', '$500 OR LESS'],
                   'MOST_SEVERE_INJURY': ['NO INDICATION OF INJURY', 'INCAPACITATING INJURY', 'NONINCAPACITATING INJURY', 'REPORTED, NOT EVIDENT', 'FATAL'],
                   'INJURIES_TOTAL': ['0', '1', '2', '3', '4']}
        
        self.data_x = []
        self.data_y = []
        file_name = 'dataset/' + type + '_set.csv'
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.data_x.append(row[-1].lower())
                self.data_y.append(self.classes[class_type].index(row[self.col_names.index(class_type)]))
    
    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]
    
    def __len__(self):
        return len(self.data_x)
