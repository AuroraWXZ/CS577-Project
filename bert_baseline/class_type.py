import csv
import matplotlib.pyplot as plt

col_names = ['TRAFFIC_CONTROL_DEVICE',
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
label_names = ['CRASH_TYPE', 'DAMAGE', 'MOST_SEVERE_INJURY', 'INJURIES_TOTAL']
feature_names = ['TRAFFIC_CONTROL_DEVICE', 'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'ALIGNMENT', 'ROADWAY_SURFACE_COND', 'POSTED_SPEED_LIMIT', 'TRAFFICWAY_TYPE']

classes = {'CRASH_TYPE': ['NO INJURY / DRIVE AWAY', 'INJURY AND / OR TOW DUE TO CRASH'],
           'DAMAGE': ['OVER $1,500', '$501 - $1,500', '$500 OR LESS'],
           'MOST_SEVERE_INJURY': ['NO INDICATION OF INJURY', 'INCAPACITATING INJURY', 'NONINCAPACITATING INJURY', 'REPORTED, NOT EVIDENT', 'FATAL'],
           'INJURIES_TOTAL': ['0', '1', '2', '3', '4']}

def get_all_data():
    all_data = []
    with open('dataset/train_set.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            all_data.append(row)

    with open('dataset/valid_set.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            all_data.append(row)
            
    with open('dataset/test_set.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            all_data.append(row)
    return all_data

def get_class_type():
    all_data = get_all_data()

    for c in label_names:
        class_list = []
        for d in all_data:
            y = d[col_names.index(c)]
            if y not in class_list:
                class_list.append(y)
        print(class_list)
        
def get_feature_type():
    all_data = get_all_data()
    feature_types = {}
    for c in feature_names:
        class_list = []
        for d in all_data:
            y = d[col_names.index(c)]
            if y not in class_list:
                class_list.append(y)
        feature_types[c] = class_list
    return feature_types
        

def get_overlap_coef():
    max_overlap_coefficient = 0
    min_overlap_coefficient = 1
    coef = []
    all_data = get_all_data()
    for i in range(len(all_data)):
        for j in range(i+1, len(all_data)):
            set_a = set(all_data[i][-1].lower().split(' '))
            set_b = set(all_data[j][-1].lower().split(' '))
            intersection = set_a.intersection(set_b)
            overlap_coefficient = len(intersection) / min(len(set_a), len(set_b))
            coef.append(overlap_coefficient)
            max_overlap_coefficient = max(overlap_coefficient, max_overlap_coefficient)
            min_overlap_coefficient = min(overlap_coefficient, min_overlap_coefficient)
    print("max overlap coefficients:", round(max_overlap_coefficient, 4))
    print("min overlap coefficients:", round(min_overlap_coefficient, 4))
    
    plt.hist(coef, bins=30, density=True)
    plt.xlabel('overlap coeffiecient')
    plt.ylabel('density')
    plt.savefig('coefficient_nums.png')
    
def get_length_hist():
    all_data = get_all_data()
    length = []
    for i in range(len(all_data)):
        length.append(len(all_data[i][-1].split(' ')))
    
    plt.hist(length, density=True)
    plt.xlabel('scenario_length')
    plt.ylabel('density')
    plt.savefig("sentence_length.png")
    
def get_label_num():
    all_data = get_all_data()

    plt.figure(figsize=[10,10])
    for idx, c in enumerate(label_names):
        class_nums = {}
        for name in classes[c]:
            class_nums[name] = 0
        for d in all_data:
            y = d[col_names.index(c)]
            class_nums[y] += 1
        
        labels = []
        height = []
        cnt = 0
        for key in class_nums.keys():
            labels.append(str(cnt))
            height.append(class_nums[key])
            cnt += 1
        
        plt.subplot(2, 2, idx+1)
        plt.bar(labels, height)
        # plt.xticks(rotation=30, ha='right')
        plt.xlabel(c)
        plt.ylabel('num')
    
    plt.savefig('label_num.png')
    
def get_feature_num():
    all_data = get_all_data()
    feature_types = get_feature_type()

    plt.figure(figsize=[15,15])
    for idx, c in enumerate(feature_names):
        class_nums = {}
        for name in feature_types[c]:
            class_nums[name] = 0
        for d in all_data:
            y = d[col_names.index(c)]
            class_nums[y] += 1
        
        labels = []
        height = []
        cnt = 0
        for key in class_nums.keys():
            labels.append(str(cnt))
            height.append(class_nums[key])
            cnt += 1
        
        plt.subplot(3, 3, idx+1)
        plt.bar(labels, height)
        plt.xticks(rotation=30, ha='right')
        plt.xlabel(c)
        plt.ylabel('num')
    
    plt.savefig('feature_num.png')
    
    
if __name__ == "__main__":
    # get_overlap_coef()
    get_length_hist()
    # get_label_num()
    # get_feature_num()