from collections import Counter
import os

data_dir = "classified_dataset"

labels = []

for i in os.listdir(data_dir):
    if os.path.isdir(os.path.join(data_dir, i)):
        count = len(os.listdir(os.path.join(data_dir, i)))
        labels.extend([i] * count)



class_counts = Counter(labels)
print(class_counts)