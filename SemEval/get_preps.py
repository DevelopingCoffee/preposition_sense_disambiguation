import xml.etree.ElementTree as ET
import csv
import io
import os
import random
from collections import defaultdict

dataset = []
all_labels = set()

data_dir = 'Train/Source/'
def_dir = 'Tratz Variations/definitions/'

label_id = -1
sentence_id = 0
skipped_examples = 0


skipped = defaultdict(list)

preps = []
# Go thorugh all files
for file in os.scandir(data_dir):
    data_file = str(file)[11:-2]
    preps.append(data_file.split('.')[0][3:])
print(preps)
