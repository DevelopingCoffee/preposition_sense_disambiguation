# Load Huggingface transformers
from transformers import TFBertModel, BertConfig, BertTokenizerFast
import simplejson 
# Then what you need from tensorflow.keras
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
# And pandas for data import + sklearn because you allways need sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
import csv

data = pd.read_csv('data/training_data.tsv',engine='python', encoding='utf-8', error_bad_lines=False,sep="\t")
data = data[['sentence', 'label_id', 'definition']]
data = data.dropna()
data = data.groupby('label_id').filter(lambda x : len(x) > 1)
data['cat_label'] = pd.Categorical(data['label_id'])
data['training_label'] = data['cat_label'].cat.codes
data, data_test = train_test_split(data, test_size = 0.2, stratify = data[['training_label']])

definitions = {}
for row in data.itertuples():
    print(row.training_label, row.definition)
    if row.training_label not in definitions:
        definitions[row.training_label] = row.definition


with open('definitions.tsv', 'w') as tsv_file:  
    writer = csv.writer(tsv_file, delimiter='\t')
    for key, value in definitions.items():
       writer.writerow([key, value])


print(data)
