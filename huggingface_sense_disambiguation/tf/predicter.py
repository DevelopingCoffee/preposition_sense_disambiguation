from tensorflow import keras
import numpy as np
import tensorflow as tf
from transformers import TFBertModel,  BertConfig, BertTokenizerFast
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.layers import Input
import csv


print("---------------------------------")
print("--------loading model------------")
print("---------------------------------")

print("getting preposition-definitons...")
with open('definitions.tsv') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    dict = dict(reader)

definitions = {}
for key in dict.keys():
    definitions[int(key)] = dict[key]


print('reconstruct model from scratch...')

model_name = 'bert-base-uncased'
max_length = 100
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False
transformer_model = TFBertModel.from_pretrained(model_name, config = config)

#adjust to fit our input
bert = transformer_model.layers[0]
input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
inputs = {'input_ids': input_ids}
bert_model = bert(inputs)[1]
dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
pooled_output = dropout(bert_model, training=False)
training_label = Dense(units=224, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='training_label')(pooled_output)
outputs = {'training_label': training_label}
model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')

# load weights into new model
print("loadeding weights...")
model.load_weights("model.h5")


print("---------------------------------")
print("---------tokenize input----------")
print("---------------------------------")

print("loading tokenizer...")
model_name = 'bert-base-uncased'
max_length = 100
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)

print("tokenize input...")
to_predict = ["I have not much money <head>on</head> the bank", "I am <head>in</head> big trouble"]
x = tokenizer(
        text=to_predict,
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding='max_length', 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = False,
    verbose = True)


print("---------------------------------")
print("--------generate output----------")
print("---------------------------------")

output = model.predict(x['input_ids'])
for i in range(len(to_predict)):
    print("sentence : {}     prediction : {}".format(str(to_predict[i]), str(definitions[np.argmax(output['training_label'][i])])))
