import tensorflow as tf
import pandas as pd
import numpy as np
from transformers import TFBertModel,  BertConfig
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import wandb
from wandb.keras import WandbCallback
import pickle

data = pd.read_csv('../data/training_data.tsv',engine='python', encoding='utf-8', error_bad_lines=False,sep="\t")
data = data[['sentence', 'label_id']]
data = data.dropna()
data = data.groupby('label_id').filter(lambda x : len(x) > 1)
data['cat_label'] = pd.Categorical(data['label_id'])
data['training_label'] = data['cat_label'].cat.codes
#data, data_test = train_test_split(data, test_size = 0.2, stratify = data[['training_label']])

y_training_label = to_categorical(data['training_label']) 


def create_model():

    model_name = 'bert-base-uncased'
    max_length = 100
    bert_config = BertConfig.from_pretrained(model_name)
    bert_config.output_hidden_states = False
    print("loading pretrained bert/tokenizer...")
    transformer_model = TFBertModel.from_pretrained(model_name, config = bert_config)

    print("construct network...")
    bert = transformer_model.layers[0]
    input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
    inputs = {'input_ids': input_ids}
    bert_model = bert(inputs)[1]
    dropout = Dropout(bert_config.hidden_dropout_prob, name='pooled_output')
    pooled_output = dropout(bert_model, training=False)
    training_label = Dense(units=len(data.training_label.value_counts()), kernel_initializer=TruncatedNormal(stddev=bert_config.initializer_range), name='training_label')(pooled_output)
    outputs = {'training_label': training_label}
    model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')
    return model


def train():
    # Initialize wandb with a sample project name
    wandb.init(project="hyperparameter-sweeps-partI")
    config = wandb.config

    print("holy")
    model = create_model()
    print("shit")

    with open(r"examplePickle", "rb") as input_file:
        x = pickle.load(input_file)
    
    optimizer = Adam(learning_rate=wandb.config.learning_rate, epsilon=1e-08, decay=0.01, clipnorm=1.0)
    
    loss = {'training_label': CategoricalCrossentropy(from_logits = True)}
    metric = {'training_label': CategoricalAccuracy('accuracy')}
    
    print("compiling model...")
    model.compile(optimizer = optimizer, loss = loss, metrics = metric)
    print(x)
    print("fit")
    print(y_training_label)

    print(len(x['input_ids']))
    print(len(y_training_label))
    history = model.fit(
        x={'input_ids': x['input_ids']},
        y={'training_label': y_training_label},
        validation_split=0.2,
        batch_size=64,
        epochs=10)

sweep_config = {
    'method': 'grid',
    'parameters': {
        'learning_rate': {
            'values': [5e-5, 3e-5, 2e-5]
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
    }
}
sweep_id = wandb.sweep(sweep_config)

wandb.agent(sweep_id, function=train)
