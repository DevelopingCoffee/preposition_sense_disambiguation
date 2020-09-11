# hyperopt, transformers

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
from transformers import TFBertModel,  BertConfig, BertTokenizerFast
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
import sys


print("---------------------------------")
print("---------reading data------------")
print("---------------------------------")

data = pd.read_csv('/content/gdrive/My Drive/training_data.tsv',engine='python', encoding='utf-8', error_bad_lines=False,sep="\t")
#data = pd.read_csv('data/training_data.tsv',engine='python', encoding='utf-8', error_bad_lines=False,sep="\t")
data = data[['sentence', 'label_id']]
data = data.dropna()
data = data.groupby('label_id').filter(lambda x : len(x) > 1)
data['cat_label'] = pd.Categorical(data['label_id'])
data['training_label'] = data['cat_label'].cat.codes
data, eval_data = train_test_split(data, test_size = 0.2, stratify = data[['training_label']])

#model-settings
model_name = 'bert-base-uncased'
max_length = 100
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False

#tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
print("tokenize data...")
x = tokenizer(
    text=data['sentence'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = False,
    verbose = True)

y_training_label = to_categorical(data['training_label']) # produces out of 224 unique labels 314 labels ?????

space = {
            'batch_size' : hp.uniform('batch_size', 28,128),
            'learning_rate': hp.choice('learning_rate', [5e-5, 3e-5, 2e-5]),  # https://mccormickml.com/2019/07/22/BERT-fine-tuning/
            'epochs' :  10,
            'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
        }

def f_nn(params):   
    print ('Params testing: ', params)
    transformer_model = TFBertModel.from_pretrained(model_name, config = config)
    
    bert = transformer_model.layers[0]
    input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
    inputs = {'input_ids': input_ids}
    bert_model = bert(inputs)[1]
    dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
    pooled_output = dropout(bert_model, training=False)
    training_label = Dense(units=len(data.training_label.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='training_label')(pooled_output)
    outputs = {'training_label': training_label}
    model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')
    
    optimizer = Adam(learning_rate=5e-05, epsilon=1e-08, decay=0.01, clipnorm=1.0)
    loss = {'training_label': CategoricalCrossentropy(from_logits = True)}
    metric = {'training_label': CategoricalAccuracy('accuracy')}
    model.compile(optimizer = params['optimizer'], loss = loss, metrics = metric)
    
    
    history = model.fit(x={'input_ids': x['input_ids']}, y={'training_label': y_training_label}, validation_split=0.2, batch_size=int(params['batch_size']), epochs=params['epochs'])
    acc = history.history['val_acc']
    print('AUC:', acc)
    sys.stdout.flush() 
    return {'loss': -acc, 'status': STATUS_OK}

print("---------------------------------")
print("----------hyper opt--------------")
print("---------------------------------")

trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
print('best: ', best)
