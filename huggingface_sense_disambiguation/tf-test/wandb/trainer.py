### import transformers and simplejson

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


print("---------------------------------")
print("---------reading data------------")
print("---------------------------------")

#data = pd.read_csv('/content/gdrive/My Drive/training_data.tsv',engine='python', encoding='utf-8', error_bad_lines=False,sep="\t")
data = pd.read_csv('../data/training_data.tsv',engine='python', encoding='utf-8', error_bad_lines=False,sep="\t")
data = data[['sentence', 'label_id']]
data = data.dropna()
data = data.groupby('label_id').filter(lambda x : len(x) > 1)
data['cat_label'] = pd.Categorical(data['label_id'])
data['training_label'] = data['cat_label'].cat.codes
data, data_test = train_test_split(data, test_size = 0.2, stratify = data[['training_label']])

print(data)

print("---------------------------------")
print("-------configure Bert------------")
print("---------------------------------")

print("loading pretrained bert/tokenizer...")
model_name = 'bert-base-uncased'
max_length = 100
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
transformer_model = TFBertModel.from_pretrained(model_name, config = config)

print("construct network...")
bert = transformer_model.layers[0]
input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
inputs = {'input_ids': input_ids}
bert_model = bert(inputs)[1]
dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
pooled_output = dropout(bert_model, training=False)
training_label = Dense(units=len(data.training_label.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='training_label')(pooled_output)
outputs = {'training_label': training_label}
model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')
model.summary()

# Set an optimizer
optimizer = Adam(
    learning_rate=5e-05,
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)

# Set loss and metrics
loss = {'training_label': CategoricalCrossentropy(from_logits = True)}
metric = {'training_label': CategoricalAccuracy('accuracy')}

# Compile the model
print("compiling model...")
model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)

# Ready output data for the model
y_training_label = to_categorical(data['training_label']) # produces out of 224 unique labels 314 labels ?????
# Tokenize the input (takes some time)

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
print("---------------------------------------------------------")
print(len(y_training_label))
print(len(x['input_ids']))

print("---------------------------------")
print("--------fine-tune bert-----------")
print("---------------------------------")

history = model.fit(
    x={'input_ids': x['input_ids']},
    y={'training_label': y_training_label},
    validation_split=0.2,
    batch_size=64,
    epochs=10)

print("---------------------------------")
print("----------saving model-----------")
print("---------------------------------")

model.save("model_trained.h5")
