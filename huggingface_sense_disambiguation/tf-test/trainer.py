# Load Huggingface transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast
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


print("---------------------------------")
print("---------reading data------------")
print("---------------------------------")

data = pd.read_csv('data/training_data.tsv',engine='python', encoding='utf-8', error_bad_lines=False,sep="\t")
# Select required columns
print(data)
data = data[['sentence', 'label_id']]
# Remove a row if any of the three remaining columns are missing
data = data.dropna()
# Remove rows, where the label is present only ones (can't be split)
data = data.groupby('label_id').filter(lambda x : len(x) > 1)
# Set your model output as categorical and save in new label col
data, data_test = train_test_split(data, test_size = 0.2, stratify = data[['label_id']])


print(data)

print("---------------------------------")
print("-------configure Bert------------")
print("---------------------------------")

# Name of the BERT model to use
model_name = 'bert-base-uncased'
# Max length of tokens
max_length = 100
# Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False
# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
# Load the Transformers BERT model
transformer_model = TFBertModel.from_pretrained(model_name, config = config)



# TF Keras documentation: https://www.tensorflow.org/api_docs/python/tf/keras/Model
# Load the MainLayer
bert = transformer_model.layers[0]
# Build your model input
input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
inputs = {'input_ids': input_ids}
# Load the Transformers BERT model as a layer in a Keras model
bert_model = bert(inputs)[1]
dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
pooled_output = dropout(bert_model, training=False)
# Then build your model output
label_id = Dense(units=len(data.label_id.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='label_id')(pooled_output)
#product = Dense(units=len(data.Product_label.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='product')(pooled_output)
outputs = {'label_id': label_id}
# And combine it all in a model object
model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')
# Take a look at the model
model.summary()


print("---------------------------------")
print("-------Train Bert------------")
print("---------------------------------")

# Set an optimizer
optimizer = Adam(
    learning_rate=5e-05,
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)
# Set loss and metrics
loss = {'label_id': CategoricalCrossentropy(from_logits = True)}
metric = {'label_id': CategoricalAccuracy('accuracy')}

# Compile the model
print("compiling model...")
model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)
# Ready output data for the model
y_label_id = to_categorical(data['label_id'])
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

# Fit the model
print("fit model...")
history = model.fit(
    x={'input_ids': x['input_ids']},
    y={'label_id': y_label_id},
    validation_split=0.2,
    batch_size=64,
    epochs=10)

# todo shapes dont fit -> print tokenized words and check

model.save('output/model2')
