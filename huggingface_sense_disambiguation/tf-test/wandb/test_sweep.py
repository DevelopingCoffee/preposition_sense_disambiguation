### import transformers and wandb on colab
# do: wandb login
#!pip install tensorflow==2.0.0


import wandb
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
import pandas as pd
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback

#sweep_config = {
#    'method': 'grid',
#    'parameters': {
#        'learning_rate': {
#            'values': [5e-5, 3e-5, 2e-5]
#        },
#        'batch_size': {
#            'values': [32, 64, 128]
#        },
#    }
#}
sweep_config = {
    'method': 'grid',
    'parameters': {
        'layers': {
            'values': [32, 64, 96, 128, 256]
        }
    }
}
sweep_id = wandb.sweep(sweep_config)

print("---------------------------------")
print("---------reading data------------")
print("---------------------------------")

#data = pd.read_csv('/content/gdrive/My Drive/training_data.tsv',engine='python', encoding='utf-8', error_bad_lines=False,sep="\t")
#data = pd.read_csv('../data/training_data.tsv',engine='python', encoding='utf-8', error_bad_lines=False,sep="\t")
#data = data[['sentence', 'label_id']]
#data = data.dropna()
#data = data.groupby('label_id').filter(lambda x : len(x) > 1)
#data['cat_label'] = pd.Categorical(data['label_id'])
#data['training_label'] = data['cat_label'].cat.codes
#data, data_test = train_test_split(data, test_size = 0.2, stratify = data[['training_label']])

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# Scale the pixel values of the images
train_images = train_images / 255.0
test_images = test_images / 255.0

#labels = data['training_label'].tolist()
#labels = sorted(list(set(labels)))



#model_name = 'bert-base-uncased'
#max_length = 100
#bert_config = BertConfig.from_pretrained(model_name)
#bert_config.output_hidden_states = False
#tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = bert_config)
#
#print("tokenize data...")
#x = tokenizer(
#    text=data['sentence'].to_list(),
#    add_special_tokens=True,
#    max_length=max_length,
#    truncation=True,
#    padding=True, 
#    return_tensors='tf',
#    return_token_type_ids = False,
#    return_attention_mask = False,
#    verbose = True)
#
#y = to_categorical(data['training_label']) 

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
   
   (X_train, y_train) = train_images, train_labels
   (X_test, y_test) = test_images, test_labels
   
   # Specify the hyperparameter to be tuned along with
   # an initial value
   configs = {
       'layers': 128
   }
   
   # Specify the other hyperparameters to the configuration
   config = wandb.config
   config.epochs = 5
   
   # Define the model
   model = tf.keras.Sequential([
       tf.keras.layers.Flatten(input_shape=(28, 28)),
       tf.keras.layers.Dense(wandb.config.layers, activation=tf.nn.relu),
       tf.keras.layers.Dense(10, activation=tf.nn.softmax)
   ])
   
   # Compile the model
   model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
   
   # Train the model
   model.fit(X_train, y_train, epochs=config.epochs,
                 validation_data=(X_test, y_test),
            callbacks=[WandbCallback(data_type="image", labels=labels)])
wandb.agent(sweep_id, function=train)
