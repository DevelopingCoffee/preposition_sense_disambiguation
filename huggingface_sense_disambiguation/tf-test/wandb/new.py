from wandb.keras import WandbCallback
import tensorflow as tf
import pandas as pd
import numpy as np
from transformers import TFBertModel,  BertConfig, BertTokenizer
import wandb
from wandb.keras import WandbCallback
import pickle


# data = pd.read_csv('../data/training_data.tsv',engine='python', encoding='utf-8', error_bad_lines=False,sep="\t")
# data = data[['sentence', 'label_id']]
# data = data.dropna()
# data = data.groupby('label_id').filter(lambda x : len(x) > 1)
# data['cat_label'] = pd.Categorical(data['label_id'])
# data['training_label'] = data['cat_label'].cat.codes
#data, data_test = train_test_split(data, test_size = 0.2, stratify = data[['training_label']])

model_name = 'bert-base-uncased'
max_length = 100
bert_config = BertConfig.from_pretrained(model_name)
bert_config.output_hidden_states = False
# tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path = model_name, config = bert_config)
#
# print("tokenize data...")
# # x = tokenizer(
# #    text=data['sentence'].to_list(),
# #    add_special_tokens=True,
# #    max_length=max_length,
# #    truncation=True,
# #    padding=True,
# #    return_tensors='tf',
# #    return_token_type_ids = False,
# #    return_attention_mask = False,
# #    verbose = True)
#
# with open(r"x.pickle", "wb") as output_file:
#     pickle.dump(x, output_file)

# with open(r"examplePickle", "rb") as input_file:
#     nada_pls_forget_me = pickle.load(input_file)
#
# # print(hallo)
# # print(type(hallo))
#
# print(x)
# print(type(x))

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Scale the pixel values of the images to 
train_images = train_images / 255.0
test_images = test_images / 255.0



labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
        "Sandal","Shirt","Sneaker","Bag","Ankle boot"]


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

    print("holy")
    transformer_model = TFBertModel.from_pretrained(model_name, config = bert_config)
    print("shit")

    with open(r"examplePickle", "rb") as input_file:
        nada_pls_forget_me = pickle.load(input_file)
    
    # Add the config item (layers) to wandb
    if wandb.run:
        wandb.config.update({k: v for k, v in configs.items() if k not in dict(wandb.config.user_items())})
        configs = dict(wandb.config.user_items())
    
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

sweep_config = {
    'method': 'grid',
    'parameters': {
        'layers': {
            'values': [32, 64, 96, 128, 256]
        }
    }
}

sweep_id = wandb.sweep(sweep_config)

wandb.agent(sweep_id, function=train)
