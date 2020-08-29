from tensorflow import keras
import tensorflow as tf
from transformers import TFBertModel,  BertConfig, BertTokenizerFast
from tensorflow.keras.layers import Input

print("--------------------")
print("loading trained model ...")
print("--------------------")
model = keras.models.load_model('output-example/model2')

print("--------------------")
print("tokenize input ...")
print("--------------------")
# Name of the BERT model to use
model_name = 'bert-base-uncased'
# Max length of tokens
max_length = 100
# Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False
# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)

x = tokenizer(
        text=["'transworld systems inc. \nis trying to collect a debt that is not mine, not owed and is inaccurate.", "But if i artfically enlarge this stupid compliant than it is possible for the neural neworkt to work? This is quite unfortunate since I need a way to predict  prepositions no matter how long the input is.", "If you’re working in customer service, you’ll have to help unhappy customers at times. No matter what, you want customers to have a good image of your company so they will continue to do business with you."],
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding='max_length', 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = False,
    verbose = True)

print(x['input_ids'])

print("--------------------")
print("generate the output")
print("--------------------")
#output = model(x['input_ids'])
output = model.predict(x['input_ids'])

print(output)

