import tensorflow as tf
import numpy as np
from transformers import PreTrainedModel 
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import TFBertForSequenceClassification, BertConfig, TokenClassificationPipeline


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertForSequenceClassification.from_pretrained('output/test/')
model.summary()


sentence = "Louisa put down the book , crossed to the bed and wrapped an arm <head>about</head> Emilia 's shoulders ."
inputs = tokenizer(sentence, return_tensors="tf")

inputs["labels"] = tf.reshape(tf.constant(1), (-1, 1)) # Batch size 1
print(type(inputs))
outputs = model(inputs)
#print(outputs)
#loss, logits = outputs[:2]

#print(loss, logits)

