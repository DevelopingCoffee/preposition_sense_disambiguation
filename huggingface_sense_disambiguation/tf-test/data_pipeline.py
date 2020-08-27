from tqdm.notebook import tqdm
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from transformers import TFTrainer 
from transformers import HfArgumentParser 
from transformers import TFTrainingArguments 
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification, BertConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import wandb


print("-----------------------------------")
print("Parse Arguments....")
print("-----------------------------------")

parser = HfArgumentParser((TFTrainingArguments))
training_args = parser.parse_args_into_dataclasses()[0]
print(training_args)
print(training_args.__annotations__)

train_path = 'data/training_about.tsv'
#test_path = 'data/test.csv'
#test_labels_path = 'data/test_labels.csv'
#subm_path = 'data/sample_submission.csv'


df_train = pd.read_csv(train_path, sep='\t')
#df_test = pd.read_csv(test_path)
#df_test_labels = pd.read_csv(test_labels_path)
#df_test_labels = df_test_labels.set_index('id')
#print(df_test_labels .head())
#df_train.head()

# Tokenizaton
print("-----------------------------------")
print("START TOKENIZATION....")
print("-----------------------------------")
bert_model_name = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
MAX_LEN = 128

def tokenize_sentences(sentences, tokenizer, max_seq_len = 128):
    tokenized_sentences = []

    for sentence in tqdm(sentences):
        tokenized_sentence = tokenizer.encode(
                            sentence,                  # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_seq_len,  # Truncate all sentences.
                    )
        
        tokenized_sentences.append(tokenized_sentence)

    return tokenized_sentences

def create_attention_masks(tokenized_and_padded_sentences):
    attention_masks = []

    for sentence in tokenized_and_padded_sentences:
        att_mask = [int(token_id > 0) for token_id in sentence]
        attention_masks.append(att_mask)

    return np.asarray(attention_masks)

input_ids = tokenize_sentences(df_train['sentence'], tokenizer, MAX_LEN)
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
attention_masks = create_attention_masks(input_ids)


print("-----------------------------------")
print("Creating Dataset...")
print("-----------------------------------")
# Create Dataset
label_cols =  ['about_affect', 'about_spatial', 'about_topic']
labels =  df_train[label_cols].values


def create_dataset(ids, masks, labels):
    def gen():
        for i in range(len(input_ids)):
            yield (
                {
                    "input_ids": ids[i],
                    "attention_mask": masks[i]
                },
                tf.reshape(tf.constant(labels[i]), [1,3]),
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None])
            },
            tf.TensorShape([1, 3]),
        ),
    )

train_dataset = create_dataset(input_ids, attention_masks, labels)
print("----------------------------------------")
print(train_dataset)
print("----------------------------------------")
validation_dataset = create_dataset(input_ids, attention_masks, labels)


print("-----------------------------------")
print("Creating Model...")
print("-----------------------------------")
# Create Model

model = TFBertForSequenceClassification.from_pretrained(
    bert_model_name, 
    config=BertConfig.from_pretrained(bert_model_name, num_labels=3)
)

print("-----------------------------------")
print("Preparing Model...")
print("-----------------------------------")
wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=vars(training_args))
# Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.CategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

## Initialize our Trainer
#trainer = TFTrainer(
#    model=model,
#    args=training_args,
#    train_dataset=train_dataset,
#    eval_dataset=validation_dataset,
#)
print("-----------------------------------")
print("Training Model...")
print("-----------------------------------")
# Train and evaluate using tf.keras.Model.fit()
history = model.fit(train_dataset, epochs=1, steps_per_epoch=115, validation_data=validation_dataset, validation_steps=7)
print("-----------------------------------")
print("Saving Model...")
print("-----------------------------------")
model.save('output/model')

## Training
#if training_args.do_train:
#    trainer.train()
#    trainer.save_model()
#    tokenizer.save_pretrained(training_args.output_dir)