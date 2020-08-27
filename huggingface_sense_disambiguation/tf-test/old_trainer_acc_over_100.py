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



train_path = 'data/training_about.tsv'
val_path = 'data/val_about.tsv'
df_train = pd.read_csv(train_path, sep='\t')
df_val = pd.read_csv(val_path, sep='\t')

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

train_input_ids = tokenize_sentences(str(df_train['sentence']), tokenizer, MAX_LEN)
train_input_ids = pad_sequences(train_input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
train_attention_masks = create_attention_masks(train_input_ids)
train_labels =  df_train['label'].values

val_input_ids = tokenize_sentences(str(df_val['sentence']), tokenizer, MAX_LEN)
val_input_ids = pad_sequences(val_input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
val_attention_masks = create_attention_masks(val_input_ids)
val_labels =  df_val['label'].values


#num_labels = len(set(train_labels + val_labels))
num_labels = 3

print("-----------------------------------")
print("Creating Dataset...")
print("-----------------------------------")


def create_dataset(ids, masks, labels):
    def gen():
        for i in range(len(ids)):
            label_vector = [0] * num_labels
            label_vector[labels[i]] = 1
            print(i)
            print(label_vector )
            yield (
                {
                    "input_ids": ids[i],
                    "attention_mask": masks[i]
                },
                tf.reshape(tf.constant(label_vector), [1,num_labels]),
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None])
            },
            tf.TensorShape([1,num_labels]),
        ),
    )

train_dataset = create_dataset(train_input_ids, train_attention_masks, train_labels)
validation_dataset = create_dataset(val_input_ids, val_attention_masks, val_labels)

print(train_dataset)

print("-----------------------------------")
print("Creating Model...")
print("-----------------------------------")

model = TFBertForSequenceClassification.from_pretrained(
    bert_model_name, 
    config=BertConfig.from_pretrained(bert_model_name, num_labels=3)
)

model.summary()


print("-----------------------------------")
print("Preparing Model...")
print("-----------------------------------")
#wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=vars(training_args))
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
history = model.fit(train_dataset, epochs=3, steps_per_epoch=115, validation_data=validation_dataset, validation_steps=7)
print("-----------------------------------")
print("Saving Model...")
print("-----------------------------------")
model.save_pretrained('output/test') # new
model.save('output/model')

## Training
#if training_args.do_train:
#    trainer.train()
#    trainer.save_model()
#    tokenizer.save_pretrained(training_args.output_dir)
