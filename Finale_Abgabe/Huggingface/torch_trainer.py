### import transformers 

from transformers import BertConfig, BertTokenizerFast, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch
import time
import datetime
import random
import wandb



# Set your custom Parameter here:

wandb_apikey = "ef94a92678c9a088899b27fb3eb2ca4b7c19642c" # your personal api-key from weights and bias
path_data = "preposition_sense_disambiguation/huggingface_sense_disambiguation/torch-test/data/training_data.tsv" # where your input-data is
label_prep_map_path = "preposition_sense_disambiguation/huggingface_sense_disambiguation/torch-test/label_prep_map.tsv" # location of label-prep-map (for error analysis)

# choose parameter (for preposition disambiguation a batch-size of 16 and a learning rate of 1e-4 seems to be very good). The other parameters are also recommended by google
batch_size = 16  #8, 16, 32, 64, 128
learning_rate = 1e-4 #  3e-4, 1e-4, 5e-5, 3e-5

# logging-settings
wandb.init(project = 'Torch-Best-Model-Google-HO')
wandb.config.epochs = 4
wandb.config.batch_size = batch_size
wandb.config.learning_rate = learning_rate


# train on gpu if possible
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



print("---------------------------------")
print("--------preparing data-----------")
print("---------------------------------")

data = pd.read_csv(path_data, engine='python', encoding='utf-8', error_bad_lines=False,sep="\t")
data = data[['sentence', 'label_id']]
data = data.dropna()
data = data.groupby('label_id').filter(lambda x : len(x) > 1)
data['cat_label'] = pd.Categorical(data['label_id'])
data['training_label'] = data['cat_label'].cat.codes
data_train, data_val = train_test_split(data, test_size = 0.1, stratify = data[['training_label']])

print("loading pretrained bert/tokenizer...")
model_name = 'bert-base-uncased'
max_length = 100
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, do_lower_case=True)


print("tokenize data...")
# creating input-ids and attention-masks of the sentences
x_train = tokenizer(
    text=data_train['sentence'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='pt',
    return_token_type_ids = False,
    verbose = True)

x_val = tokenizer(
    text=data_val['sentence'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='pt',
    return_token_type_ids = False,
    verbose = True)

# creating the target-label
y_train = torch.tensor(data_train.training_label.values, dtype=torch.long)
y_val = torch.tensor(data_val.training_label.values, dtype=torch.long)

# bringing it into pytorch format
train_dataset = TensorDataset(x_train['input_ids'], x_train['attention_mask'], y_train)
val_dataset = TensorDataset(x_val['input_ids'], x_val['attention_mask'], y_val)
train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), 
            batch_size = batch_size 
        )

validation_dataloader = DataLoader(
            val_dataset, 
            sampler = SequentialSampler(val_dataset), 
            batch_size = batch_size 
        )


print("---------------------------------")
print("-------configure Bert------------")
print("---------------------------------")

# loading bert for classification from huggingface
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels = len(data_train.training_label.value_counts()), 
    output_attentions = False, 
    output_hidden_states = False, 
)


model.cuda()

# print some general information about the layers and parameters:
params = list(model.named_parameters())
print('The BERT model has {:} different named parameters.\n'.format(len(params)))
print('==== Embedding Layer ====\n')
for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== First Transformer ====\n')
for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Output Layer ====\n')
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = 1e-8 
                )
epochs = 4
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

print("---------------------------------")
print("--------fine-tune bert-----------")
print("---------------------------------")

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
training_stats = []
total_t0 = time.time()

for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_train_loss = 0
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)

        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)            
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
    print("")
    print("Running Validation...")
    
    # validation
    t0 = time.time()

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():        

            (loss, logits) = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    validation_time = format_time(time.time() - t0)
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # log results to wandb! :)
    wandb.log({'val_accuracy': avg_val_accuracy, 'epoch': epoch_i + 1, 'val_loss': avg_val_loss, 'train_loss': avg_train_loss})

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
print("---------------------------------")
print("----------saving model-----------")
print("---------------------------------")

# save trained model
output_dir = './model_save/'
model_to_save = model.module if hasattr(model, 'module') else model 
model_to_save.save_pretrained(output_dir)

