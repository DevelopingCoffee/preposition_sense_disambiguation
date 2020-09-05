
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



def get_model():

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        num_labels = len(data_train.training_label.value_counts()), 
        output_attentions = False, 
        output_hidden_states = False, 
    )
    return model
    
def train():
    print("wtf")
    wandb.init(project="hyperparameter-sweeps-torch")
    print("loading model")
    model = get_model()
    print("Using Cuda")

    model.cuda()
    optimizer = AdamW(model.parameters(),
                      lr = wandb.config.learning_rate, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )

    #train_dataloader, validation_dataloader = dataloaders[wandb.config.batch_size]
    epochs = 4
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
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
    wandb.init()
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
    
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

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

data = pd.read_csv('/content/gdrive/My Drive/training_data.tsv',engine='python', encoding='utf-8', error_bad_lines=False,sep="\t")
#data = pd.read_csv('data/training_data.tsv',engine='python', encoding='utf-8', error_bad_lines=False,sep="\t")
data = data[['sentence', 'label_id']]
data = data.dropna()
data = data.groupby('label_id').filter(lambda x : len(x) > 1)
data['cat_label'] = pd.Categorical(data['label_id'])
data['training_label'] = data['cat_label'].cat.codes
data_train, data_val = train_test_split(data, test_size = 0.1, stratify = data[['training_label']])

print("loading pretrained bert/tokenizer...")
model_name = 'bert-base-uncased'
max_length = 100
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)


print("tokenize data...")
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

y_train = torch.tensor(data_train.training_label.values, dtype=torch.long)
y_val = torch.tensor(data_val.training_label.values, dtype=torch.long)
train_dataset = TensorDataset(x_train['input_ids'], x_train['attention_mask'], y_train)
val_dataset = TensorDataset(x_val['input_ids'], x_val['attention_mask'], y_val)
dataloaders = {}
print("starting dataloader")
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
sweep_config = {
    'method': 'grid',
    'parameters': {
        'learning_rate': {
            'values': [5e-5, 3e-5, 2e-5]
        },
        'batch_size': {
            'values': [32, 64]
        },
    },
    'early_terminate':{
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 27.
    }
}
sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, function=train)
