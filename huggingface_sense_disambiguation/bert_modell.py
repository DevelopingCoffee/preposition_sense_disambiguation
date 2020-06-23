from fast_bert.data_cls import BertDataBunch

from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging
import torch


print("Reading trainings data ...")
databunch = BertDataBunch('data/', 'data/',
                          tokenizer='bert-base-uncased',
                          train_file='train.csv',
                          val_file='val.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col='label',
                          batch_size_per_gpu=16,
                          max_seq_length=512,
                          multi_gpu=True,
                          multi_label=False,
                          model_type='bert')

print("Start training ...")

logger = logging.getLogger()
device_cuda = torch.device("cuda")
metrics = [{'name': 'accuracy', 'function': accuracy}]

learner = BertLearner.from_pretrained_model(	databunch,
						pretrained_path='bert-base-uncased',
						metrics=metrics,
						device=device_cuda,
						logger=logger,
						output_dir='output/',
						finetuned_wgts_path=None,
						warmup_steps=500,
						multi_gpu=True,
						is_fp16=True,
						multi_label=False,
						logging_steps=50)



learner.fit(epochs=6,
            lr=6e-5,
	    validate=True, 	# Evaluate the model after each epoch
	    schedule_type="warmup_cosine",
	    optimizer_type="lamb")

print("Training complete! Saving model ..")
learner.save_model()
print("Model saved!")



 
