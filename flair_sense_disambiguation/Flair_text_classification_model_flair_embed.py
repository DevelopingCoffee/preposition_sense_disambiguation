import flair, torch
from flair.data import Corpus
from Dataset_class import CSVClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, DocumentPoolEmbeddings, OneHotEmbeddings, StackedEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
# import wandb
#
# # weights and biases init
# wandb.init(project="gpt-3")
#
# # 2. Save model inputs and hyperparameters
# config = wandb.config
# config.learning_rate = 0.01

col_name_map = {0: "label", 1: "text"}

# 1. get the corpus
corpus: Corpus = CSVClassificationCorpus('data/', col_name_map)
print(Corpus)

# 2. create the label dictionary
label_dict = corpus.make_label_dictionary()

#############################################################################
# Word Embeddings with pre-trained
# 3. make a list of word embeddings
#word_embeddings = [WordEmbeddings('glove')]

# 4. initialize document embedding by passing list of word embeddings
# Can choose between many RNN types (GRU by default, to change use rnn_type parameter)
#document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=256)
#############################################################################
# instantiate one-hot encoded word embeddings with your corpus
hot_embedding = OneHotEmbeddings(corpus)

# init standard GloVe embedding
glove_embedding = WordEmbeddings('glove')

# init Flair forward and backwards embeddings
flair_embedding_forward = FlairEmbeddings('news-forward')
flair_embedding_backward = FlairEmbeddings('news-backward')

# create a StackedEmbedding object that combines glove and forward/backward flair embeddings
stacked_embeddings = StackedEmbeddings([
                                        glove_embedding,
                                        flair_embedding_forward,
                                        flair_embedding_backward,
                                        hot_embedding
                                       ])

# document pool embeddings
document_embeddings = DocumentPoolEmbeddings([hot_embedding, glove_embedding], fine_tune_mode='none')

# 5. create the text classifier
classifier = TextClassifier(stacked_embeddings, label_dictionary=label_dict)
#classifier = TextClassifier.load('resources/best-model.pt')

# 3. Log gradients and model parameters
#wandb.watch(classifier)
# for batch_idx, (data, target) in enumerate(train_loader):
#   ...
#   if batch_idx % args.log_interval == 0:
#     # 4. Log metrics to visualize performance
#     wandb.log({"loss": loss})

flair.device = torch.device('cuda:0')

# 6. initialize the text classifier trainer
trainer = ModelTrainer(classifier, corpus)

# 7. start the training
trainer.train('resources/flair_embed/',
              learning_rate=0.1,
              mini_batch_size=32,
              anneal_factor=0.5,
              patience=5,
              max_epochs=10)
