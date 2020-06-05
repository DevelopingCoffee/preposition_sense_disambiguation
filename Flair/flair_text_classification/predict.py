from flair.data import Corpus
from flair.datasets import TREC_6
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

classifier = TextClassifier.load('resources/taggers/trec/final-model.pt')

# create example sentence
sentence = Sentence('I ran for three hours ?')

# predict class and print
classifier.predict(sentence)

print(sentence.labels)