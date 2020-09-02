import flair
import torch
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, OneHotEmbeddings, StackedEmbeddings
from flair.embeddings import DocumentRNNEmbeddings, DocumentPoolEmbeddings, DocumentLSTMEmbeddings
from flair.models import TextClassifier, SequenceTagger
from flair.datasets import SentenceDataset
from flair.models import SequenceTagger
from flair.tokenization import SegtokTokenizer, SpacyTokenizer, SpaceTokenizer

from flair.trainers import ModelTrainer
from flair.data import Sentence
from re import sub, finditer
import sys
import csv
import os
from random import shuffle

from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter
from flair.hyperparameter.param_selection import TextClassifierParamSelector, OptimizationValue


class BaseModel:
    """Base Model for flair"""

    def __init__(self,
                 directory: str='resources/',
                 mini_batch_size=32,
                 verbose: bool=False
    ):
        """Base model for flair classifier to predict sense of prepositions

        :param directory: base directory where files will be stored
        :param use_tokenizer: bool variable to select if the sentecnes should be tokenized
        :param mini_batch_size: mini batch size to use
        :param verbose: set to True to display a progress bar
        """

        self.__directory = directory
        self.__mini_batch_size=mini_batch_size
        self.__verbose = verbose

        self.__classifier = None
        self.__corpus = None

    def _load_classifier(self):
        """Loading a classifier from file"""

        try:
            self.__classifier = TextClassifier.load(self.__directory+'final-model.pt')
        except:
            print("Unable to load classifier")

    def _create_classifier(self, data_dir='data/'):
        """Create a new classifier
           :param data dir: directory where training data is stored (optimal is train, test and dev file)
        """

        if self.__corpus is None:
            self.__create_corpus(data_dir)

        # Create the label dictionary
        label_dict = self.__corpus.make_label_dictionary()

        #############################################################################
        # Word Embeddings with pre-trained
        # Make a list of word embeddings
        # word_embeddings = [WordEmbeddings('glove')]

        # Initialize document embedding by passing list of word embeddings
        # Can choose between many RNN types (GRU by default, to change use rnn_type parameter)
        # document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=256)
        #############################################################################

        # Instantiate one-hot encoded word embeddings with your corpus
        hot_embedding = OneHotEmbeddings(self.__corpus)

        # Init standard GloVe embedding
        glove_embedding = WordEmbeddings('glove')

        # Document pool embeddings
        document_embeddings = DocumentPoolEmbeddings([hot_embedding, glove_embedding], fine_tune_mode='none')

        # word_embeddings = [WordEmbeddings('glove'), FlairEmbeddings('news-forward-fast'),
        #                    FlairEmbeddings('news-backward-fast')]
        #
        # document_embeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_size=512, reproject_words=True,
        #                                              reproject_words_dimension=256)

        # Create the text classifier
        self.__classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, multi_label=False)

    def __create_corpus(self, data_dir="data/"):
        """Create a new corpus
           :param data dir: directory where training data is stored (optimal is train, test and dev file)
        """

        col_name_map = {0: "label", 1: "text"}

        # Create the corpus
        self.__corpus: Corpus = CSVClassificationCorpus(data_folder=data_dir,
                                                        column_name_map=col_name_map,
                                                        tokenizer=SpaceTokenizer())
        print(Corpus)


    def train(self, data_dir="data/", mini_batch_size=32, learning_rate=0.1, epochs=10):
        """Train a model
           :param data dir: directory where training data is stored (optimal is train, test and dev file)
        """

        if self.__classifier is None:
            self._load_classifier()
            if self.__classifier is None:
                self._create_classifier(data_dir=data_dir)
            else:
                self.__create_corpus(data_dir=data_dir)
        elif self.__corpus is None:
            self.__create_corpus(data_dir=data_dir)


        flair.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the text classifier trainer
        trainer = ModelTrainer(self.__classifier, self.__corpus)

        # Start the training
        trainer.train(self.__directory,
                      learning_rate=learning_rate,
                      mini_batch_size=mini_batch_size,
                      anneal_factor=0.5,
                      patience=5,
                      max_epochs=epochs)

    def optimize(self):

        # define your search space
        search_space = SearchSpace()
        search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[
            [WordEmbeddings('en')],
            [FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')],
            [WordEmbeddings('glove')],
            [WordEmbeddings('glove'), OneHotEmbeddings(self.__corpus)],
            [FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward'), OneHotEmbeddings(self.__corpus)]
        ])
        search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128])
        search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
        search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
        search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2])
        search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8, 16, 32])

        # create the parameter selector
        param_selector = TextClassifierParamSelector(
            self.__corpus,
            False,
            'optimization/results',
            'lstm',
            max_epochs=50,
            training_runs=3,
            optimization_value=OptimizationValue.DEV_SCORE
        )

        # start the optimization
        param_selector.optimize(search_space, max_evals=100)


def main():
    model = BaseModel(directory="resources/")
    model.optimize()

if __name__ == "__main__":
    main()