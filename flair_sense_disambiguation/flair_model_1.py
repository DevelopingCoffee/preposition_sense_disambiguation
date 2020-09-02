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


class BaseModelOne:
    """Base Model for flair"""

    def __init__(self,
                 directory: str='resources/',
                 use_tokenizer: bool=True,
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

        self.use_tokenizer = use_tokenizer
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

        if (not(self.use_tokenizer)):
            # Already tokenized sentences will be 'tokenized' by spaces only if set to true
            # Otherwise standard tokenizer is used (SegTok is default by flair)
            tokenizer = SpaceTokenizer()
        else:
            tokenizer = SegtokTokenizer()

        # Create the corpus
        self.__corpus: Corpus = CSVClassificationCorpus(data_folder=data_dir,
                                                        column_name_map=col_name_map,
                                                        tokenizer=tokenizer)
        print(Corpus)


    def train(self, data_dir="data/"):
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
                      learning_rate=0.1,
                      mini_batch_size=self.__mini_batch_size,
                      anneal_factor=0.5,
                      patience=5,
                      max_epochs=10)

    def predict(self, sentences):
        """Predict a list of sentences
           :param sentences: list of sentences to predict
        """

        if self.__classifier is None:
            self._load_classifier()
            if self.__classifier is None:
                print("Prediction not possible.")
                return

        tagger = Tagger()
        tagger.set_input(sentences)
        dataset = tagger.do_tagging()

        dataset = SentenceDataset([Sentence(text, use_tokenizer=self.use_tokenizer) for text in dataset])
        self.__classifier.predict(
            dataset,
            mini_batch_size=self.__mini_batch_size,
            verbose=self.__verbose
        )
        return [sentence for sentence in dataset]


class Tagger:
    """Class Tagger: Tags the prepositions in a sentence."""

    def __init__(self):
        """Tags the prepositions in a sentence.

        **Input**: List with strings

        **Output**: List with strings, marked prepositions

        If a sentence contains several prepositions each preposition will be marked in an individual sentence.
        """
        # Load model
        self.__tagger = SequenceTagger.load('pos')
        self.__sentences = []

    def set_input(self, inputs):
        """Set input list with type string of sentences to tag and mark
           :param inputs: list of strings
        """

        self.__sentences = inputs

    def do_tagging(self):
        """Tagging of prepositions. Input has to be set with the *set_input* method.
           Returns a list of strings with tagged prepositions.
        """

        predict_sentences = []

        for sentence in self.__sentences:
            # Predict PoS tags
            self.__tagger.predict(sentence)

            tagged_sentence = str(sentence.to_tagged_string())

            # Extract all prepositions
            prepositions = []
            helper = tagged_sentence
            while helper.__contains__("<IN>"):
                temp, helper = helper.split("<IN>", 1)
                temp = temp.rsplit(">", 1)[1]
                temp = temp.replace(" ", "")
                prepositions.append(temp)

            # Substitute preposition with marker
            tagged_sentence = sub('> [^>]+ <IN>', '> _prep_', tagged_sentence)
            # Delete all other tags (not needed)
            tagged_sentence = sub(' <[^>]+>', '', tagged_sentence)

            # Extract position (index) of preposition
            prep_count = []
            for match in finditer("_prep_", tagged_sentence):
                match = str(match)
                match = ((match.split("span=(")[1]).split("),")[0]).split(",")[0]
                match = int(match)
                prep_count.append(match)

            # Insert the prepositions back into the sentence; Every preposition will be inserted once with
            # <head> & <\head> markers (in an individual sentence)
            for j in range(len(prep_count)):
                count = 0  # Counter which preposition (1st, 2nd, 3rd, ...) is inserted next
                begin = 0  # Counter where to continue inserting the next preposition
                tmp = ""
                i = 0
                for i in prep_count:
                    # Insert Text from previously inserted preposition to next preposition to insert
                    if(j == count):
                        # ... with head-marker
                        tmp += tagged_sentence[begin:i]+"<head>"+prepositions[count]+"<\head>"
                    else:
                        # without head-marker
                        tmp += tagged_sentence[begin:i] + prepositions[count]
                    count += 1
                    begin = i+6
                # Append rest of sentence
                tmp += tagged_sentence[i+6:]
                # Add complete sentence to return list
                predict_sentences.append(tmp)

        # Returns the list of sentences with marked prepositions
        return predict_sentences


def main():
    model = BaseModel(directory="resources/test0/")
    model.train()

if __name__ == "__main__":
    main()