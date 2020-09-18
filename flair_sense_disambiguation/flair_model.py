import flair
import torch
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, OneHotEmbeddings
from flair.embeddings import DocumentRNNEmbeddings, DocumentPoolEmbeddings
from flair.models import TextClassifier
from flair.tokenization import SpaceTokenizer, SegtokTokenizer

from flair.trainers import ModelTrainer
import sys

from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter
from flair.hyperparameter.param_selection import TextClassifierParamSelector, OptimizationValue


class BaseModel:
    """Train Model for flair"""

    def __init__(self,
                 directory: str = 'resources/',
                 verbose: bool = False
    ):
        """
        Train model for flair classifier to predict sense of prepositions

        :param directory: base directory where files will be stored
        :param verbose: set to True to display a progress bar
        """

        self.__directory = directory
        self.__verbose = verbose

        self.__classifier = None
        self.__corpus = None

    def _load_classifier(self):
        """Loading a classifier from file"""

        try:
            self.__classifier = TextClassifier.load(self.__directory+'final-model.pt')
        except:
            print("Unable to load classifier")

    def _create_classifier(self, data_dir: str = 'data/'):
        """
        Create a new classifier

        :param data_dir: directory where training data is stored (optimal is train, test and dev file)
        """

        if self.__corpus is None:
            self.__create_corpus(data_dir)

        # Create the label dictionary
        label_dict = self.__corpus.make_label_dictionary()

        # Instantiate Embeddings: Flair + OneHot (self-learning Embeddings)
        
        hot_embedding = OneHotEmbeddings(self.__corpus)

        glove_embedding = WordEmbeddings('glove')

        document_embeddings = DocumentPoolEmbeddings([hot_embedding, glove_embedding], fine_tune_mode='none')

        # Create the text classifier        
        self.__classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, multi_label=False)

    def __create_corpus(self, data_dir: str = "data/"):
        """
        Create a new corpus

        :param data_dir: directory where training data is stored (optimal is train, test and dev file)
        """

        col_name_map = {0: "label", 1: "text"}

        # Create the corpus
        self.__corpus: Corpus = CSVClassificationCorpus(data_folder=data_dir,
                                                        column_name_map=col_name_map,
                                                        tokenizer=SpaceTokenizer())
        print(Corpus)

    def train(self,
              data_dir: str = "data/",
              mini_batch_size: int = 16,
              learning_rate: float = 0.1,
              epochs: int = 10
    ):
        """
        Train a model

        :param data_dir: directory where training data is stored (optimal is train, test and dev file)
        :param mini_batch_size: mini batch size to use
        :param learning_rate: learning rate to use
        :param epochs: number of epochs to train
        """

        # Load classifier if none is yet loaded / create a new classifier if none can be loaded and create corpus
        if self.__classifier is None:
            self._load_classifier()
            if self.__classifier is None:
                self._create_classifier(data_dir=data_dir)
            else:
                self.__create_corpus(data_dir=data_dir)
        elif self.__corpus is None:
            self.__create_corpus(data_dir=data_dir)

        # Use GPU if available
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
        """
        Optimize hyper parameters with flair hyperopt wrapper
        """

        # Create corpus if none exists
        if self.__corpus is None:
            self.__create_corpus()

        # Use GPU if available
        flair.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        search_space = SearchSpace()

        # Define search space
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


        # Create the parameter selector
        param_selector = TextClassifierParamSelector(
            self.__corpus,
            False,
            base_path='optimization/results',
            document_embedding_type='lstm',
            max_epochs=30,
            training_runs=3,
            optimization_value=OptimizationValue.DEV_SCORE
        )

        # Start the optimization
        param_selector.optimize(search_space, max_evals=40)


def main():
    model = BaseModel(directory="resources/")
    model.train(epochs=500)


if __name__ == "__main__":
    main()
