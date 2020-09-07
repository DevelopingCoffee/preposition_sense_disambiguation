from flair.models import TextClassifier
from flair.tokenization import SpaceTokenizer
from flair.data import Sentence


class BaseModel:
    """Base Model for flair"""

    def __init__(self,
                 directory: str = 'resources/',
                 mini_batch_size: int = 32,
                 verbose: bool = False
    ):
        """
        Base model for flair classifier to predict sense of prepositions

        :param directory: directory where the model to use can be found
        :param mini_batch_size: mini batch size to use
        :param verbose: set to True to display a progress bar
        """

        self.__classifier = None

        try:
            self.__classifier = TextClassifier.load(directory + 'final-model.pt')
        except:
            print("No classifier found in " + directory)

        self.__mini_batch_size = mini_batch_size
        self.__verbose = verbose

    def _load_classifier(self, directory: str = 'resources/'):
        """
        Loading a classifier from file

        :param directory: directory where the model to use can be found
        """

        try:
            self.__classifier = TextClassifier.load(directory + 'final-model.pt')
        except:
            print("No classifier found in " + directory)



    def predict(self, sentence: str):
        """
        Predict a sentences

        :param sentence: sentence to predict
        :return: sense id of the predicted preposition
        """

        # (Try to) load classifier if none has yet been loaded
        if self.__classifier is None:
            self._load_classifier()
            if self.__classifier is None:
                raise ValueError('Unable to load classifier. Prediction not possible')

        # Tokenize sentence with space tokenizer
        sentence = Sentence(sentence, SpaceTokenizer())
        self.__classifier.predict(
            sentence=sentence,
            mini_batch_size=self.__mini_batch_size,
            verbose=self.__verbose
        )

        # Return sense id
        return str(sentence.labels).split(" ")[0].split("__")[2]
