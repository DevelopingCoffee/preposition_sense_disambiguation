from flair.models import TextClassifier, SequenceTagger
from flair.datasets import SentenceDataset
from flair.tokenization import SpaceTokenizer

from flair.data import Sentence
from re import sub, finditer


class BaseModel:
    """Base Model for flair"""

    def __init__(self,
                 directory: str = 'resources/',
                 mini_batch_size: int = 32,
                 verbose: bool = False
    ):
        """Base model for flair classifier to predict sense of prepositions

        :param directory: base directory where files will be stored
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


    def predict(self, sentences: list):
        """Predict a list of sentences
           :param sentences: list of sentences to predict
        """

        if self.__classifier is None:
            self._load_classifier()
            if self.__classifier is None:
                print("Prediction not possible.")
                return

        tagger = Tagger(sentences)
        dataset = tagger.do_tagging()

        dataset = SentenceDataset([Sentence(text, SpaceTokenizer()) for text in dataset])
        self.__classifier.predict(
            dataset,
            mini_batch_size=self.__mini_batch_size,
            verbose=self.__verbose
        )
        return [sentence for sentence in dataset]


class Tagger:
    """Class Tagger: Tags the prepositions in a sentence."""

    def __init__(self, inputs: list):
        """Tags the prepositions in a sentence.

        :param inputs: list of strings

        **Output**: List with strings, marked prepositions

        If a sentence contains several prepositions each preposition will be marked in an individual sentence.
        """
        # Load model
        self.__tagger = SequenceTagger.load('pos')
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
