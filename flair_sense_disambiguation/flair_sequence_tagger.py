from flair.models import SequenceTagger
from flair.data import Sentence
from re import sub, finditer

class Tagger():
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
        """Set input list with type string of sentences to tag and mark"""

        for i in range(len(inputs)):
            # Convert String inputs to flair Sentences
            inputs[i] = Sentence(inputs[i])
        self.__sentences = inputs

    def do_tagging(self):
        """Tagging of prepositions. Input has to be set with the *set_input* method."""

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
    """Example usage"""

    # Create a list with sentences you want to tag / mark the prepositions in
    sentences_to_tag = ["Jonathan swaggered to and fro before his private army .",
                        'Lamps swung above their heads , red and green and white in the warm darkness .']

    # Create an object of the class
    tagger = Tagger()

    # Set the inputs we created
    tagger.set_input(sentences_to_tag)

    # Mark the prepositions
    sentences = tagger.do_tagging()

    # Print the sentences
    for i in range(len(sentences)):
        print(sentences[i])

if __name__ == "__main__":
    main()