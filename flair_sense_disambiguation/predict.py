from flair.data import Sentence
from flair.models import TextClassifier

# make a sentence
sentence = Sentence('I love Berlin .')

# load the NER tagger
classifier = TextClassifier.load('resources/taggers/trec/final-model.pt')

# run NER over sentence
classifier.predict(sentence)

print(sentence.labels)

