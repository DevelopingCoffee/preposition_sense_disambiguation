from flair.data import Sentence
from flair.models import TextClassifier

classifier = TextClassifier.load('resources/taggers/trec/final-model.pt')

# create example sentence
sentence = Sentence('He went to the station.')

# predict class and print
classifier.predict(sentence)

print(sentence.labels)