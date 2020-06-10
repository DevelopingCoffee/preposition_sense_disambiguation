from flair.data import Sentence
from flair.models import TextClassifier

classifier = TextClassifier.load('resources/taggers/trec/final-model.pt')

# create example sentence
sentence = Sentence('Why had she never imagined Bella <head>as</head> a redhead ?')

# predict class and print
classifier.predict(sentence)

print(sentence.labels)