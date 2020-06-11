from flair.data import Sentence
from flair.models import TextClassifier

classifier = TextClassifier.load('resources/taggers/trec/final-model.pt')

# create example sentence
sentence = Sentence('Peter ambled <head>after</head> them and joined other fathers who would doubtless have to help with bootlaces .')

# predict class and print
classifier.predict(sentence)

print(sentence.labels)