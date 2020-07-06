from flair.data import Sentence
from flair.models import TextClassifier
import sys

classifier = TextClassifier.load('resources/best-model.pt')

# create example sentence
if(len(sys.argv) < 2):
    sentence = Sentence('Peter ambled <head>after</head> them and joined other fathers who would doubtless have to help with bootlaces .') # 5(2)
    sentence2 = Sentence('The queen smiled and tweaked me gently <head>by</head> the cheek .') # 5(2)
    sentence3 = Sentence('Gingerly I squeezed a bit <head>on</head> my fingertip .') # 3(1b)
else:
    print(sys.argv[1])
    sentence = Sentence(sys.argv[1])

# predict class and print
classifier.predict(sentence)
classifier.predict(sentence2)
classifier.predict(sentence3)

print(sentence.labels +", "+ sentence2.labels +", "+ sentence3.labels)