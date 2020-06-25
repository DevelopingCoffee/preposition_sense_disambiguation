from flair.data import Sentence
from flair.models import TextClassifier
import sys

classifier = TextClassifier.load('resources/taggers/trec/best-model.pt')

# create example sentence
# sentence = Sentence('Peter ambled <head>after</head> them and joined other fathers who would doubtless have to help with bootlaces .')
if(len(sys.argv) < 2):
    sentence = Sentence('Peter ambled <head>after</head> them and joined other fathers who would doubtless have to help with bootlaces .')
else:
    print(sys.argv[1])
    sentence = Sentence(sys.argv[1])

# predict class and print
classifier.predict(sentence)

print(sentence.labels) # 5(2)