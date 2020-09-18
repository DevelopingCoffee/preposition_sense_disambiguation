from flair.data import Sentence
from flair.models import TextClassifier
import sys
import csv
from random import shuffle

do_test = False

def testing():
    sentence = Sentence("Peter sat <head>on<\head> the bench .")
    print(sentence)
    print(sentence.labels)
    print(sentence.embedding)
    print(sentence.tokens)
    print(sentence.tokenized)

if(do_test):
    testing()
else:
    classifier = TextClassifier.load('resources_old/final-model.pt')

    # create example sentence
    if(len(sys.argv) < 2):

        predict_data = []

        with open("data/test.csv") as csvdatei:
            csv_reader_object = csv.reader(csvdatei)

            for row in csv_reader_object:
                element = []

                labels = row[0]
                sentence = Sentence(row[1])

                element.append(labels)
                element.append(sentence)
                predict_data.append(element)

        shuffle(predict_data)

        total = 0
        correct = 0
        for data in predict_data:
            classifier.predict(data[1])

            print("Prediction:"+str(data[1].labels)+"; Correct label: "+str(data[0]))
            only_label, waste = str(data[1].labels).split(" (")
            if(str(only_label).__contains__(" ")):
                only_label, waste = only_label.split(" ")
            waste, only_label = only_label.split("label__")
            if(str(data[0]).__contains__(" ")):
                label1, label2 = str(data[0]).split(" ")
                waste, correct_label1 = label1.split("label__")
                waste, correct_label2 = label2.split("label__")
            else:
                waste, correct_label1 = str(data[0]).split("label__")
                correct_label2 = ""
            if(only_label == correct_label1 or only_label == correct_label2):
                correct += 1
            total += 1

        accuracy = correct / total
        print("Total sentences: "+str(total)+"; Correct: "+str(correct)+"; % correct: "+str(accuracy))

    else:
        print(sys.argv[1])
        sentence = Sentence(sys.argv[1])

        # predict class and print
        classifier.predict(sentence)

        print(sentence.labels)
