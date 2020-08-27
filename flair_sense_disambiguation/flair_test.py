import sys
import os.path
import csv
from flair_model import BaseModel
from flair.data import Sentence
from random import shuffle

if(sys.argv[1] == "True"):
    model = BaseModel(external_tokenizer=True)
else:
    model = BaseModel()

if os.path.isfile('resources/final-model.pt'):
    model.load_classifier()
else:
    model.create_classifier()

if(sys.argv[2] == "train"):
    model.train()
else:
    predict_data = []

    with open("data/test.csv") as csvfile:
        csv_reader_object = csv.reader(csvfile)

        for row in csv_reader_object:
            element = []

            # Labels
            labels = row[0]
            # Text
            sentence = row[1]

            # Combine in list
            element.append(labels)
            element.append(sentence)
            predict_data.append(element)

    shuffle(predict_data)

    total = 0
    correct = 0
    for data in predict_data:
        # Data = [label, text]
        # Predict text with the trained model
        data[1] = model.predict([data[1]])[0]

        # Compare the prediction and given (correct) label
        print("Prediction:" + str(data[1].labels) + "; Correct label: " + str(data[0]))
        only_label, waste = str(data[1].labels).split(" (")
        if (str(only_label).__contains__(" ")):
            only_label, waste = only_label.split(" ")
        only_label = only_label.split("label__")[1]
        if (str(data[0]).__contains__(" ")):
            label1, label2 = str(data[0]).split(" ")
            correct_label1 = label1.split("label__")[1]
            correct_label2 = label2.split("label__")[1]
        else:
            correct_label1 = str(data[0]).split("label__")[1]
            correct_label2 = ""
        if (only_label == correct_label1 or only_label == correct_label2):
            correct += 1
        total += 1

    accuracy = correct / total
    print("Total sentences: " + str(total) + "; Correct: " + str(correct) + "; % correct: " + str(accuracy))



