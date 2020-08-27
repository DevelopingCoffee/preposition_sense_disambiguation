import csv

label_set = set()

with open('train_data.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
       label_set.add(row[0])


with open('label.csv', 'wt') as csv_file:
    for label in list(label_set):
        csv_file.write(label+'\n')

