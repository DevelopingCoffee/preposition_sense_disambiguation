import xml.etree.ElementTree as ET
import csv
import os
import random


def version_1():
    rootdir = '../SemEval/Train/Source/'

    allPrep = []

    # Go through all files
    for file in os.scandir(rootdir):
        file = str(file)[11:-2]
        #print(file)
        root = ET.parse(rootdir+str(file)).getroot()

        # Go through the file
        for type_tag in root.findall('instance'):
            element = []
            answer = type_tag.find("answer")

            # Get sense_id (label)
            sense_id = answer.get("senseid")
            #print(sense_id)
            label_ids = ""

            multi = True
            while(multi):
                if(sense_id.__contains__(" ")):
                    id1, sense_id = sense_id.split(" ", 1)
                    label_ids += "__label__"+id1+" "
                else:
                    multi = False
                    label_ids += "__label__"+sense_id+" "
            label_ids = label_ids[0:-1]
            #print(label_ids)
            element.append(label_ids)

            # Get sentence (text)
            content = type_tag.find("context")
            newString = ET.tostring(content, encoding="unicode")
            newString = newString[9:-13].strip()
            contentRaw = content.text
            contentStriped = contentRaw.strip()
            # Filter corrupt data
            if not (newString.__contains__("head") and newString.__contains__("/head")):
                instance_id = answer.get("instance")
                print("ALARM -- INSTANCE ID:", instance_id)
                continue
            element.append(newString)

            allPrep.append(element)

    # Randomize data (for split)
    random.shuffle(allPrep)

    # Split data for training, testing and dev
    train_data = allPrep[0:int(len(allPrep)*0.8)]
    test_data = allPrep[int(len(allPrep)*0.8):int(len(allPrep)*0.9)]
    dev_data = allPrep[int(len(allPrep)*0.9):]

    # Write all data in one file
    with open('old_data/train_data.csv', 'wt') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(allPrep)

    # Write 80% of data in a train file
    with open('old_data/train.csv', 'wt') as train_file:
        writer = csv.writer(train_file)
        writer.writerows(train_data)

    # Write 10% of data in a test file
    with open('old_data/test.csv', 'wt') as test_file:
        writer = csv.writer(test_file)
        writer.writerows(test_data)

    # Write 10% of data in a dev file
    with open('old_data/dev.csv', 'wt') as dev_file:
        writer = csv.writer(dev_file)
        writer.writerows(dev_data)

def version2():
    rootdir = "../SemEval/"

    allPrep = []

    with open(rootdir+'training_data.tsv') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        for line in reader:
            allPrep.append(["__label__"+line[2],line[1]])

    random.shuffle(allPrep)

    # Split data for training, testing and dev
    train_data = allPrep[0:int(len(allPrep) * 0.8)]
    test_data = allPrep[int(len(allPrep) * 0.8):int(len(allPrep) * 0.9)]
    dev_data = allPrep[int(len(allPrep) * 0.9):]

    # Write all data in one file
    with open('data/train_data.csv', 'wt') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(allPrep)

    # Write 80% of data in a train file
    with open('data/train.csv', 'wt') as train_file:
        writer = csv.writer(train_file)
        writer.writerows(train_data)

    # Write 10% of data in a test file
    with open('data/test.csv', 'wt') as test_file:
        writer = csv.writer(test_file)
        writer.writerows(test_data)

    # Write 10% of data in a dev file
    with open('data/dev.csv', 'wt') as dev_file:
        writer = csv.writer(dev_file)
        writer.writerows(dev_data)


if __name__ == "__main__":
    version2()