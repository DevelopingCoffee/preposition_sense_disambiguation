import xml.etree.ElementTree as ET
import csv
import os
import random

rootdir = '../SemEval/Train/Source/'

allPrep = []

# Go thorugh all files
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

        # Get sentence (text)
        content = type_tag.find("context")
        newString = ET.tostring(content, encoding="unicode")
        newString = newString[9:-13].strip()
        contentRaw = content.text
        contentStriped = contentRaw.strip()
        if not (newString.__contains__("head") and newString.__contains__("/head")):
            instance_id = answer.get("instance")
            print("ALARM -- INSTANCE ID:",instance_id)
            continue;

        element.append(newString)
        element.append(sense_id)

        allPrep.append(element)

random.shuffle(allPrep)

# Split data for training, testing and dev
train_data = allPrep[0:int(len(allPrep)*0.8)]
test_data = allPrep[int(len(allPrep)*0.8):int(len(allPrep)*0.9)]
dev_data = allPrep[int(len(allPrep)*0.9):]

# Write all data in one file
with open('data/train_data.tsv','w') as tsv_file:
    tsv_writer = csv.writer(tsv_file, delimiter='\t')
    tsv_writer.writerows(allPrep)

# Write 80% of data in a train file
with open('data/train.tsv', 'wt') as train_file:
    tsv_writer = csv.writer(train_file, delimiter='\t')
    tsv_writer.writerows(train_data)

# Write 10% of data in a test file
with open('data/dev.tsv', 'wt') as dev_file:
    tsv_writer = csv.writer(dev_file, delimiter='\t')
    tsv_writer.writerows(dev_data)

# Write 10% of data in a dev file
with open('data/test.tsv', 'wt') as test_file:
    tsv_writer = csv.writer(test_file, delimiter='\t')
    tsv_writer.writerows(test_data)
