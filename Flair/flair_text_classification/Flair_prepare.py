import xml.etree.ElementTree as ET
import csv
import pandas as pd
import os

rootdir = '../../SemEval/Train/Source/'

allPrep = []

# Go thorugh all files
for file in os.scandir(rootdir):
    file = str(file)[11:-2]
    print(file)
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
                multi = False;
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
        element.append(newString)

        allPrep.append(element)

# Write all data into a single csv file
with open('data/train_data.csv', 'wt') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(allPrep)

data = pd.read_csv("./data/train_data.csv", encoding='latin-1').sample(frac=1).drop_duplicates()

#data = data[['v1', 'v2']].rename(columns={"v1": "label", "v2": "text"})
#data['label'] = '__label__' + data['label'].astype(str)

data.iloc[0:int(len(data) * 0.8)].to_csv('data/train.csv', sep='\t', index=False, header=False)
data.iloc[int(len(data) * 0.8):int(len(data) * 0.9)].to_csv('data/test.csv', sep='\t', index=False, header=False)
data.iloc[int(len(data) * 0.9):].to_csv('data/dev.csv', sep='\t', index=False, header=False);