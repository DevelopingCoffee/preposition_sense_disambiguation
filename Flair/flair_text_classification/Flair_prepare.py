import xml.etree.ElementTree as ET
import csv
import pandas as pd

root = ET.parse('../../SemEval/Train/Source/pp-with.sents.trng.xml').getroot()
allPrep = []

for type_tag in root.findall('instance'):
    element = []
    answer = type_tag.find("answer")

    instance_id = answer.get("instance")
    element.append(instance_id)

    sense_id = answer.get("senseid")
    #print(sense_id)
    label_ids = ""

    multi = True
    while(multi):
        if(sense_id.__contains__(" ")):
            id1, sense_id = sense_id.split(" ")
            label_ids += "__label__"+id1+" "
        else:
            multi = False;
            label_ids += "__label__"+sense_id+" "
    label_ids = label_ids[0:-1]
    #print(label_ids)
    element.append(label_ids)

    content = type_tag.find("context")
    newString = ET.tostring(content, encoding="unicode")
    newString = newString[9:-13].strip()
    contentRaw = content.text
    contentStriped = contentRaw.strip()
    element.append(newString)

    allPrep.append(element)

rows = []
for e in allPrep:
    rows.append([e[1],e[2]])

with open('data/train_data.csv', 'wt') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(rows)

data = pd.read_csv("./train_data.csv", encoding='latin-1').sample(frac=1).drop_duplicates()
#data = data[['v1', 'v2']].rename(columns={"v1": "label", "v2": "text"})

#data['label'] = '__label__' + data['label'].astype(str)
data.iloc[0:int(len(data) * 0.8)].to_csv('train.csv', sep='\t', index=False, header=False)
data.iloc[int(len(data) * 0.8):int(len(data) * 0.9)].to_csv('test.csv', sep='\t', index=False, header=False)
data.iloc[int(len(data) * 0.9):].to_csv('dev.csv', sep='\t', index=False, header=False);