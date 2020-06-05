import xml.etree.ElementTree as ET
import csv

root = ET.parse('../../SemEval/Train/Source/pp-with.sents.trng.xml').getroot()
allPrep = []

for type_tag in root.findall('instance'):
    element = []
    answer = type_tag.find("answer")

    instance_id = answer.get("instance")
    element.append(instance_id)

    sense_id = answer.get("senseid")
    label_ids = ""
    while(sense_id.__contains__(" ")):
        id1, sense_id = sense_id.split(" ")
        label_ids += "__label__"+id1+" "
    else:
        label_ids = "__label__"+sense_id+" "
    label_ids = label_ids[0:-1]
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
