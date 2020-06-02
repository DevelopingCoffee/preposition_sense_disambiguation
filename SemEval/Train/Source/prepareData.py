import xml.etree.ElementTree as ET

root = ET.parse('pp-with.sents.trng.xml').getroot()
allPrep = []

for type_tag in root.findall('instance'):
    element = []
    answer = type_tag.find("answer")

    instance_id = answer.get("instance")
    element.append(instance_id)

    sense_id = answer.get("senseid")
    element.append(sense_id)

    content = type_tag.find("context")
    newString = ET.tostring(content, encoding="unicode")
    newString = newString[9:-13].strip()
    contentRaw = content.text
    contentStriped = contentRaw.strip()
    element.append(newString)

    allPrep.append(element)

with open('training.txt', 'w') as the_file:
    for e in allPrep:
        the_file.write('[' + e[0] + '] [' + e[1] + '] [' + e[2] + ']\n')
