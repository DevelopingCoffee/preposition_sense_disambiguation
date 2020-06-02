import xml.etree.ElementTree as ET

root = ET.parse('pp-with.sents.test.xml').getroot()
allPrep = []

for type_tag in root.findall('instance'):
    element = []

    instance_id = type_tag.get('id')
    element.append(instance_id)

    sense_id = ''
    element.append(sense_id)

    content = type_tag.find("context")
    newString = ET.tostring(content, encoding="unicode")
    newString = newString[9:-13].strip()
    contentRaw = content.text
    contentStriped = contentRaw.strip()
    element.append(newString)

    allPrep.append(element)

with open('test.txt', 'w') as the_file:
    for e in allPrep:
        the_file.write('[' + e[0] + '] [' + e[1] + '] [' + e[2] + ']\n')
