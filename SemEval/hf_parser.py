import xml.etree.ElementTree as ET
import csv
import io
import os
import random
from collections import defaultdict

dataset = []
all_labels = set()

data_dir = 'Train/Source/'
def_dir = 'Tratz Variations/definitions/'

label_id = 0
sentence_id = 0
skipped_examples = 0


skipped = defaultdict(list)


# Go thorugh all files
for file in os.scandir(data_dir):
    data_file = str(file)[11:-2]
    prep = data_file.split('.')[0][3:]
    definition_file = prep +'.def.xml' 


    data_xml = ET.parse(data_dir+str(data_file)).getroot()
    def_xml = ET.parse(def_dir+str(definition_file)).getroot()


    senses = {} # dict which maps id to acutal id and also contains defintions

    print('\n')
    print('------------------------------------')
    print(definition_file)
    print('------------------------------------')
    print('\n')

    # target-ids
    for sense in def_xml.findall('sense'):
        for pprojmap in sense.findall('pprojmap'):
            if pprojmap is None:
                continue
            target_ids = pprojmap.get('targetid')
            if target_ids is None:
                continue
            definition = sense.find('definition').text
            if definition == '':
                continue
            for target_id in target_ids.split(','):
                if target_id not in senses:
                    label_id += 1
                    senses[target_id] = [label_id, definition] 

    # union-of
    for sense in def_xml.findall('sense'):
        for pprojmap in sense.findall('pprojmap'):
            if pprojmap is None:
                continue
            union_of= pprojmap.get('unionof')
            if union_of is None:
                continue
            definition = sense.find('definition').text
            if definition == '':
                continue
            for target_id in union_of.split(','):
                if target_id not in senses:
                    label_id += 1
                    senses[target_id] = [label_id, definition] 
    # subset-of
    for sense in def_xml.findall('sense'):
        for pprojmap in sense.findall('pprojmap'):
            if pprojmap is None:
                continue
            subset_of = pprojmap.get('subsetof')
            if subset_of is None:
                continue
            definition = sense.find('definition').text
            if definition == '':
                continue
            for target_id in subset_of.split(','):
                if target_id not in senses:
                    label_id += 1
                    senses[target_id] = [label_id, definition] 

    # For the current preposition, this is oure sense-dictionary: sense-id -> (our-new-sense-id, definition)
    print(senses.keys())



    for root in data_xml.findall('instance'):
        content = root.find("context")
        sentence = ET.tostring(content, encoding="unicode")[9:-13].strip()

        sense_id = root.find("answer").get("senseid").split(',')[0]

        if sense_id not in senses:
            skipped_examples += 1
            skipped[prep].append(sense_id)
            continue

        element = [sentence_id, sentence]
        sentence_id += 1


        element += senses[sense_id]
    
        ###
        # element = [sentence_id, sentence, label_id, definition]
        ###
        dataset.append(element)
    print("skip-counter: {}".format(str(skipped_examples)))

print("CSV has been created, {} examples were skipped, because no definition available".format(str(skipped_examples)))
print(skipped)


with open('training_data.tsv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    header = ["sentence_id","sentence","label_id","definition"] 
    writer.writerow(header)
    for _, e in enumerate(dataset):
        writer.writerow(e)
