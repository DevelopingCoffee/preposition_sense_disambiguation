# And pandas for data import + sklearn because you allways need sklearn
import pandas as pd
import csv
import re
data = pd.read_csv('data/training_data.tsv',engine='python', encoding='utf-8', error_bad_lines=False,sep="\t")
data = data[['sentence', 'label_id', 'definition']]
data = data.dropna()
data = data.groupby('label_id').filter(lambda x : len(x) > 1)
data['cat_label'] = pd.Categorical(data['label_id'])
data['training_label'] = data['cat_label'].cat.codes

definitions = {}
for row in data.itertuples():
    if row.training_label not in definitions:
        preposition = re.search("<head>(.*)<\/head>", row.sentence).group(1)
        definitions[row.training_label] = (row.definition, preposition)


with open('label_prep_map.tsv', 'w') as tsv_file:  
    writer = csv.writer(tsv_file, delimiter='\t')
    for label, (definition, prep) in definitions.items():
       writer.writerow([label, prep])


