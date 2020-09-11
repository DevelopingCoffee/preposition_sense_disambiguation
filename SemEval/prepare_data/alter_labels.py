import csv

data = open("training_data.tsv")
read_tsv = csv.reader(data, delimiter="\t")

dataset = list(read_tsv)
dataset = dataset[1:]



for i in range(len(dataset)):
    dataset[i][2] = int(dataset[i][2])

maxLabel_id = float('-inf')
all_labels = set()
for i in range(len(dataset)):
    maxLabel_id = max(maxLabel_id, dataset[i][2])
    all_labels.add(dataset[i][2])

not_used = [i for i in range(maxLabel_id) if i not in all_labels]
not_used.reverse()
used = sorted(list(all_labels))

while used and not_used and not_used[-1] < used[-1]:
    to_be_subs = used.pop()
    sub = not_used.pop()
    for i in range(len(dataset)):
        if dataset[i][2] == to_be_subs:
            dataset[i][2] = sub

with open('training_data_labels.tsv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    header = ["sentence_id","sentence","label_id","definition"] 
    writer.writerow(header)
    for _, e in enumerate(dataset):
        writer.writerow(e)


maxE = float('-inf')
arr = set()
for i in range(len(dataset)):
    maxE = max(maxE, dataset[i][2])
    if dataset[i][2] == 12:
        print(dataset[i])
    arr.add(dataset[i][2])

for i in range(237):
    if i not in arr:
        print(i)

print(maxE)
print(sorted(list(arr)))
print(len(arr))
