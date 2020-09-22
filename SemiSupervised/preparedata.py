import os
import sys
with open(os.listdir()[1], 'r') as f:
    prepositions = []
    for l in f:
        prepositions.append(l[:-1])
textdata = []
with open('europarl-v7.de-en.cleanpendetl', 'r') as e: # the text file
    with open('europarl-v7.de-en.gdfa', 'r') as g: # the alignment info file
        for a, b in zip(e,g):
            a = a.replace(' ||| ', '\t')
            textdata.append((a[:-1],b[:-1]))
output = []
for t in textdata:
    matches = t[1].split(' ')
    words = t[0].split('\t')
    enwords = words[0].split(' ')
    dewords = words[1].split(' ')
    for i, word in enumerate(enwords):
        if word in set(prepositions):
            preposition_matches = []
            for match in matches:
                match = match.split('-')
                if int(match[0]) == i:
                    preposition_matches.append(match)
            for pmatch in preposition_matches:
                sentence = ""
                for j, sword in enumerate(enwords):
                    if j == int(pmatch[0]):
                        continue
                    sentence += sword+" "
                target = dewords[int(pmatch[-1])]
                sentence = sentence[:-1]
                output.append((sentence, target))
with open('training.data', 'w') as w:
    for o in output:
        w.write(o[0]+'\t'+o[1]+'\n')
