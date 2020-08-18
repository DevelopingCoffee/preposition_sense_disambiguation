from flair.models import SequenceTagger
from flair.data import Sentence
from re import sub, search, finditer

# load model
tagger = SequenceTagger.load('pos')

# text with English and German sentences
sentence = Sentence('Lamps swung above their heads , red and green and white in the warm darkness .')

# predict PoS tags
tagger.predict(sentence)

tagged_sentence = sentence.to_tagged_string()

prepositions = []
helper = str(tagged_sentence)
while helper.__contains__("<IN>"):
    temp, helper = helper.split("<IN>", 1)
    temp = temp.rsplit(">", 1)[1]
    temp = temp.replace(" ", "")
    prepositions.append(temp)

tagged_sentence = sub('> [^>]+ <IN>', '> $$$ _prep_ $$$$', tagged_sentence)
tagged_sentence = sub('<[^>]+>', '', tagged_sentence)

predict_sentences = []
prep_count = []

print(tagged_sentence)

for match in finditer("_prep_", tagged_sentence):
    match = str(match)
    match = ((match.split("span=(")[1]).split("),")[0]).split(",")[0]
    print(match)
    match = int(match)
    prep_count.append(match)

tagged_sentence = str(tagged_sentence)

for j in range(len(prep_count)):
    count = 0
    x = 0
    tmp = ""
    for i in prep_count:
        if(j == count):
            tmp += tagged_sentence[x:i-4]+"<head>"+prepositions[count]+"<\head> "
        else:
            tmp += tagged_sentence[x:i - 4] + prepositions[count]
        count += 1
        x = i+11
    tmp += tagged_sentence[i+11:]
    predict_sentences.append(tmp)


for i in range(len(predict_sentences)):
    predict_sentences[i].replace("$", "")
    print(predict_sentences[i])
