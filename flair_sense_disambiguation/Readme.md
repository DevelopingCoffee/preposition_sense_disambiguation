# Preposition sense disambiguation with FlairNLP

FlairNLP is a framework for NLP tasks. We use a flair text classifier to classify the prepositions and a flair sequence tagger to tag the prepositions in a sentence.

## How to work with this project

### Files

This project consists of four main scripts. These are

* Flair\_text\_classification\_model.py which is the training script for the classifier,
* predict.py which is the script to use the trained model, namely the predictor,
* Dataset\_class.py - the script to hold the needed classes to create the Corpus and
* Flair\_prepare.py which is used to prepare our training data from SemEval07.

Furthermore there is a script, flair\_sequence\_tagger.py, which holds the sequence tagger and two other scripts which are a copy of other scripts. These two are used to experiment with other embeddings.

### Training

The train data has to comply with the standard flair format (\_\_label\_\_<label>) and has to be saved in a csv file. To change the data for training you only need to change the driectroy for the files in the training script. By default the training data will be used from the _data_ directory

### Predicting

To predict a sentence you will first need to run the sequence tagger script on it to mark the prepositions. If the preposition is not marked - especially in sentences with several - the classifier cannot reliably predict the sense. We use <head> and </head> html-tags as marker. If needed, these can also be added manually without using the tagging script.
Sentences with marked prepositions can now be predicted using the prediction script.

### Model files

After every epoch of training the model will be saved if an improvement has been made. Furthermore a log will be created. All these files are in the _resources_ folder.
