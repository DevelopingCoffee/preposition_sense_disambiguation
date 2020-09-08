# Preposition sense disambiguation with FlairNLP

FlairNLP is a framework for NLP tasks. We use a flair text classifier to classify the prepositions and a flair sequence tagger to tag the prepositions in a sentence.

## How to work with this project

The flair model trainer is located in _flair\_model.py_ and the predictor is located in _flair\_disambiguation.py_. The second script is also integrated into the TextImager (link).

### Data

The Data to train a model has to correspond to the standard flair format in a csv file (\_\_label\_\_<label>, <text>). 

### Trainer

The trainer can be started with the train-method. As parameter this method takes the directory for the **csv**-formatted training files. Standard is '_data/_'.

This base model contains methods to load an existing classifier or create a new one and train the model, but when starting a training run an existing classifier in the given resources folder will always be loaded if no classifier is yet imported. In the case no classifier can be loaded a new one will be created - however it will always try to laod an existing classifier first.

### Predictor

The predictor can be started with the predict-method. As parameter this method takes a sentence to predict and returns the sense id (no definition). When using the provided classifier, the sentence has to be marked with the html-Tags <head> and <\\head>.

This base model contains a methods to load an existing classifier, but when creating an object a classifier in the given directory will be loaded. If none can be loaded a prediction is not possible.

### Directories

The important parameter for this model are the directory for resources and data (standard is _resources/_ and _data/_). The directory for resources - necessary to save e.g. log files and the model itself - must be given when initialising a new model. The directory for data has to be given when starting a training run.
