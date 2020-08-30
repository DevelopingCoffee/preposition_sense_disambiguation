# Preposition sense disambiguation with FlairNLP

FlairNLP is a framework for NLP tasks. We use a flair text classifier to classify the prepositions and a flair sequence tagger to tag the prepositions in a sentence.

## How to work with this project

A simple test script is provided with _flair\_test.py_. It can take command line arguments to select training or testing and to select if a tokenizer may be used.

The flair model itself is located in _flair\_model.py_. The script includes the base flair Model for preposition classification and a (Sequence) Tagger to tag prepositions in a sentence. The latter is used when predicting a sentence and it marks the preposition in a sentence to be predicted. It is only necessary for use within the Base Model class and it is not needed to use this class oneself.

The base model contains several methods to load an existing classifier or create a new one and train the model, but when starting a training run an existing classifier in the given resources folder will always be loaded if no classifier is yet available. In the case no classifier can be loaded a new one will be created - however it will always try to laod an existing classifier first.

### Trainer and predictor

The trainer can be started with the train-method. As parameter this method takes the directory for the **csv**-formatted training files. Standard is '_data/_'.

The predictor can be started with the predict-method. As parameter this method takes a list of sentences to predict. Standard tokenizer is SegTok. A list of predictions will be returned.

### Directories

The important parameter for this model are the directory for resources and data. The directory for resources - necessary to save e.g. log files and the model itself - must be given when initialising a new model. The directory for data has to be given when starting a training run.

### Tokenization

Furthermore the model provides the option to 'skip' tokenization by setting the _use\_tokenizer_ parameter to _False_ - it means, that a sentence will only be tokenized due to spaces in the text. This can be useful, when training / predicting already tokenized sentences. If this option is not selected, the standard tokenizer (SegTok) will be used.
