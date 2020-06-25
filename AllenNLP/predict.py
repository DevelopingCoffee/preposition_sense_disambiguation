import tempfile
from typing import Dict, Iterable, List, Tuple

import torch

import allennlp
from allennlp.common import JsonDict
from allennlp.data import DataLoader, DatasetReader, Instance
from allennlp.data import Vocabulary
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.predictors import Predictor
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer

class SentenceClassifierPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)


# We've copied the training loop from an earlier example, with updated model
# code, above in the Setup section. We run the training loop to get a trained
# model.
vocab = Vocabulary.from_files("/tmp/vocabulary")

model = SimpleClassifier(vocab, embedder, encoder)

with open("/tmp/model.th", 'rb') as f:
    model.load_state_dict(torch.load(f))



model, dataset_reader = run_training_loop()
vocab = model.vocab
predictor = SentenceClassifierPredictor(model, dataset_reader)

output = predictor.predict('A good movie!')
print([(vocab.get_token_from_index(label_id, 'labels'), prob)
       for label_id, prob in enumerate(output['probs'])])
output = predictor.predict('This was a monstrous waste of time.')
print([(vocab.get_token_from_index(label_id, 'labels'), prob)
       for label_id, prob in enumerate(output['probs'])])