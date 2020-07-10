import tempfile
from typing import Dict, Iterable, List, Tuple

import torch

import allennlp
from allennlp.common import JsonDict
from allennlp.common.params import Params
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
from overrides import overrides

# Dataset reader
@DatasetReader.register('classification-tsv')
class ClassificationTsvReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None):
        super().__init__(lazy)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields = {'text': text_field}
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as lines:
            for line in lines:
                text, sentiment = line.strip().split('\t')
                yield self.text_to_instance(text, sentiment)

def build_dataset_reader() -> DatasetReader:
    return ClassificationTsvReader()


# Predictor
def make_predictions(model: Model, vocab: Vocabulary, dataset_reader: DatasetReader) \
        -> List[Dict[str, float]]:
    """Make predictions using the given model and dataset reader."""
    predictions = []
    predictor = SentenceClassifierPredictor(model, dataset_reader)
    output = predictor.predict('Peter ambled <head>after</head> them and joined other fathers who would doubtless have to help with bootlaces .') # 5(2)
    predictions.append({vocab.get_token_from_index(label_id, 'labels'): prob
                        for label_id, prob in enumerate(output['probs'])})
    output = predictor.predict('Willie rose and clattered <head>down</head> the hallway .') # 3(1b)
    predictions.append({vocab.get_token_from_index(label_id, 'labels'): prob
                        for label_id, prob in enumerate(output['probs'])})
    return predictions

@Predictor.register("sentence_classifier")
class SentenceClassifierPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)


# Main method
def main():
    # Load Model
    loaded_params = Params.from_file('resources/config.json')
    loaded_model = Model.load(loaded_params, 'resources', 'final_model.th')
    loaded_vocab = loaded_model.vocab

    dataset_reader = build_dataset_reader()

    make_predictions(loaded_model, loaded_vocab, dataset_reader)

if __name__ == '__main__':
    main()