import csv
import json
import os
from pathlib import Path
from typing import List, Dict, Union, Callable
import flair
from flair.data import (
    Sentence,
    Corpus,
    Token,
    FlairDataset,
    space_tokenizer,
    segtok_tokenizer,
)
from flair.datasets.base import find_train_dev_test_files
from flair.file_utils import cached_path, unzip_file


class CSVClassificationDataset(FlairDataset):
    """
    Dataset for text classification from CSV column formatted data.
    """
    def __init__(
            self,
            path_to_file: Union[str, Path],
            column_name_map: Dict[int, str],
            label_type: str = "class",
            max_tokens_per_doc: int = -1,
            max_chars_per_doc: int = -1,
            tokenizer=segtok_tokenizer,
            in_memory: bool = True,
            skip_header: bool = False,
            encoding: str = 'utf-8',
            **fmtparams,
    ):
        """
        Instantiates a Dataset for text classification from CSV column formatted data
        :param path_to_file: path to the file with the CSV data
        :param column_name_map: a column name map that indicates which column is text and which the label(s)
        :param label_type: name of the label
        :param max_tokens_per_doc: If set, truncates each Sentence to a maximum number of Tokens
        :param max_chars_per_doc: If set, truncates each Sentence to a maximum number of chars
        :param tokenizer: Tokenizer for dataset, default is segtok
        :param in_memory: If True, keeps dataset as Sentences in memory, otherwise only keeps strings
        :param skip_header: If True, skips first line because it is header
        :param encoding: Most datasets are 'utf-8' but some are 'latin-1'
        :param fmtparams: additional parameters for the CSV file reader
        :return: a Corpus with annotated train, dev and test data
        """

        if type(path_to_file) == str:
            path_to_file: Path = Path(path_to_file)

        assert path_to_file.exists()

        # variables
        self.path_to_file = path_to_file
        self.in_memory = in_memory
        self.tokenizer = tokenizer
        self.column_name_map = column_name_map
        self.max_tokens_per_doc = max_tokens_per_doc
        self.max_chars_per_doc = max_chars_per_doc

        self.label_type = label_type

        # different handling of in_memory data than streaming data
        if self.in_memory:
            self.sentences = []
        else:
            self.raw_data = []

        self.total_sentence_count: int = 0

        # most data sets have the token text in the first column, if not, pass 'text' as column
        self.text_columns: List[int] = []
        for column in column_name_map:
            if column_name_map[column] == "text":
                self.text_columns.append(column)

        with open(self.path_to_file, encoding=encoding) as csv_file:

            csv_reader = csv.reader(csv_file, **fmtparams)

            if skip_header:
                next(csv_reader, None)  # skip the headers

            for row in csv_reader:

                # test if format is OK
                wrong_format = False
                for text_column in self.text_columns:
                    if text_column >= len(row):
                        wrong_format = True

                if wrong_format:
                    continue

                # test if at least one label given
                has_label = False
                for column in self.column_name_map:
                    if self.column_name_map[column].startswith("label") and row[column]:
                        has_label = True
                        break

                if not has_label:
                    continue

                if self.in_memory:

                    text = " ".join(
                        [row[text_column] for text_column in self.text_columns]
                    )

                    if self.max_chars_per_doc > 0:
                        text = text[: self.max_chars_per_doc]

                    sentence = Sentence(text, use_tokenizer=self.tokenizer)

                    for column in self.column_name_map:
                        if (
                                self.column_name_map[column].startswith("label")
                                and row[column]
                        ):
                            sentence.add_label(label_type, row[column])

                    if 0 < self.max_tokens_per_doc < len(sentence):
                        sentence.tokens = sentence.tokens[: self.max_tokens_per_doc]
                    self.sentences.append(sentence)

                else:
                    self.raw_data.append(row)

                self.total_sentence_count += 1

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:
        if self.in_memory:
            return self.sentences[index]
        else:
            row = self.raw_data[index]

            text = " ".join([row[text_column] for text_column in self.text_columns])

            if self.max_chars_per_doc > 0:
                text = text[: self.max_chars_per_doc]

            sentence = Sentence(text, use_tokenizer=self.tokenizer)
            for column in self.column_name_map:
                if self.column_name_map[column].startswith("label") and row[column]:
                    sentence.add_label(self.label_type, row[column])

            if 0 < self.max_tokens_per_doc < len(sentence):
                sentence.tokens = sentence.tokens[: self.max_tokens_per_doc]

            return sentence


class CSVClassificationCorpus(Corpus):
    """
    Classification corpus instantiated from CSV data files.
    """
    def __init__(
            self,
            data_folder: Union[str, Path],
            column_name_map: Dict[int, str],
            label_type: str = 'class',
            train_file=None,
            test_file=None,
            dev_file=None,
            max_tokens_per_doc=-1,
            max_chars_per_doc=-1,
            tokenizer: Callable[[str], List[Token]] = segtok_tokenizer,
            in_memory: bool = False,
            skip_header: bool = False,
            encoding: str = 'utf-8',
            **fmtparams,
    ):
        """
        Instantiates a Corpus for text classification from CSV column formatted data
        :param data_folder: base folder with the task data
        :param column_name_map: a column name map that indicates which column is text and which the label(s)
        :param label_type: name of the label
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param max_tokens_per_doc: If set, truncates each Sentence to a maximum number of Tokens
        :param max_chars_per_doc: If set, truncates each Sentence to a maximum number of chars
        :param tokenizer: Tokenizer for dataset, default is segtok
        :param in_memory: If True, keeps dataset as Sentences in memory, otherwise only keeps strings
        :param skip_header: If True, skips first line because it is header
        :param encoding: Default is 'uft-8' but some datasets are in 'latin-1
        :param fmtparams: additional parameters for the CSV file reader
        :return: a Corpus with annotated train, dev and test data
        """

        # find train, dev and test files if not specified
        dev_file, test_file, train_file = \
            find_train_dev_test_files(data_folder, dev_file, test_file, train_file)

        train: FlairDataset = CSVClassificationDataset(
            train_file,
            column_name_map,
            label_type=label_type,
            tokenizer=tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            skip_header=skip_header,
            encoding=encoding,
            **fmtparams,
        )

        test: FlairDataset = CSVClassificationDataset(
            test_file,
            column_name_map,
            label_type=label_type,
            tokenizer=tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            skip_header=skip_header,
            encoding=encoding,
            **fmtparams,
        ) if test_file is not None else None

        dev: FlairDataset = CSVClassificationDataset(
            dev_file,
            column_name_map,
            label_type=label_type,
            tokenizer=tokenizer,
            max_tokens_per_doc=max_tokens_per_doc,
            max_chars_per_doc=max_chars_per_doc,
            in_memory=in_memory,
            skip_header=skip_header,
            encoding=encoding,
            **fmtparams,
        ) if dev_file is not None else None

        super(CSVClassificationCorpus, self).__init__(
            train, dev, test, name=str(data_folder)
        )