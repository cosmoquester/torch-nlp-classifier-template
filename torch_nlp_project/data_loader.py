from __future__ import annotations

import sys
from typing import Callable, List, Optional, Tuple

import dill
from torchtext import vocab
from torchtext.data import Dataset, Field, TabularDataset

SPECIALS = ["<SOS>", "<EOS>", "<SEP>"]


class Vocab(vocab.Vocab):
    """
    Vocab object to map word to index.
    """

    UNK = "<UNK>"

    def __init__(self, specials: List[str] = SPECIALS):
        """
        :param specials: (iterable) Special tokens that considered as words.
        """
        super().__init__({}, specials=())

        class special_tokens:
            itos = ["<PAD>", "<UNK>", *specials]

        self.extend(special_tokens)
        self.unk_index = self["<UNK>"]
        self.stoi.default_factory = lambda: self.unk_index

    def save(self, path: str) -> None:
        """
        Save vocab object to text file.
        :param path: (str) Path to save. The file format is just lines of words. There is a word in a line.
                      This is for easily looking up and appending words.
        """
        with open(path, "w") as f:
            for word in self.itos:
                f.write(word + "\n")

    @classmethod
    def load(cls, path: str) -> Vocab:
        """
        Load from saved vocab text file.
        :param path: (str) Path to load.
        """
        # Read a files
        vocab = cls(specials=())
        with open(path) as f:
            words = [x.strip() for x in f]

        # Create vocab
        class pseudo_vocab:
            itos = words

        vocab.extend(pseudo_vocab)

        return vocab


class Fields:
    def __init__(
        self,
        vocab_path: Optional[str] = None,
        tokenize: Optional[Callable[[str], List[str]]] = None,
        preprocessing: Optional[Callable[[List[str]], List[str]]] = None,
        **kwargs
    ):
        """
        object containing torchtext text and label fields.
        :param vocab_path: (str) vocab file path to create Vocab.
        :param tokenize: (func) function to tokenize texts (str) -> (list) of tokens format.
                         default tokenizer is Mecab and str.split (if mecab is unavailable)
        :param preprocessing: (func) torchtext preprocessing function after tokenize before numericalize.
        :param kwargs: (dict) keyword arguments for text field.
        """
        # Set tokenize
        if not tokenize:
            try:
                from konlpy.tag import Mecab

                mecab = Mecab()
                tokenize = mecab.morphs
            except:
                tokenize = str.split
                print("Error occured! Please install Mecab to use mecab tokenizer!", file=sys.stderr)
                print("Tokenize with 'str.split'", file=sys.stderr)

        # Set fields
        self.text_field = Field(
            tokenize=tokenize, preprocessing=preprocessing, pad_token="<PAD>", unk_token="<UNK>", **kwargs
        )
        self.label_field = Field(sequential=False, use_vocab=False)

        # Set vocab to text field
        if vocab_path:
            vocab = Vocab.load(vocab_path)
            self.text_field.vocab = vocab


class DataLoader:
    """
    Represent by train, test datasets
    """

    def __init__(self):
        self.train_dataset: Optional[TabularDataset] = None
        self.test_dataset: Optional[TabularDataset] = None

    def make_dataset(
        self,
        fields: Optional[Fields, Tuple] = None,
        train_data_path: Optional[str] = None,
        test_data_path: Optional[str] = None,
        format: str = "tsv",
        skip_header: bool = True,
    ) -> Tuple[TabularDataset, TabularDataset]:
        """
        Make datasets from train_data_path, test_data_path.
        :param fields: (list, Fields) This is used for TabularDataset fields. If the type of fields is list, it's format should be TabularDataset fields.
                        [defulat file form] type tsv, contains header row, column format ('id', 'text', 'label')
        :param train_data_path: (str) training data file path.
        :param test_data_path: (str) testing data file path.
        :param format: (str) TabularDataset processing file format. (default: tsv)
        :param skip_header: (bool) Whether skip header or not. Skip head if True.
        :return: (Dataset, Dataset) return train dataset and test dataset but this instance also contains datasets.
        """
        # Convert fields
        if not fields:
            fields = Fields()
        if isinstance(fields, Fields):
            fields = [("id", None), ("text", fields.text_field), ("label", fields.label_field)]
        else:
            assert "text" in (field_name for field_name, _ in fields)

        # Make train dataset
        if train_data_path:
            self.train_dataset = TabularDataset(
                path=train_data_path, format=format, skip_header=skip_header, fields=fields
            )
        # Make test dataset
        if test_data_path:
            self.test_dataset = TabularDataset(
                path=test_data_path, format=format, skip_header=skip_header, fields=fields
            )

        return self.train_dataset, self.test_dataset

    def build_vocab(self, dataset: TabularDataset = None, specials: List[str] = None) -> Vocab:
        """
        Build Vocab from Train dataset.
        :param dataset: (Dataset) dataset used to build vocab. (default: self.train_dataset)
        :param specials: (list) Vocab Special tokens.
        :return: (Vocab) return created vocab instance.
        """
        if dataset is None:
            dataset = self.train_dataset

        # Build vocab using "text" fields
        text_field = self.train_dataset.fields["text"]
        text_field.build_vocab(dataset)

        # Convert to "Vocab" type
        vocab = Vocab(specials) if specials else Vocab()
        vocab.extend(text_field.vocab)

        text_field.vocab = vocab
        return vocab

    def save(self, path: str) -> None:
        """
        Save dataloader to a file.
        Be careful that all functions such as tokenizer or preprocessing, postproessing are removed to save fields.
        :param path: (str) The path to save datasets.
        """
        from torch import save

        # Remove functions
        train_fields = self._get_serializable_fields(self.train_dataset) if self.train_dataset else None
        test_fields = self._get_serializable_fields(self.test_dataset) if self.test_dataset else None

        # Save to file
        save(
            {
                "train_fields": train_fields,
                "test_fields": test_fields,
                "train_examples": self.train_dataset.examples if self.train_dataset else None,
                "test_examples": self.test_dataset.examples if self.test_dataset else None,
            },
            path,
            pickle_module=dill,
        )

    @classmethod
    def load(cls, path: str) -> DataLoader:
        """
        Load dataloader from a file.
        Be careful that all functions such as tokenizer or preprocessing, postproessing are not loaded.
        :param path: (str) The path to load datasets.
        """
        from torch import load

        # Load from file
        checkpoint = load(path, pickle_module=dill)
        dataloader = cls()
        if checkpoint["train_examples"] and checkpoint["train_fields"]:
            dataloader.train_dataset = Dataset(examples=checkpoint["train_examples"], fields=checkpoint["train_fields"])
        if checkpoint["test_examples"] and checkpoint["test_fields"]:
            dataloader.test_dataset = Dataset(examples=checkpoint["test_examples"], fields=checkpoint["test_fields"])
        return dataloader

    def _get_serializable_fields(self, dataset: TabularDataset):
        from copy import copy

        fields = {k: copy(v) for k, v in dataset.fields.items()}
        fields["text"].tokenize = None
        fields["text"].tokenizer_args = (None, "ko")
        fields["text"].preprocessing = None
        fields["text"].postprocessing = None
        return fields
