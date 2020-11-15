import os

import pytest

from torch_nlp_project.data_loader import Fields, Vocab


@pytest.fixture
def resource_path():
    resource_path = os.path.join(os.path.dirname(__file__), "files")
    return resource_path


def test_vocab_load(resource_path):
    vocab = Vocab.load(os.path.join(resource_path, "sample_vocab.txt"))

    assert len(vocab.itos) == 11
    assert vocab.itos[3] == "<EOS>"
    assert vocab.stoi["안녕"] == 5


def test_vocab_save(resource_path):
    vocab = Vocab.load(os.path.join(resource_path, "sample_vocab.txt"))

    vocab_path_to_save = os.path.join(resource_path, "sample_vocab2.txt")
    vocab.save(vocab_path_to_save)

    vocab2 = Vocab.load(vocab_path_to_save)

    assert vocab.itos == vocab2.itos
    os.remove(vocab_path_to_save)


def test_fields_tokenize():
    fields = Fields(tokenize=str.split)
    assert fields.text_field.tokenize("안 녕하세 요?") == ["안", "녕하세", "요?"]
