import os

import pytest

from torch_nlp_project.config import InferConfig, TrainConfig


@pytest.fixture
def resource_path():
    resource_path = os.path.join(os.path.dirname(__file__), "files")
    return resource_path


def test_training_config(resource_path):
    config = TrainConfig.from_yaml(os.path.join(resource_path, "training_config.yml"))

    assert config.model_type == "SampleModel"
    assert config.epoch == 1
    assert config.learning_rate == 1e-4


def test_inference_config(resource_path):
    config = InferConfig.from_json(os.path.join(resource_path, "inference_config.json"))

    assert config.model_type == "SampleModel"
    assert config.val_batch_size == 32
