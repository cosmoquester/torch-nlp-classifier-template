from datetime import datetime

import torch
from torchtext.data import Dataset, Example, Iterator, Field

from . import models
from .config import InferConfig
from .data_loader import Fields, Vocab
from .utils import get_logger

from typing import Optional, Callable, List, Union


class InferManager:
    def __init__(
        self,
        inference_config_path: str,
        tokenize: Optional[Callable[[str], List[str]]] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        :param inference_config_path: (str) inference config file path.
        :param tokenize: (func) tokenizing function. (str) -> (list) of (str) tokens.
        :prarm device: (str, torch.device) device for inference.
        """
        # Set loogger
        self.logger = get_logger(f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_infer.log")
        self.logger.info("Setting logger is complete")

        # Set device
        self.device = torch.device(device)
        self.logger.info(f"Setting device:{self.device} is complete")

        # Load training Config
        self.config = InferConfig.from_json(inference_config_path)
        self.logger.info(f"Loaded inference config from '{inference_config_path}'")

        # Load Model
        self.model = getattr(models, self.config.model_type).load(self.config.model_path)
        self.model.to(self.device)
        self.logger.info(f"Prepared model type: {type(self.model)}")

        # Load vocab
        self.vocab = Vocab.load(self.config.vocab_path)
        self.logger.info(f"Set vocab from '{self.config.vocab_path}'")

        # Set fields
        self.fields = Fields(vocab_path=self.config.vocab_path, tokenize=tokenize)
        self.logger.info(f"Set fields tokenize with '{self.fields.text_field.tokenize}'")

    def inference_texts(self, texts: List[str]) -> List[int]:
        """
        :param texts: (list) list of texts to inferece.
        :return: (list) of (int) labels about each text.
        """
        # Make inference batches
        dataset = self._list_to_dataset(texts, self.fields.text_field)
        batches = Iterator(
            dataset, batch_size=self.config.val_batch_size, device=self.device, train=False, shuffle=False, sort=False
        )

        # Predict
        labels = []
        total_step = int(len(dataset) / self.config.val_batch_size + 1)
        for batch in batches:
            output = self.model(batch.text)
            label = output.argmax(dim=1).cpu().detach().numpy()
            labels.extend(label)

        return labels

    def _list_to_dataset(self, texts: List[str], text_field: Field):
        """
        Make dataset from list of texts.
        :param texts: (list) list of texts to inferece.
        :param text_field: (Field) fields having tokenize function and vocab.
        """
        # Tokenize texts
        tokenized = [[text_field.tokenize(text)] for text in texts]

        # Make dataset from list
        fields = [("text", text_field)]
        examples = [Example.fromlist(text, fields=fields) for text in tokenized]
        dataset = Dataset(examples, fields=fields)

        return dataset
