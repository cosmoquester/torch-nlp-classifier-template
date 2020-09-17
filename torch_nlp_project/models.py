import torch
import torch.nn as nn


class BaseModel(nn.Module):
    save_params = {}

    def save(self, path: str):
        """
        Save model to file.
        :param path: (str) Path to save model a file.
        'save_params' is dict used as keyword arguments when loading the model.
        """
        torch.save({"save_params": self.save_params, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str):
        """
        Load model from file.
        :param path: (str) Path to load model from the file.
        :return: (torch.nn) Loaded model.
        """
        checkpoint = torch.load(path)
        model = cls(**checkpoint["save_params"])
        model.load_state_dict(checkpoint["state_dict"])
        return model


class SampleModel(BaseModel):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, num_classes: int):
        super(SampleModel, self).__init__()

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.tag = nn.Linear(hidden_dim, num_classes)

        # Set parameters for save
        self.save_params = {
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_classes": num_classes,
        }

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embed(input_tokens)
        outputs, _ = self.lstm(embedded)
        outputs = self.tag(outputs[0])
        return outputs
