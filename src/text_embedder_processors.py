import torch
from sentence_transformers import SentenceTransformer
from typing import List, Any
from enum import Enum as BaseEnum
from psqlextra.types import StrEnum as BaseStrEnum
from os import environ

import abc
from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = environ.get("PINECONE_ENVIRONMENT")

parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.01,
    "top_p": 0.8,
    "top_k": 40
}


class Enum(BaseEnum):
    """Extends the base enum class with some useful methods."""

    @classmethod
    def all(cls) -> List["Enum"]:
        return [choice for choice in cls]  # pylint: disable=unnecessary-comprehension

    @classmethod
    def values(cls) -> List[int]:
        return [choice.value for choice in cls]


class StrEnum(BaseStrEnum):
    """String Enum class"""


class TextEmbeddingMethods(StrEnum):
    MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
    MPNET = 'sentence-transformers/all-mpnet-base-v2'


class TextProcessor(abc.ABC):
    def __init__(self, text: str):
        self.text = text

    @abc.abstractmethod
    def process(self) -> Any:
        pass


def load_model(method):
    model = SentenceTransformer(method)
    if torch.cuda.is_available():
        model.cuda()
    return model


class TextEmbedderProcessor(TextProcessor):
    """
    This class will embed a text and return its embedding
    """
    METHOD = TextEmbeddingMethods.MINILM
    MODEL = load_model(METHOD)

    def __init__(self, text: str):
        super().__init__(text)

    def process(self) -> Any:
        return TextEmbedderProcessor.MODEL.encode(
            self.text, convert_to_tensor=False
        ).astype("float")
