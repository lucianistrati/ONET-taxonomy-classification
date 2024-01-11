from typing import Optional

import pinecone

from text_embedder_processors import TextEmbedderProcessor
from os import environ
from dotenv import load_dotenv

from typing import Any, List
from tqdm import tqdm

import numpy as np
import pandas as pd
import json

load_dotenv()


def flatten(nested_list: List[Any]) -> List[Any]:
    flat_list = []
    stack = [nested_list]

    while stack:
        current = stack.pop()
        for item in current:
            if isinstance(item, list):
                stack.append(item)
            else:
                flat_list.append(item)

    return flat_list


class Singleton(type):
    """
    Singleton metaclass.
    This can be added to any class to make it a singleton.

    e.g.
    class GlobalSettings(metaclass=Singleton):
        pass

    usage:
    settings1 = GlobalSettings()
    settings2 = GlobalSettings()
    settings1 is settings2  # True
    """

    _instances = {}  # type: ignore

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


PINECONE_API_KEY = environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = environ.get("PINECONE_ENVIRONMENT")


class VectorDBAPI(metaclass=Singleton):
    """
    Class that gets requests results from Pinecone's Vector DB
    """

    def __init__(
        self,
        index_name: str = "main",
        dimension: int = 384,
        metric: str = "cosine",
        renew_index: bool = False,
    ):
        self.set_pinecone_api()
        self.index_name = index_name
        self.renew_index = renew_index
        self.metric = metric
        self.dimension = dimension
        if self.index_name not in pinecone.list_indexes():
            self.create_index()
        elif renew_index:
            self.delete_index()
            self.create_index()
        self.index = pinecone.Index(index_name=index_name)
        self.text_embedder = TextEmbedderProcessor("")

    @staticmethod
    def set_pinecone_api():
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

    def create_index(self) -> Any:
        return pinecone.create_index(
            self.index_name, dimension=self.dimension, metric=self.metric
        )

    def insert(self, text: str, embedding: Optional[Any] = None) -> Any:
        if embedding is None:
            self.text_embedder.text = text
            embedding = list(flatten(self.text_embedder.process()))
        self.index.upsert([(text, embedding)])
        return embedding

    def query(self, embedding: Optional[Any] = None, top_k: int = 5) -> Any:
        return self.index.query(vector=embedding, top_k=top_k)

    def delete_index(self) -> Any:
        return pinecone.delete_index(self.index_name)


def populate_vector_db():
    train_df = pd.read_csv("data/train_data.csv")
    test_df = pd.read_csv("data/test_data.csv")

    texts = list(set(test_df["ONET_NAME"].to_list()).
                                union(set(train_df["ONET_NAME"].to_list())))
    use_mpnet = False
    if use_mpnet:
        vector_db = VectorDBAPI(renew_index=True, dimension=768)
        embeddings = np.load("data/all_possible_classes.npy")
    else:
        vector_db = VectorDBAPI(renew_index=True)
        embeddings = np.load("data/all_possible_classes.npy")
    for (text, embedding) in tqdm(zip(texts, embeddings)):
        embedding = flatten(embedding)
        vector_db.insert(text=text, embedding=embedding[0].tolist())


def query_database(vector_db_api: Any, embedding: Any, top_k: int = 1):
    try:
        output = vector_db_api.query(embedding=embedding, top_k=top_k)
        matches = output["matches"]
        result = [{"label": match["id"],
                   "score": match["score"]}
                  for match in matches]
        return result
    except pinecone.core.client.exceptions.ServiceException:
        return [{"label": "29-1212.00"}]


def main():
    test_df = pd.read_csv("data/test_data.csv")
    job_titles = test_df["TITLE_RAW"].to_list()[:200]
    top_10_most_similar = []
    vector_db_api = VectorDBAPI()
    embs = np.load("data/X_test_TITLE_RAW.npy", allow_pickle=True)
    
    for (job_title, emb) in tqdm(zip(job_titles, embs)):
        result = query_database(vector_db_api, emb.tolist(), top_k=10)
        top_10_most_similar.append([elem["label"] for elem in result])
        with open("data/top_10_most_similar.json", 'w') as json_file:
            json.dump(top_10_most_similar, json_file)

if __name__ == "__main__":
    main()
