from typing import Any, Dict, List
import openai
from dotenv import load_dotenv
import pandas as pd
import json
from tqdm import tqdm

load_dotenv()


from os import environ

OPENAI_API_BASE = environ.get("OPEN_AI_ENDPOINT")
OPENAI_API_TYPE = environ.get("OPEN_AI_API_TYPE")
OPENAI_API_KEY = environ.get("OPEN_AI_API_KEY")
OPEN_AI_VERSION = environ.get("OPEN_AI_API_VERSION")


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


class ChatGpt(metaclass=Singleton):
    """ChatGpt communication class"""

    def __init__(self):
        openai.api_base = OPENAI_API_BASE
        openai.api_key = OPENAI_API_KEY
        openai.api_type = OPENAI_API_TYPE
        openai.api_version = OPEN_AI_VERSION

    @staticmethod
    def get_output_from_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
        output_messages = []
        choices = response["choices"]
        for choice in choices:
            message = choice["message"]
            content = message["content"]
            role = message["role"]
            output_message = {"content": content, "role": role}
            output_messages.append(output_message)
        return output_messages

    @staticmethod
    def default_configuration():
        return {
            "engine": "gpt35",
            "temperature": 0.01,
            "max_tokens": 200,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

    def request(self, messages: List[Dict[Any, Any]], **kwargs: Any) -> Any:
        return openai.ChatCompletion.create(
            messages=messages, **{**self.default_configuration(), **kwargs}
        )


GPT = ChatGpt()


def openai_answer(query):
    messages = [
        {"role": "system", "content": query},
    ]
    return GPT.request(messages, temperature=0.01)["choices"][0]["message"]["content"]


def main():
    test_df = pd.read_csv("data/test_data.csv")
    job_titles = test_df["TITLE_RAW"].to_list()
    job_descriptions = test_df["BODY"].to_list()
    labels = test_df["ONET"].to_list()

    with open("data/answers.json", 'r') as json_file:
        answers = json.load(json_file)

    correct = 0
    for (answer, label) in zip(answers, labels):
        if label.lower() in answer.lower():
            correct += 1

    print(correct/len(answers))

    with open("data/top_10_most_similar.json", 'r') as json_file:
        similar_onet_values_list = json.load(json_file)

    answers = []
    correct = 0
    for (job_title, job_description, similar_onet_values, label) in tqdm(zip(job_titles,
                                                                      job_descriptions,
                                                                      similar_onet_values_list,
                                                                             labels)):
        query = f"""Given this job title: {job_title},
                    this job description:{job_description[:min(20_000, len(job_description))]}
                    and the following 10
                    choices for a ONET taxonomy of this job: {similar_onet_values} pick
                    only the first choice out of those 10 choices that you
                    see as the best fit!"""
        answers.append(openai_answer(query))
        with open("data/answers_with_top_10.json", 'w') as json_file:
            json.dump(answers, json_file)

    for (answer, label) in zip(answers, labels):
        if label.lower() in answer.lower():
            correct += 1

    print(correct/len(answers))


if __name__ == "__main__":
    main()
