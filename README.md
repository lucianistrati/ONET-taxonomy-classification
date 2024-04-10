# ONET taxonomy classification

## Install dependencies

Install all the project dependencies by running the following command in the 
project's root directory:

```bash
poetry install
```

You can use the following command to activate the virtual environment

```bash
poetry shell
```

Note: Make sure you have Poetry installed on your system. If not, you can install it
using:

```bash
pip install poetry
```

## Other prerequisites

Ensure you have the following API keys before stored in the .env file:

- PINECONE_ENVIRONMENT
- PINECONE_API_KEY
- OPEN_AI_API_KEY
- OPEN_AI_API_TYPE
- OPEN_AI_API_VERSION
- OPEN_AI_ENDPOINT

Ensure you have the following folders created: checkpoints, data.

Ensure you have test_data.csv and train_data.csv files in the data folder.

## Run the code

You would need to run to create the checkpoint of the model, the embeddings files 
and the label encoder checkpoint:
```bash
python src/main.py
```

If you already have the mentioned above parts, you can just run this to obtain 
predictions:
```bash
python src/predict.py
```

## MEMO / Documentation

For more details of the implementation you may check "MEMO.md" file

## Issues

### OPEN AI issue

If you encounter any issues with the open ai version, run this:

```bash
pip install openai==0.28
```

### Path issue

If you encounter issues with the relative path, run this:

```bash
export PYTHONPATH=$PWD
```
