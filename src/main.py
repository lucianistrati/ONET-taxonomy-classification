import os.path
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix)
from nltk.tokenize import word_tokenize
from typing import Set, List, Any
from matplotlib import pyplot as plt
from statistics import mean, median
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
from src.vectordb_api import query_database, VectorDBAPI
import numpy as np
import pandas as pd

import json


def flatten(my_list):
    return [val for sub_list in my_list for val in sub_list]


def encode(sentence: str, model) -> List[float]:
    return model.encode([sentence])


def filter_off_oov(text: str, vocab: Set[str]) -> str:
    tokens = word_tokenize(text)
    return " ".join([token for token in tokens if token in vocab])


def plot_histogram(data, title):
    plt.hist(data, bins=100, color='blue', alpha=0.7)

    # Add labels and title
    plt.xlabel('Number of characters')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.savefig(f"plots/{title}.png")
    plt.show()


def encode_datapoints(datapoints: List[str], filepath: str, model: Any):
    if os.path.exists(filepath):
        return
    arr = np.array([encode(datapoint, model) for datapoint in datapoints])
    np.save(file=filepath, arr=arr, allow_pickle=True)


def create_embeddings(train_df: pd.DataFrame, test_df: pd.DataFrame, onet_vocab: Set[
    str]):
    features = ["TITLE_RAW", "BODY"] * 3
    label = "ONET_NAME"
    use_mpnet_options = [False, True, False] * 2
    use_oov_options = [True, False, False] * 2
    for (feature, use_mpnet, use_oov) in zip(features, use_mpnet_options,
                                             use_oov_options):

        if use_mpnet:
            model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        else:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        X_train, X_test = train_df[feature].to_list(),  test_df[feature].to_list()
        y_train, y_test = train_df[label].to_list(), test_df[label].to_list()

        if not use_mpnet:
            X_train = [filter_off_oov(text.lower(), onet_vocab) for text in X_train]
            X_test = [filter_off_oov(text.lower(), onet_vocab) for text in X_test]

        x_ending = f"_{feature}"
        if not use_oov and not use_mpnet:
            pass
        elif use_oov and not use_mpnet:
            x_ending += "_oov"
        elif not use_oov and use_mpnet:
            x_ending += "_mpnet"
        elif use_oov and use_mpnet:
            continue
        y_ending = "_mpnet" if use_mpnet else ""
        print(use_mpnet, use_oov, feature, x_ending, y_ending)

        encode_datapoints(X_train, f"data/X_train{x_ending}.npy", model)
        encode_datapoints(X_test, f"data/X_test{x_ending}.npy", model)
        encode_datapoints(y_test, f"data/y_test{y_ending}.npy", model)


class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'embedding': torch.FloatTensor(self.embeddings[idx]),
            'label': torch.LongTensor([self.labels[idx]])
        }


class FCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


def train_nn(train_embeddings, train_labels, test_embeddings, test_labels):
    # Train-validation split
    train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(
        train_embeddings, train_labels, test_size=0.02, random_state=42
    )

    # Create instances of the custom dataset
    train_dataset = EmbeddingsDataset(train_embeddings, train_labels)
    val_dataset = EmbeddingsDataset(val_embeddings, val_labels)
    test_dataset = EmbeddingsDataset(test_embeddings, test_labels)

    # DataLoader for training and validation
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Set random seed for reproducibility
    torch.manual_seed(13)

    # Hyperparameters
    input_size = train_embeddings.shape[1]  # Size of Sentence Transformer embeddings
    hidden_size = 512
    output_size = max(train_labels) + 1
    learning_rate = 0.001
    epochs = 15

    # Initialize the model, loss function, and optimizer
    model = FCNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        model.train()  # Set the model to training mode
        for batch in train_loader:
            inputs, labels = batch['embedding'], batch['label'].squeeze()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()

        # Print training loss after each epoch
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Training Loss: {total_loss / len(train_loader)}")

        val_loss = 0
        # Validation loop after each epoch
        with torch.no_grad():
            model.eval()  # Set the model to evaluation mode
            correct = 0
            total = 0
            for val_batch in val_loader:
                val_inputs, val_labels = val_batch['embedding'], val_batch[
                    'label'].squeeze()

                # Forward pass
                val_outputs = model(val_inputs)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()

                _, predicted = torch.max(val_outputs.data, 1)

                # Accuracy calculation
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()

            accuracy = correct / total
            print(f'Validation Accuracy after Epoch {epoch + 1}: {accuracy}')
            print(f'Validation Loss after Epoch {epoch + 1}: {val_loss}')

    # Testing loop
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for batch in test_loader:
            inputs, labels = batch['embedding'], batch['label'].squeeze()

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Accuracy calculation
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy}')

    print("Training finished!")

    # Save the trained model
    torch.save(model.state_dict(), "checkpoints/fcnn.pth")


def main():
    train_df = pd.read_csv("data/train_data.csv")
    test_df = pd.read_csv("data/test_data.csv")

    onet_label_to_name = dict()
    for (label, name) in zip(set(train_df["ONET"].to_list() +
                                 test_df["ONET"].to_list()),
                             set(train_df["ONET_NAME"].to_list() +
                                 test_df["ONET_NAME"].to_list())):
        if label not in onet_label_to_name:
            onet_label_to_name[label] = name
    onet_name_to_label = {name: label for (label, name) in onet_label_to_name.items()}

    with open("data/onet_name_to_label.json", 'w') as json_file:
        json.dump(onet_name_to_label, json_file)

    with open("data/onet_label_to_name.json", 'w') as json_file:
        json.dump(onet_label_to_name, json_file)

    onet_choices = set(train_df["ONET_NAME"].to_list() +
    test_df["ONET_NAME"].to_list())
    onet_vocab = set(flatten([word_tokenize(val.lower()) for val in onet_choices]))

    body_lenghts = [len(text) for text in train_df["BODY"].to_list() + test_df[
        "BODY"].to_list()]
    title_lenghts = [len(text) for text in train_df["TITLE_RAW"].to_list() + test_df[
        "TITLE_RAW"].to_list()]
    # many of them appear like only once or twice

    print(len(set(train_df["ONET"].to_list())), Counter(train_df["ONET"].to_list()))
    print(len(set(test_df["ONET"].to_list())), Counter(test_df["ONET"].to_list()))

    all_possible_classes = list(set(test_df["ONET"].to_list()).
                            union(set(train_df["ONET"].to_list())))
    use_mpnet = True
    if use_mpnet:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    else:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    encode_datapoints(all_possible_classes, "data/all_possible_classes_mpnet.npy",
                      model)
    # 699 train, 708 in test set, 805 in both train+test, so about 100 which are in
    # one and not in the other, many are only once

    print(min(body_lenghts), max(body_lenghts), mean(body_lenghts),
          median(body_lenghts))
    print(min(title_lenghts), max(title_lenghts), mean(title_lenghts),
          median(title_lenghts))

    plot_histogram(body_lenghts, "Body lengths")
    plot_histogram(title_lenghts, "Title Lengths")

    create_embeddings(train_df, test_df, onet_vocab)

    endings = ["_BODY", "_BODY_mpnet", "_BODY_oov", "_TITLE_RAW", "_TITLE_RAW_mpnet",
               "_TITLE_RAW_oov"]
    model_option = "NN"
    vector_db_api = VectorDBAPI()
    for ending in endings:
        X_train = np.load(f"data/X_train{ending}.npy")
        X_test = np.load(f"data/X_test{ending}.npy")
        y_train = train_df["ONET"].to_list()
        y_test = test_df["ONET"].to_list()
        label_encoder = LabelEncoder()
        common_labels = list(set(y_train).intersection(set(y_test)))
        label_encoder.fit(common_labels)
        X_train = [x for (x, label) in zip(X_train, y_train) if label in
                   common_labels]
        X_test = [x for (x, label) in zip(X_test, y_test) if label in
                   common_labels]
        original_y_test = deepcopy(y_test)
        y_train = [y for y in y_train if y in common_labels]
        y_test = [y for y in y_test if y in common_labels]
        y_train = label_encoder.transform(y_train)
        y_test = label_encoder.transform(y_test)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        X_train = np.squeeze(X_train, axis=1)
        X_test = np.squeeze(X_test, axis=1)

        filename = 'checkpoints/label_encoder.joblib'
        joblib.dump(label_encoder, filename)

        if model_option == "XGB":
            # model = LogisticRegression(class_weight="balanced")
            # model = XGBClassifier()
            model = SVC(class_weight="balanced")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("#" * 150)
            print("Files extension:", ending)
            print("Accuracy:", accuracy_score(y_test, y_pred))
            print("Precision:", precision_score(y_test, y_pred, average="weighted"))
            print("Recall:", recall_score(y_test, y_pred, average="weighted"))
            print("F1:", f1_score(y_test, y_pred, average="weighted"))
            joblib.dump(model, 'checkpoints/xgb_classifier.joblib')
        elif model_option == "NN":
            train_nn(X_train, y_train, X_test, y_test)
        elif model_option == "SIMILARITY":
            correct = 0
            i = 0
            for (emb, label) in tqdm(zip(X_test, original_y_test)):
                i += 1
                result = query_database(vector_db_api, emb.tolist())
                correct += int(onet_name_to_label[result[0]["label"]] == label)
                if i % 100 == 0:
                    print(correct, correct / i)
            print("Accuracy: ", correct / len(X_test))
        else:
            raise ValueError(f"Wrong model_option: {model_option}!")


if __name__ == "__main__":
    main()
