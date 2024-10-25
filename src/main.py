import os
import json
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import Counter
from typing import Set, List, Any
from matplotlib import pyplot as plt
from statistics import mean, median
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
from src.vectordb_api import query_database, VectorDBAPI


def flatten(my_list: List[List[Any]]) -> List[Any]:
    """
    Flattens a list of lists into a single list.

    Args:
        my_list (List[List[Any]]): A list of lists.

    Returns:
        List[Any]: A flattened list containing all values from the sub-lists.
    """
    return [val for sub_list in my_list for val in sub_list]


def encode(sentence: str, model) -> List[float]:
    """
    Encodes a single sentence using a specified model.

    Args:
        sentence (str): The sentence to encode.
        model: The model used for encoding.

    Returns:
        List[float]: A list of floats representing the encoded sentence.
    """
    return model.encode([sentence])


def filter_off_oov(text: str, vocab: Set[str]) -> str:
    """
    Filters out out-of-vocabulary (OOV) tokens from the input text.

    Args:
        text (str): The input text to filter.
        vocab (Set[str]): A set of valid vocabulary tokens.

    Returns:
        str: The filtered text containing only tokens in the vocabulary.
    """
    tokens = word_tokenize(text)
    return " ".join([token for token in tokens if token in vocab])


def plot_histogram(data: List[int], title: str) -> None:
    """
    Plots a histogram of the given data.

    Args:
        data (List[int]): The data to plot.
        title (str): The title of the plot.
    """
    plt.hist(data, bins=100, color='blue', alpha=0.7)

    # Add labels and title
    plt.xlabel('Number of characters')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.savefig(f"plots/{title}.png")
    plt.show()


def encode_datapoints(datapoints: List[str], filepath: str, model: Any) -> None:
    """
    Encodes a list of data points and saves them to a file if it doesn't already exist.

    Args:
        datapoints (List[str]): The data points to encode.
        filepath (str): The file path to save the encoded data.
        model: The model used for encoding.
    """
    if os.path.exists(filepath):
        return
    arr = np.array([encode(datapoint, model) for datapoint in datapoints])
    np.save(file=filepath, arr=arr, allow_pickle=True)


def create_embeddings(train_df: pd.DataFrame, test_df: pd.DataFrame, onet_vocab: Set[str]) -> None:
    """
    Creates embeddings for training and testing data frames using specified vocabulary.

    Args:
        train_df (pd.DataFrame): The training data frame.
        test_df (pd.DataFrame): The testing data frame.
        onet_vocab (Set[str]): A set of vocabulary tokens.
    """
    features = ["TITLE_RAW", "BODY"] * 3
    label = "ONET_NAME"
    use_mpnet_options = [False, True, False] * 2
    use_oov_options = [True, False, False] * 2

    for (feature, use_mpnet, use_oov) in zip(features, use_mpnet_options, use_oov_options):
        # Select the SentenceTransformer model based on options
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2' if use_mpnet 
                                     else 'sentence-transformers/all-MiniLM-L6-v2')

        # Prepare training and testing datasets
        X_train, X_test = train_df[feature].to_list(), test_df[feature].to_list()
        y_train, y_test = train_df[label].to_list(), test_df[label].to_list()

        # Filter OOV tokens if specified
        if not use_mpnet:
            X_train = [filter_off_oov(text.lower(), onet_vocab) for text in X_train]
            X_test = [filter_off_oov(text.lower(), onet_vocab) for text in X_test]

        # Determine file suffix based on encoding options
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

        # Encode and save the data points
        encode_datapoints(X_train, f"data/X_train{x_ending}.npy", model)
        encode_datapoints(X_test, f"data/X_test{x_ending}.npy", model)
        encode_datapoints(y_test, f"data/y_test{y_ending}.npy", model)


class EmbeddingsDataset(Dataset):
    """
    Custom Dataset class for embedding data.

    Args:
        embeddings: The embeddings for the dataset.
        labels: The corresponding labels for the embeddings.
    """

    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves the embedding and label at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the embedding and label.
        """
        return {
            'embedding': torch.FloatTensor(self.embeddings[idx]),
            'label': torch.LongTensor([self.labels[idx]])
        }


class FCNN(nn.Module):
    """
    A simple Feedforward Neural Network for classification.

    Args:
        input_size: The number of input features.
        hidden_size: The number of hidden units.
        output_size: The number of output classes.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the network."""
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


def train_nn(train_embeddings: np.ndarray, train_labels: np.ndarray,
             test_embeddings: np.ndarray, test_labels: np.ndarray) -> None:
    """
    Trains a Feedforward Neural Network using the provided embeddings and labels.

    Args:
        train_embeddings (np.ndarray): Training embeddings.
        train_labels (np.ndarray): Training labels.
        test_embeddings (np.ndarray): Testing embeddings.
        test_labels (np.ndarray): Testing labels.
    """
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

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print average loss for the epoch
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch['embedding'], batch['label'].squeeze()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()

        # Print validation loss and accuracy
        print(f'Validation Loss: {val_loss / len(val_loader):.4f}, '
              f'Validation Accuracy: {val_correct / len(val_dataset):.4f}')

    # Testing phase
    model.eval()
    test_correct = 0
    test_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch['embedding'], batch['label'].squeeze()
            outputs = model(inputs)
            test_predictions.extend(outputs.argmax(dim=1).cpu().numpy())

    # Save the model
    torch.save(model.state_dict(), 'models/model.pt')

    # Compute and print final test accuracy
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f'Test Accuracy: {test_accuracy:.4f}')


def load_model() -> FCNN:
    """
    Loads the trained model from file.

    Returns:
        FCNN: The trained model instance.
    """
    model = FCNN(input_size=768, hidden_size=512, output_size=9)
    model.load_state_dict(torch.load('models/model.pt'))
    return model


def evaluate_model(test_embeddings: np.ndarray, test_labels: np.ndarray) -> None:
    """
    Evaluates the model on the test dataset.

    Args:
        test_embeddings (np.ndarray): Test embeddings.
        test_labels (np.ndarray): Test labels.
    """
    model = load_model()
    model.eval()

    test_loader = DataLoader(EmbeddingsDataset(test_embeddings, test_labels), 
                             batch_size=64, shuffle=False)

    test_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['embedding']
            outputs = model(inputs)
            test_predictions.extend(outputs.argmax(dim=1).cpu().numpy())

    # Calculate evaluation metrics
    acc = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions, average='weighted')
    recall = recall_score(test_labels, test_predictions, average='weighted')
    f1 = f1_score(test_labels, test_predictions, average='weighted')
    cm = confusion_matrix(test_labels, test_predictions)

    # Print evaluation results
    print(f'Accuracy: {acc:.4f}, Precision: {precision:.4f}, '
          f'Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print('Confusion Matrix:\n', cm)


def svm_classifier(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Trains and evaluates a Support Vector Machine (SVM) classifier.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing labels.
    """
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    # Calculate evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    # Print evaluation results
    print(f'SVM Accuracy: {acc:.4f}, Precision: {precision:.4f}, '
          f'Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print('Confusion Matrix:\n', cm)


def get_unique_labels(train_labels: np.ndarray) -> List[str]:
    """
    Returns unique labels from the training labels.

    Args:
        train_labels (np.ndarray): The training labels.

    Returns:
        List[str]: A list of unique labels.
    """
    return list(set(train_labels))


def create_and_save_label_encoder(train_labels: np.ndarray) -> None:
    """
    Creates a label encoder from the training labels and saves it to a file.

    Args:
        train_labels (np.ndarray): The training labels.
    """
    le = LabelEncoder()
    le.fit(train_labels)
    joblib.dump(le, 'models/label_encoder.pkl')


def load_label_encoder() -> LabelEncoder:
    """
    Loads the label encoder from file.

    Returns:
        LabelEncoder: The label encoder instance.
    """
    return joblib.load('models/label_encoder.pkl')


def save_predictions_to_json(predictions: List[str], output_file: str) -> None:
    """
    Saves the model predictions to a JSON file.

    Args:
        predictions (List[str]): The model predictions.
        output_file (str): The output file path.
    """
    with open(output_file, 'w') as f:
        json.dump(predictions, f)
        print(f"Predictions saved to {output_file}")


def main() -> None:
    """
    Main function to execute the workflow of data processing, model training, 
    evaluation, and predictions.
    """
    # Load and process data
    train_df, test_df = load_data()
    onet_vocab = load_vocab()
  
    # Call functions for creating embeddings
    create_embeddings(train_df, test_df, onet_vocab)
    train_embeddings = np.load('data/X_train.npy')
    train_labels = np.load('data/y_train.npy')
    test_embeddings = np.load('data/X_test.npy')
    test_labels = np.load('data/y_test.npy')

    # Train neural network
    train_nn(train_embeddings, train_labels, test_embeddings, test_labels)

    # Evaluate model
    evaluate_model(test_embeddings, test_labels)

    # Save predictions
    save_predictions_to_json(test_predictions, 'predictions.json')


if __name__ == "__main__":
    main()
