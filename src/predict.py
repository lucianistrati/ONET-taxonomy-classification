import torch
import torch.nn.functional as F
import numpy as np
import joblib
import json
from src.main import FCNN
from src.text_embedder_processors import TextEmbedderProcessor

# Model configuration
input_size = 384
hidden_size = 512
output_size = 602

# Initialize the model
model = FCNN(input_size, hidden_size, output_size)

# Load the model checkpoint
checkpoint_path = "checkpoints/fcnn.pth"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
model.eval()  # Set the model to evaluation mode

# Load the label encoder
label_encoder_filename = 'checkpoints/label_encoder.joblib'
label_encoder = joblib.load(label_encoder_filename)

# Load mappings from ONET name to label and vice versa
with open("data/onet_name_to_label.json", 'r') as json_file:
    onet_name_to_label = json.load(json_file)

with open("data/onet_label_to_name.json", 'r') as json_file:
    onet_label_to_name = json.load(json_file)


def predict(job_title: str, top_k: int = 5):
    """
    Predicts the top_k ONET labels for a given job title.

    Args:
        job_title (str): The title of the job to predict ONET labels for.
        top_k (int): The number of top predictions to return.
    """
    # Process the job title into an embedding
    embedding = np.array([TextEmbedderProcessor(job_title).process()])

    # Perform a forward pass through the model
    output_tensor = model(torch.from_numpy(embedding).float())

    # Apply softmax to get class probabilities
    class_probs = F.softmax(output_tensor, dim=1)

    # Get the top_k predictions
    top_k_results = torch.topk(class_probs, k=top_k)
    top_k_values = top_k_results.values.detach().numpy()[0]
    top_k_indices = top_k_results.indices.detach().numpy()[0]

    # Display the predictions
    for i, (value, index) in enumerate(zip(top_k_values, top_k_indices)):
        onet_label = label_encoder.inverse_transform([int(index)])[0]
        onet_name = onet_label_to_name[onet_label]
        print("#" * 10)
        print(f"Prediction no. {i + 1}")
        print(f"Confidence: {value:.4f}")  # Format confidence to 4 decimal places
        print("ONET NAME:", onet_name)
        print("ONET LABEL:", onet_label)


def main():
    job_title = "Grocery Order Writer (Buyer / Inventory Replenishment)"
    predict(job_title)


if __name__ == "__main__":
    main()
