from src.main import FCNN

import torch

from src.text_embedder_processors import TextEmbedderProcessor
import torch.nn.functional as F
import numpy as np
import joblib
import json


input_size = 384
hidden_size = 512
output_size = 602
model = FCNN(input_size, hidden_size, output_size)
checkpoint_path = "checkpoints/fcnn.pth"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
model.eval()

filename = 'checkpoints/label_encoder.joblib'
label_encoder = joblib.load(filename)

with open("data/onet_name_to_label.json", 'r') as json_file:
    onet_name_to_label = json.load(json_file)

with open("data/onet_label_to_name.json", 'r') as json_file:
    onet_label_to_name = json.load(json_file)


def predict(job_title: str, top_k: int = 5):

    embedding = np.array([TextEmbedderProcessor(job_title).process()])
    output_tensor = model(torch.from_numpy(embedding).float())
    class_probs = F.softmax(output_tensor, dim=1)
    top_k = torch.topk(class_probs, k=top_k)
    top_k_values = top_k.values.detach().numpy()[0]
    top_k_indices = top_k.indices.detach().numpy()[0]
    for i, (value, index) in enumerate(zip(top_k_values, top_k_indices)):
        onet_label = label_encoder.inverse_transform([int(index)])[0]
        onet_name = onet_label_to_name[onet_label]
        print("#" * 10)
        print(f"Prediction no. {i + 1}")
        print(f"Confidence: {value}")
        print("ONET NAME:", onet_name)
        print("ONET LABEL:", onet_label)


job_title = "Grocery Order Writer (Buyer / Inventory Replenishment)"

predict(job_title)

