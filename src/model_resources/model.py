# import tensorflow as tf
# from tensorflow import keras
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader

os.system("cls")

SEQUENCE_LENGTH = 30
NUM_FEATURES = 5

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.3, device="cuda"):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, (h, c) = self.lstm(x, (h0, c0))
        h = h.detach()
        c = c.detach()
        out = self.fc(out[:, -1, :])  # last time step output
        return out

class Network_Utils:

    def __init__(self):
        self.main_network = None
        self.l_r = 0.0001
        self.epoch = 50
        self.input_size = 5
        self.hidden_size = 30
        self.output_size = 3
        self.device = "cpu"
        # self.load_model_default()

    def build_model(self, model_path):
        self.main_network = StockLSTM(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimiser = optim.Adam(self.main_network.parameters(), lr=self.l_r)
        torch.save(self.main_network.state_dict(), model_path)

    def train(self, x, y,model_path):
        self.load_model(model_path=model_path)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.main_network.to(self.device)
        for epoch in range(self.epoch):

            total_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                self.optimiser.zero_grad()
                output = self.main_network(X_batch)
                loss = self.loss_function(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), 1.0)
                self.optimiser.step()

                total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{self.epoch}], Loss: {total_loss / len(loader):.4f}")

        torch.save(self.main_network.state_dict(), "model_rnn_1.pth")



    def load_model_default(self):
        self.main_network = StockLSTM(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.main_network.load_state_dict(torch.load("model_rnn_1.pth"))
        self.loss_function = nn.CrossEntropyLoss()
        self.optimiser = optim.Adam(self.main_network.parameters(), lr=self.l_r)

    def load_model(self, model_path:str):
        self.main_network = StockLSTM(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.main_network.load_state_dict(torch.load(model_path))
        self.main_network = self.main_network.to(self.device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimiser = optim.Adam(self.main_network.parameters(), lr=self.l_r)

    def output(self, x):
        self.main_network.eval()
        with torch.no_grad():
            output = self.main_network(x)
            # print("Model raw output..", output)
        return output
    # Add this function to model.py

def check_accuracy(model, x_test, y_test, device="cpu"):
        """ Checks the model's accuracy on unseen test data. """
        print("--- Checking accuracy on test data... ---")

        # Set the model to evaluation mode
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        model = model.to(device)
        model.eval()

        # Disable gradient calculations for efficiency
        with torch.no_grad():
            # Get the model's raw output (logits) for the entire test set
            outputs = model(x_test)

            # Find the class with the highest score for each prediction
            # torch.max returns (values, indices)
            _, predictions = torch.max(outputs, 1)

            # Count how many predictions match the true labels
            num_correct = (predictions == y_test).sum()
            num_samples = predictions.size(0)

            # Calculate the accuracy percentage
            accuracy = (num_correct.item() * 100) / num_samples

            print(f"Got {num_correct} / {num_samples} correct ({accuracy:.2f}%)")

        # Set the model back to training mode
        model.train()



if __name__ == "__main__":
    import torch
    import numpy as np
    from sklearn.model_selection import train_test_split
    import os

    DEVICE = "cuda"
    # === Personality Masks ===
    PERSONALITY_MASKS = {
        "Momentum Trader": torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool),
        "Trend Follower":  torch.tensor([0, 1, 0, 1, 1], dtype=torch.bool),
        "Hybrid Trader":   torch.tensor([1, 0, 1, 0, 1], dtype=torch.bool),
        "All-Rounder":     torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool),
    }

    # === Load dataset once ===
    x_list, y_list = [], []
    for i in range(0, 30):
        training_data_path = rf"D:\NetworkPrediction\data\training_data\training_data{i}.npy"
        data_matrix = np.load(training_data_path, allow_pickle=True)
        x_list.extend(row[0] for row in data_matrix)
        y_list.extend(row[1] + 1 for row in data_matrix)  # [-1,0,1] → [0,1,2]

    x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32, device=DEVICE)
    y_tensor = torch.tensor(y_list, dtype=torch.long, device=DEVICE)

    print(f"Dataset loaded: X={x_tensor.shape}, y={y_tensor.shape}")
    print(f"NaN: {torch.isnan(x_tensor).any()}, Inf: {torch.isinf(x_tensor).any()}")

    # === Train a separate model per personality ===
    for personality_name, mask in PERSONALITY_MASKS.items():
        print(f"\n=== Training for {personality_name} ===")

        # Apply personality mask to input features
        # (broadcasts mask across sequence dimension)
        x_masked = x_tensor.clone()
        x_masked[:, :, ~mask] = 0  # zero out features not used by this personality

        # Split dataset
        x_train, x_test, y_train, y_test = train_test_split(
            x_masked, y_tensor, test_size=0.2, random_state=42
        )

        # Initialize and train
        obj = Network_Utils()
        # remeber to update the device
        # eg:
        # obj.device = "cpu"
        # obj.device = "cuda"
        model_save_path = rf"D:\NetworkPrediction\data\models\retail\{personality_name}.pth"

        obj.build_model(model_path=model_save_path)
        obj.train(x_train, y_train, model_path=model_save_path)

        # Check test accuracy
        check_accuracy(obj.main_network, x_test, y_test)
        print(f"✅ Saved {personality_name} → {model_save_path}")
