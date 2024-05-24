import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import yaml
import os

import sys


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        out = self.hidden1(x)
        out = self.relu(out)
        out = self.hidden2(out)
        out = self.relu(out)
        out = self.output(out)
        return out


def main():

    params = yaml.safe_load(open("params.yaml"))["train_nn"]

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare_nn.py input_path output_path\n")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    # Move the datasets to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    train_data = torch.load(input_path)

    # Extract the tensors and move them to the device
    train_x, train_y = train_data.dataset.tensors[0].to(
        device
    ), train_data.dataset.tensors[1].to(device)

    # Initialize model
    model = NeuralNet(train_x.size(1), params["hidden_size"], 1).to(device)

    # Use the adam optimizer
    lr = params["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define loss function
    criterion = nn.MSELoss()

    # Train the model
    model.train()

    pbar = tqdm(range(params["training_iter"]))

    for i in pbar:
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()

        print(
            "Iter %d/%d - Loss: %.3f"
            % (
                i + 1,
                params["training_iter"],
                loss.item(),
            )
        )
        optimizer.step()

    torch.save(model.state_dict(), output_path)  # save the state of the model

    return model.state_dict()


if __name__ == "__main__":
    main()
