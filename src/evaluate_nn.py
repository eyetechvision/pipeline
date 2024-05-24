import sys
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from dvclive import Live
from train_nn import NeuralNet
import yaml


def evaluate(network_model, features, labels, split, live, save_path):
    """
    Dump all evaluation metrics and plots for given datasets.

    Args:
        network_model (NeuralNet): Trained regressor.
        features (torch.Tensor): Input features.
        labels (torch.Tensor): True labels.
        split (str): Dataset name.
        live (dvclive.Live): Dvclive instance.
        save_path (str): Path to save the metrics.
    """
    network_model.eval()
    with torch.no_grad():
        predictions = network_model(features)

    mse = mean_squared_error(labels.detach().numpy(), predictions.detach().numpy())
    r2 = r2_score(labels.detach().numpy(), predictions.detach().numpy())
    if not live.summary:
        live.summary = {"mse": {}, "r2": {}}
    live.summary["mse"][split] = mse
    live.summary["r2"][split] = r2


def main():

    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython evaluate.py model features\n")
        sys.exit(1)

    model_path_name = sys.argv[1]
    testdata_path_name = sys.argv[2]
    EVAL_PATH = sys.argv[3]

    state_dict = torch.load(model_path_name)
    test_data = torch.load(testdata_path_name)

    test_x, test_y = test_data.dataset.tensors[0], test_data.dataset.tensors[1]

    params = yaml.safe_load(open("params.yaml"))["train_nn"]

    network_model = NeuralNet(test_x.size(1), params["hidden_size"], 1)

    # Load the state into the model
    network_model.load_state_dict(state_dict)

    # Print the state of the model
    print(network_model.state_dict())

    # Evaluate train and test datasets.
    with Live(EVAL_PATH, dvcyaml=False) as live:
        evaluate(network_model, test_x, test_y, "test", live, save_path=EVAL_PATH)


if __name__ == "__main__":
    main()
