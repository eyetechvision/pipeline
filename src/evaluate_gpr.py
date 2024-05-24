import sys
import torch
import gpytorch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from dvclive import Live


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def evaluate(model, likelihood, features, labels, split, live, save_path):
    """
    Dump all evaluation metrics and plots for given datasets.

    Args:
        model (gpytorch.models.ExactGP): Trained regressor.
        likelihood (gpytorch.likelihoods.GaussianLikelihood): Likelihood function used with the model.
        features (torch.Tensor): Input features.
        labels (torch.Tensor): True labels.
        split (str): Dataset name.
        live (dvclive.Live): Dvclive instance.
        save_path (str): Path to save the metrics.
    """
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(features))

    # Use dvclive to log a few simple metrics...
    mse = mean_squared_error(labels.detach().numpy(), predictions.mean.detach().numpy())
    r2 = r2_score(labels.detach().numpy(), predictions.mean.detach().numpy())
    if not live.summary:
        live.summary = {"mse": {}, "r2": {}}
    live.summary["mse"][split] = mse
    live.summary["r2"][split] = r2

    # ... and plots...
    # ... like a residuals plot...
    residuals = labels.detach().numpy() - predictions.mean.detach().numpy()
    fig, ax = plt.subplots()
    ax.scatter(predictions.mean.detach().numpy(), residuals)
    ax.hlines(y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors="r")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    live.log_image(f"residuals/{split}.png", fig)


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

    # Create a new GP model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(test_x, test_y, likelihood)

    # Load the state into the model
    model.load_state_dict(state_dict)

    # Print the state of the model
    print(model.state_dict())

    # Evaluate train and test datasets.
    # with Live(EVAL_PATH) as live:
    with Live(EVAL_PATH, dvcyaml=False) as live:
        evaluate(model, likelihood, test_x, test_y, "test", live, save_path=EVAL_PATH)


if __name__ == "__main__":
    main()
