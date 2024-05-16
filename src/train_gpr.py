import pandas as pd
import numpy as np
import torch
import gpytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import dvc.api
import yaml
import os
import pickle
import sys


# Define a Gaussian Process Model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def main():

    params = yaml.safe_load(open("params.yaml"))["train_gpr"]

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython train_gpr.py features model\n")
        sys.exit(1)

    input = sys.argv[1]
    output = sys.argv[2]
    # seed = params["seed"]
    # n_est = params["n_est"]
    # min_split = params["min_split"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(input, delimiter=",")
    print(data.columns)
    ###### Data Preprocessing ######
    data["diopter_s"] = pd.to_numeric(data["diopter_s"], errors="coerce")
    data["diopter_c"] = pd.to_numeric(data["diopter_c"], errors="coerce")
    data["CR"] = pd.to_numeric(data["CR"], errors="coerce")

    X = data[["diopter_s", "CR", "gender"]]

    y = data["AL"]

    X = X.dropna()
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    ###
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.values)
    X_test = scaler.transform(X_test.values)
    joblib.dump(scaler, "scaler.pkl")

    # Convert your data to PyTorch tensors and move them to the GPU if available
    train_x = torch.tensor(X_train).float().to(device)
    train_y = torch.tensor(y_train.values).float().to(device)
    test_x = torch.tensor(X_test).float().to(device)
    test_y = torch.tensor(y_test.values).float().to(device)

    import os

    smoke_test = "CI" in os.environ
    training_iter = 2 if smoke_test else 50

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(train_x, train_y, likelihood).to(device)

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Train the model
    model.train()
    likelihood.train()

    pbar = tqdm(range(100))

    for i in pbar:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()

        print(
            "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
            % (
                i + 1,
                training_iter,
                loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item(),
            )
        )
        optimizer.step()
    # Switch to evaluation mode
    model.eval()
    likelihood.eval()

    # Set the device
    device = torch.device("cpu")  # Use 'cuda' for GPU
    model = model.to(device)
    test_x = test_x.to(device)

    # Prediction and confidence interval calculation
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        try:
            lower, upper = observed_pred.confidence_region()
        except Exception as e:
            print("Exception when calculating confidence region:", e)
    # Calculate evaluation metrics
    test_y = test_y.to(device)
    mse = mean_squared_error(test_y.cpu().numpy(), observed_pred.mean.cpu().numpy())
    mae = mean_absolute_error(test_y.cpu().numpy(), observed_pred.mean.cpu().numpy())
    r2 = r2_score(test_y.cpu().numpy(), observed_pred.mean.cpu().numpy())

    print(f"MSE: {mse}, MAE: {mae}, R2 Score: {r2}")

    # Plotting
    plt.figure(figsize=(10, 7))
    plt.scatter(test_y.cpu().numpy(), observed_pred.mean.cpu().numpy())
    plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], "k--")

    print(test_x[:10])
    # Assuming 'lower' and 'upper' are your confidence bounds
    plt.fill_between(test_x_1d, lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5)

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.show()
    # Save the entire model

    torch.save(model, output)

    return "yes!"


def predict_ratio(SE, CR, gender):
    # Load the entire model
    model = torch.load("model.pth")
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model.eval()

    scaler = joblib.load("scaler.pkl")

    # Process input parameters
    inputs = np.array([[SE, CR, gender]])
    inputs = scaler.transform(inputs)  # Apply scaling
    inputs_tensor = torch.tensor(inputs).float()  # Convert to tensor

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(inputs_tensor))

    return observed_pred


if __name__ == "__main__":
    main()
