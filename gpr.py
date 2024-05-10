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


def train_model():
    data = pd.read_csv("df_high.csv", delimiter=",")
    print(data.columns)

    data["diopter_s"] = pd.to_numeric(data["diopter_s"], errors="coerce")
    data["diopter_c"] = pd.to_numeric(data["diopter_c"], errors="coerce")
    data["CR"] = pd.to_numeric(data["CR"], errors="coerce")
    data["SE"] = data["diopter_s"] + data["diopter_c"] / 2
    data["ratio"] = data["AL"] / data["CR"]

    X = data[["SE", "CR", "gender"]]
    y = data["ratio"]

    X = X.dropna()
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.values)
    X_test = scaler.transform(X_test.values)
    joblib.dump(scaler, "scaler.pkl")

    # Convert your data to PyTorch tensors
    train_x = torch.tensor(X_train).float()
    train_y = torch.tensor(y_train.values).float()
    test_x = torch.tensor(X_test).float()
    test_y = torch.tensor(y_test.values).float()

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Train the model
    model.train()
    likelihood.train()

    pbar = tqdm(range(100))

    for i in pbar:
        print(f"Fitting iteration {i+1}")
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        # Update tqdm postfix with the current loss
        pbar.set_postfix({"loss": loss.item()})

    # Switch to eval mode
    model.eval()
    likelihood.eval()

    ##################################
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        y_pred = observed_pred.mean

    mse = mean_squared_error(test_y, y_pred)
    mae = mean_absolute_error(test_y, y_pred)
    r2 = r2_score(test_y, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Coefficient of Determination: {r2}")

    # Plotting predictions vs actual
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=test_y, y=y_pred)
    plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], "k--", lw=4)
    plt.xlabel("Measured")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual Values")

    # Residual plot
    residuals = test_y - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Values")

    plt.show()

    # Save the entire model
    torch.save(model, "model.pth")

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
        predicted_value = observed_pred.mean

    return predicted_value.item()


if __name__ == "__main__":
    train_model()
