import torch
import gpytorch
import numpy as np


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def predict(SE, CR, gender):
    # Load the entire model
    model = torch.load("model.pth")
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model.eval()

    # Convert inputs to tensor
    new_sample = torch.tensor([[SE, CR, gender]]).float()

    # Make prediction
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = model.likelihood(model(new_sample))
        predicted_value = observed_pred.mean

    print(
        f"Predicted value for the new sample with SE={SE}, CR={CR}, gender={gender}: {predicted_value.item():.2f}"
    )

    return predicted_value.item()
