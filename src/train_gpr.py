import torch
import gpytorch

from tqdm import tqdm

import yaml
import os

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

    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare_gpr.py input_path output_path\n")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    params_name = sys.argv[3]

    params = yaml.safe_load(open("params.yaml"))[params_name]

    # Move the datasets to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    train_data = torch.load(input_path)

    # Extract the tensors and move them to the device
    train_x, train_y = train_data.dataset.tensors[0].to(
        device
    ), train_data.dataset.tensors[1].to(device)

    ##############？？？？？
    smoke_test = "CI" in os.environ
    training_iter = 2 if smoke_test else params["training_iter"]

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(train_x, train_y, likelihood).to(device)

    # Use the adam optimizer
    lr = params["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Train the model
    model.train()
    likelihood.train()

    pbar = tqdm(range(training_iter))

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

    torch.save(
        model.state_dict(), output_path
    )  # save the state, include likelihood and model

    return model.state_dict()


if __name__ == "__main__":
    main()
