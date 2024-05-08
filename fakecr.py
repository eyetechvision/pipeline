import pandas as pd
import numpy as np
from joblib import load, dump
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import norm

MODEL_PATH = r"D:\pipline\pipeline\cr_distribution_params.pkl"
FILE_PATH = r"D:\pipline\pipeline\df_high.csv"


def load_data(filepath):
    """Load and return the dataset from the specified CSV file."""
    return pd.read_csv(filepath)


def prepare_data(data):
    """Extracts features and target from the dataset."""
    X = data[["age", "gender"]]
    y = data["CR"]
    return X, y


def build_distribution(y):
    """Fit a Gaussian distribution to the target data."""
    mu, std = norm.fit(y)
    return mu, std


def save_distribution(params, filepath):
    """Saves the distribution parameters to the specified filepath."""
    dump(params, filepath)


def load_distribution(filepath):
    """Loads the distribution parameters from the specified filepath."""
    return load(filepath)


def sample_prediction(n_samples=1):
    """Generate predictions by sampling from the distribution, loading parameters automatically."""
    mu, std = load_distribution(MODEL_PATH)
    return norm.rvs(loc=mu, scale=std, size=n_samples)


# Main execution logic
if __name__ == "__main__":
    data = load_data(FILE_PATH)
    X, y = prepare_data(data)
    mu, std = build_distribution(y)
    print(f"Mu: {mu}, Std: {std}")
    save_distribution((mu, std), MODEL_PATH)

    # Example usage of the function
    example_samples = sample_prediction(10)
    print("Sampled Predictions:", example_samples)
