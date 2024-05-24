import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split

import argparse
from pathlib import Path
import sys
from sklearn.preprocessing import MinMaxScaler
import joblib


def main(input_path, output_path):

    output_path_train = output_path / "ratio_gpr_train_data.pt"
    output_path_test = output_path / "ratio_gpr_test_data.pt"
    output_path_scaler = output_path / "ratio_gpr_scaler.pkl"
    # Load the data
    try:
        data = pd.read_csv(input_path, delimiter=",")
    except FileNotFoundError:
        sys.exit(f"File not found: {input_path}")

    # Ensure the data is numeric
    X = data[["age", "diopter_s", "gender"]].apply(pd.to_numeric, errors="coerce")
    AL_CR = data[["AL", "CR"]].apply(pd.to_numeric, errors="coerce")

    # Apply MinMaxScaler
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    # Save the fitted scaler
    joblib.dump(scaler, output_path_scaler)

    # Calculate the ratio, replacing zero and NaN values with NaN to avoid division by zero
    y = AL_CR["AL"].div(AL_CR["CR"].replace(0, np.nan))

    # Drop rows with any NaN values in X or y
    X.dropna(inplace=True)
    y = y.loc[X.index].dropna()
    X = X.loc[y.index]

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X.values).float()
    y_tensor = torch.tensor(y.values).float()

    # Create a TensorDataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Split the dataset into training and testing sets
    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    train_data, test_data = random_split(dataset, [train_len, test_len])

    # Save the datasets to the output path
    print("Train data:")
    print(train_data[:5])  # Print the first 5 samples of the train data

    print("\nTest data:")
    print(test_data[:5])  # Print the first 5 samples of the test data

    try:
        torch.save(train_data, output_path_train)
        torch.save(test_data, output_path_test)
    except Exception as e:
        sys.exit(f"Error saving data: {e}")

    print("Data prepared and saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for GPR model.")
    parser.add_argument("input_path", type=Path, help="Input file path")
    parser.add_argument("output_path", type=Path, help="Output directory path")
    args = parser.parse_args()

    main(args.input_path, args.output_path)
