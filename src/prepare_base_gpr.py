# gpr model on df_high
# to predict diopter_s from AL/CR
# as a baseline

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import torch
from torch.utils.data import TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler


def main():

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py input_path output_path\n")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path_train = sys.argv[2] + "/base_gpr_train_data.pt"
    output_path_test = sys.argv[2] + "/base_gpr_test_data.pt"

    data = pd.read_csv(input_path, delimiter=",")

    ###### Data Preprocessing ######
    X = data[["age", "AL", "CR", "gender"]]
    y = pd.to_numeric(data["diopter_s"], errors="coerce")
    X = X.apply(pd.to_numeric, errors="coerce")
    # X = X.dropna()
    # y = y.loc[X.index]
    # Replace inf values with nan
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    y.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with nan values in X or y
    valid_data = X.join(y).dropna()
    X = valid_data[X.columns]
    y = valid_data[y.name]

    X_tensor = torch.tensor(X.values).float()
    y_tensor = torch.tensor(y.values).float()
    dataset = TensorDataset(X_tensor, y_tensor)

    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    # Determine the lengths of your training and testing sets
    train_data, test_data = random_split(dataset, [train_len, test_len])

    # Save the datasets to the output path
    torch.save(train_data, output_path_train)
    torch.save(test_data, output_path_test)

    return "Data prepared and saved!"


if __name__ == "__main__":
    main()
