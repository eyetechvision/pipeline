# First, let's load the data from the uploaded file to see the structure and contents.
import pandas as pd

import numpy as np
from fakecr import sample_prediction
from fakecr import chatGPT_cr_generator
from gpr import predict_ratio, ExactGPModel

# Load the data
df_low = pd.read_csv(r"df_low.csv")
adjust_param_m = 0
# adjust_param_m = 0.12
# adjust_param_f = 0.30
adjust_param_f = 0.4

# Display the first few rows of the dataframe
df_low.head()


df_age_10_gender_1 = df_low[(df_low["age"] == 10) & (df_low["gender"] == 1)]
df_age_10_gender_2 = df_low[(df_low["age"] == 10) & (df_low["gender"] == 2)]

# Count males and females in this filtered dataframe

print(f"Number of males aged 10: {df_age_10_gender_1.count()}")
print(f"Number of females aged 10: {df_age_10_gender_2.count()}")


print(df_age_10_gender_1.head(), df_age_10_gender_2.head())

df_age_10_gender_1_SE = pd.DataFrame()
df_age_10_gender_2_SE = pd.DataFrame()

df_age_10_gender_1_SE["SE"] = pd.concat(
    [
        df_age_10_gender_1["right_diopter_s"]
        + df_age_10_gender_1["right_diopter_c"] / 2,
        df_age_10_gender_1["left_diopter_s"] + df_age_10_gender_1["left_diopter_c"] / 2,
    ]
)

df_age_10_gender_2_SE["SE"] = pd.concat(
    [
        df_age_10_gender_2["right_diopter_s"]
        + df_age_10_gender_2["right_diopter_c"] / 2,
        df_age_10_gender_2["left_diopter_s"] + df_age_10_gender_2["left_diopter_c"] / 2,
    ]
)

# Checking the filtered data
(df_age_10_gender_1_SE.head(), df_age_10_gender_2_SE.head())


# Generate distributions for both genders
n_samples = 2000

# Draw samples from SE distributions for both genders
samples_gender_1 = (
    df_age_10_gender_1_SE["SE"].sample(n_samples, replace=True).to_numpy()
)
samples_gender_2 = (
    df_age_10_gender_2_SE["SE"].sample(n_samples, replace=True).to_numpy()
)


# cr_1 = sample_prediction(n_samples)
# cr_2 = sample_prediction(n_samples)
cr_1 = chatGPT_cr_generator(10, 'male', n_samples)
cr_2 = chatGPT_cr_generator(10, 'female', n_samples)

# Print the first 5 elements
print(cr_1[:5])

print(len(samples_gender_1))
print(len(cr_1))

# Use a for loop to predict each item in samples_gender_1 and samples_gender_2
# al_cr_1 = [predict_ratio(sample * (1 - adjust_param_m), cr, 1) for sample, cr in zip(samples_gender_1, cr_1)]
# al_cr_2 = [predict_ratio(sample * (1 - adjust_param_f), cr, 2) for sample, cr in zip(samples_gender_2, cr_2)]
al_cr_1 = [predict_ratio(sample + adjust_param_m, cr, 1) for sample, cr in zip(samples_gender_1, cr_1)]
al_cr_2 = [predict_ratio(sample +  adjust_param_f, cr, 2) for sample, cr in zip(samples_gender_2, cr_2)]

# Create final dataset with 2000 points for each gender, including the SE, AL/CR, and assigned CR
final_data_1 = np.column_stack(
    (np.full(n_samples, 10), np.full(n_samples, 1), al_cr_1, cr_1, samples_gender_1)
)
final_data_2 = np.column_stack(
    (np.full(n_samples, 10), np.full(n_samples, 2), al_cr_2, cr_2, samples_gender_2)
)

df_1 = pd.DataFrame(final_data_1, columns=["age", "gender", "ratio", "CR", "SE"])
df_2 = pd.DataFrame(final_data_2, columns=["age", "gender", "ratio", "CR", "SE"])

group_1_df_1 = df_1.loc[df_1["SE"] < -0.5]
group_2_df_1 = df_1.loc[df_1["SE"] >= -0.5]

group_1_df_2 = df_2.loc[df_2["SE"] < -0.5]
group_2_df_2 = df_2.loc[df_2["SE"] >= -0.5]

group_1_df_1.to_csv("group_1_df_1.csv", index=False)
group_2_df_1.to_csv("group_2_df_1.csv", index=False)

group_1_df_2.to_csv("group_1_df_2.csv", index=False)
group_2_df_2.to_csv("group_2_df_2.csv", index=False)


import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Prepare the data
X = df_2[['ratio', 'CR']].values
y = df_2['SE'].values

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=10)

# Define the model
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# Define the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
for epoch in range(50):
    for X_batch, y_batch in dataloader:
        # Compute predictions
        y_pred = model(X_batch).squeeze()
        # Compute the loss
        loss = loss_fn(y_pred, y_batch)
        # Zero the gradients
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # Update the weights
        optimizer.step()

# Assume we have new_data as a DataFrame containing the new samples
new_data = pd.DataFrame({'ratio': [3.091271, 2.963219], 'CR': [8.042, 8.2275]})

# Convert the new data to a PyTorch tensor
new_data_tensor = torch.tensor(new_data.values, dtype=torch.float32)

# Use the model to make predictions
with torch.no_grad():
    predictions = model(new_data_tensor)

# Convert the predictions to a numpy array
predictions = predictions.numpy()

# Print the predictions
print("曾紫蕊: ", predictions)

# Assume we have new_data as a DataFrame containing the new samples
new_data = pd.DataFrame({'ratio': [2.96,3.057, 3.140537092,2.314690208], 'CR': [7.820840102, 7.691429876,7.654741624,10.04022047]})

# Convert the new data to a PyTorch tensor
new_data_tensor = torch.tensor(new_data.values, dtype=torch.float32)

# Use the model to make predictions
with torch.no_grad():
    predictions = model(new_data_tensor)

# Convert the predictions to a numpy array
predictions = predictions.numpy()

# Print the predictions
print("文曦月/ 袁常曦: ", predictions)


# Prepare the data
X = df_1[['ratio', 'CR']].values
y = df_1['SE'].values

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=10)

# Define the model
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# Define the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
for epoch in range(50):
    for X_batch, y_batch in dataloader:
        # Compute predictions
        y_pred = model(X_batch).squeeze()
        # Compute the loss
        loss = loss_fn(y_pred, y_batch)
        # Zero the gradients
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # Update the weights
        optimizer.step()

# Assume we have new_data as a DataFrame containing the new samples
new_data = pd.DataFrame({'ratio': [3.091271, 2.963219], 'CR': [8.042, 8.2275]})

# Convert the new data to a PyTorch tensor
new_data_tensor = torch.tensor(new_data.values, dtype=torch.float32)

# Use the model to make predictions
with torch.no_grad():
    predictions = model(new_data_tensor)

# Convert the predictions to a numpy array
predictions = predictions.numpy()

# Print the predictions
print("曾紫蕊: ", predictions)

# Assume we have new_data as a DataFrame containing the new samples
new_data = pd.DataFrame({'ratio': [3.125208902,3.16085], 'CR': [7.513097759,7.463182372]})

# Convert the new data to a PyTorch tensor
new_data_tensor = torch.tensor(new_data.values, dtype=torch.float32)

# Use the model to make predictions
with torch.no_grad():
    predictions = model(new_data_tensor)

# Convert the predictions to a numpy array
predictions = predictions.numpy()

# Print the predictions
print("汪宸逸(-1.5, -1.5, 0, -0.25): ", predictions)

 