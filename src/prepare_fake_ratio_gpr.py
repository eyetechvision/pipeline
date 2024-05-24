import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle

# Set the seed for reproducibility
torch.manual_seed(0)

# Define the sizes
input_size = 3
hidden_size = 32
output_size = 1
num_samples = 100

# Generate random tensors for input and output
inputs = torch.randn(num_samples, input_size)
outputs = torch.randn(num_samples, output_size)

# Convert tensors to numpy arrays for sklearn
inputs_np = inputs.numpy()
outputs_np = outputs.numpy()

# Split the data into training and testing sets
inputs_train_np, inputs_test_np, outputs_train_np, outputs_test_np = train_test_split(
    inputs_np, outputs_np, test_size=0.2, random_state=0
)

# Apply MinMax scaling to the data
scaler = MinMaxScaler()
inputs_train_np = scaler.fit_transform(inputs_train_np)
inputs_test_np = scaler.transform(inputs_test_np)

# Convert numpy arrays back to tensors
inputs_train = torch.from_numpy(inputs_train_np)
inputs_test = torch.from_numpy(inputs_test_np)
outputs_train = torch.from_numpy(outputs_train_np)
outputs_test = torch.from_numpy(outputs_test_np)

# Create TensorDatasets
train_dataset = TensorDataset(inputs_train, outputs_train)
test_dataset = TensorDataset(inputs_test, outputs_test)

# Save the datasets
torch.save(train_dataset, "data/prepared/fake_ratio_gpr_train_data.pt")
torch.save(test_dataset, "data/prepared/fake_ratio_gpr_test_data.pt")

# Save the scaler
with open("data/prepared/fake_ratio_gpr_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
