import pandas as pd
import numpy as np
from joblib import load, dump
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import norm

MODEL_PATH = r"cr_distribution_params.pkl"
FILE_PATH = r"df_high.csv"


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

def chatGPT_cr_generator(age, gender, num_samples=1):
    """
    Simulate Corneal Radius (CR) for given age and gender.
    
    :param age: int or array-like, age of the individuals
    :param gender: str, 'male' or 'female'
    :param num_samples: int, number of samples to generate
    :return: array, simulated CR values
    """
    # Base CR values by age, linear interpolation from 3 to 18 years
    base_cr_age_3 = 7.7  # Average CR at age 3
    base_cr_age_18 = 7.8  # Average CR at age 18
    slope = (base_cr_age_18 - base_cr_age_3) / (18 - 3)
    
    # Calculate base CR based on age
    base_cr = base_cr_age_3 + slope * (age - 3)
    
    # Adjust base CR based on gender
    gender_adjustment = -0.02 if gender == 'male' else 0
    base_cr += gender_adjustment
    
    # Simulate CR with normal distribution around the base CR
    std_dev = 0.2  # Standard deviation of CR
    simulated_cr = np.random.normal(base_cr, std_dev, num_samples)
    
    return simulated_cr

# # Example usage
# ages = np.linspace(3, 18, 16)  # Ages from 3 to 18
# genders = ['male', 'female'] * 8  # Alternate genders for simplicity

# # Generate samples
# samples = [simulate_CR(age, gender) for age, gender in zip(ages, genders)]
# print(samples)



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
