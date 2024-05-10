import numpy as np

def simulate_CR(age, gender, num_samples=1):
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

# Example usage
ages = np.linspace(3, 18, 16)  # Ages from 3 to 18
genders = ['male', 'female'] * 8  # Alternate genders for simplicity

# Generate samples
samples = [simulate_CR(age, gender) for age, gender in zip(ages, genders)]
print(samples)
