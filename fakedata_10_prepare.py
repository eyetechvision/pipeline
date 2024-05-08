# First, let's load the data from the uploaded file to see the structure and contents.
import pandas as pd

import numpy as np
from fakecr import sample_prediction
from gpr import predict_ratio, ExactGPModel

# Load the data
df_low = pd.read_csv(r"D:\pipline\pipeline\df_low.csv")

# Display the first few rows of the dataframe
df_low.head()


df_age_10_gender_1 = df_low[(df_low["age"] == 10) & (df_low["gender"] == 1)]
df_age_10_gender_2 = df_low[(df_low["age"] == 10) & (df_low["gender"] == 2)]

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


cr_1 = sample_prediction(n_samples)
cr_2 = sample_prediction(n_samples)

# Print the first 5 elements
print(cr_1[:5])

print(len(samples_gender_1))
print(len(cr_1))

# Use a for loop to predict each item in samples_gender_1 and samples_gender_2
al_cr_1 = [predict_ratio(sample, cr, 1) for sample, cr in zip(samples_gender_1, cr_1)]
al_cr_2 = [predict_ratio(sample, cr, 2) for sample, cr in zip(samples_gender_2, cr_2)]

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
