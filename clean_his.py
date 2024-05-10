import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('df_combine（清洗HIS）.csv')

# Split the DataFrame into two parts
df_left = df.iloc[:, :9]
df_right = df.iloc[:, 9:]

# Rename the columns
df_left.columns = ['name', 'gender', 'age', 'K1(L)', 'K2(L)', 'AL(L)', 'K1(R)', 'K2(R)', 'AL(R)']
df_right.columns = ['name', 'gender', 'age', 'SR(L)', 'SC(L)', 'SR(R)', 'SC(R)', 'axis_L', 'axis_R']

# Drop the unnecessary columns
df_right = df_right.drop(columns=['axis_L', 'axis_R'])

df_left = df_left.dropna(subset=['name', 'gender', 'age'])
df_right = df_right.dropna(subset=['name', 'gender', 'age'])
df_left.to_csv('df_left.csv', index=False)
df_right.to_csv('df_right.csv', index=False)

# Convert 'nan' strings to actual NaN values
df_left['gender'] = df_left['gender'].replace('nan', np.nan)
df_right['gender'] = df_right['gender'].replace('nan', np.nan)

# Use combine_first to fill NaN values in df_left['gender'] with values from df_right['gender']
df_left['gender'] = df_left['gender'].combine_first(df_right['gender'])

# Convert the 'gender' column to the same data type in both DataFrames
df_left['gender'] = df_left['gender'].astype('str')
df_right['gender'] = df_right['gender'].astype('str')

# Now merge the two DataFrames
df_clean = pd.merge(df_left, df_right, on=['name', 'gender', 'age'], how='outer')

# Save the cleaned DataFrame to a new CSV file
df_clean.to_csv('cleaned_data.csv', index=False)

# Filter the DataFrame
df_age_10 = df_clean[df_clean['age'] == 10.0]

df_age_10.to_csv('df_age_10.csv', index=False)