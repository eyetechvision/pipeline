import pandas as pd

# Load the real data from df_high.csv
df_high_path = "df_high.csv"
df_high = pd.read_csv(df_high_path)


# Convert 'diopter_s' and 'diopter_c' to numeric types
df_high["diopter_s"] = pd.to_numeric(df_high["diopter_s"], errors="coerce")
df_high["diopter_c"] = pd.to_numeric(df_high["diopter_c"], errors="coerce")

# Recalculate the spherical equivalent (SE)
df_high["SE"] = df_high["diopter_s"] + df_high["diopter_c"] / 2

# Filter data for age 10
df_high_age_10 = df_high[df_high["age"] == 10]

# Display the first few rows to check
df_high_age_10.head()


# Load the fake data groups
group_1_df_1_path = "group_1_df_1.csv"
group_2_df_1_path = "group_2_df_1.csv"
group_1_df_2_path = "group_1_df_2.csv"
group_2_df_2_path = "group_2_df_2.csv"

group_1_df_1 = pd.read_csv(group_1_df_1_path)
group_2_df_1 = pd.read_csv(group_2_df_1_path)
group_1_df_2 = pd.read_csv(group_1_df_2_path)
group_2_df_2 = pd.read_csv(group_2_df_2_path)

# Display the first few rows to check
group_1_df_1.head(), group_2_df_1.head(), group_1_df_2.head(), group_2_df_2.head()


import matplotlib.pyplot as plt
import seaborn as sns

# Divide real data into two groups based on SE
real_group_1 = df_high_age_10[df_high_age_10["SE"] < -0.5]
real_group_2 = df_high_age_10[df_high_age_10["SE"] >= -0.5]

# Combining the fake data by genders
fake_group_1_total = pd.concat([group_1_df_1, group_1_df_2])
fake_group_2_total = pd.concat([group_2_df_1, group_2_df_2])

# Counts for real and fake data
real_group_1_count = real_group_1.shape[0]
real_group_2_count = real_group_2.shape[0]
fake_group_1_count = fake_group_1_total.shape[0]
fake_group_2_count = fake_group_2_total.shape[0]

# Proportion calculations
real_total = real_group_1_count + real_group_2_count
fake_total = fake_group_1_count + fake_group_2_count

real_group_1_proportion = real_group_1_count / real_total
real_group_2_proportion = real_group_2_count / real_total
fake_group_1_proportion = fake_group_1_count / fake_total
fake_group_2_proportion = fake_group_2_count / fake_total

# Prepare data for plotting
proportions_data = {
    "Group": ["Real Group 1", "Real Group 2", "Fake Group 1", "Fake Group 2"],
    "Proportion": [
        real_group_1_proportion,
        real_group_2_proportion,
        fake_group_1_proportion,
        fake_group_2_proportion,
    ],
}
proportions_df = pd.DataFrame(proportions_data)

# Create a list of counts in the same order as the bars
counts = [
    real_group_1_count,
    real_group_2_count,
    fake_group_1_count,
    fake_group_2_count,
]

# Plotting the proportions
plt.figure(figsize=(10, 6))
barplot = sns.barplot(x="Group", y="Proportion", data=proportions_df)
plt.title("Proportion of Counts between Group 1 and Group 2 for Real and Fake Data")
plt.ylabel("Proportion")
plt.xlabel("Data Group")

# Adding the count number to the top of each bar
for i, count in enumerate(counts):
    barplot.text(i, proportions_df.Proportion[i], count, color="black", ha="center")

plt.show()
#
#
#
#
#
#
# Calculate AL for total fake data
fake_group_1_total["AL"] = fake_group_1_total["ratio"] * fake_group_1_total["CR"]
fake_group_2_total["AL"] = fake_group_2_total["ratio"] * fake_group_2_total["CR"]

# Combining fake groups for plotting against real data
fake_total = pd.concat([fake_group_1_total, fake_group_2_total])

df_high_age_10.loc[:, "Ratio"] = df_high_age_10["AL"] / df_high_age_10["CR"]
print(df_high_age_10.columns)
print(df_high_age_10.head(10))


# Plot AL against SE for total real data
fig1, ax1 = plt.subplots(figsize=(6, 6))
sns.scatterplot(y=df_high_age_10["SE"], x=df_high_age_10["AL"], ax=ax1, color="blue")
ax1.set_title("AL vs. SE for Total Real Data")
ax1.set_xlabel("AL")
ax1.set_ylabel("SE")
plt.show()

# Plot AL against SE for total fake data
fig2, ax2 = plt.subplots(figsize=(6, 6))
sns.scatterplot(y=fake_total["SE"], x=fake_total["AL"], ax=ax2, color="red")
ax2.set_title("AL vs. SE for Total Fake Data")
ax2.set_xlabel("AL")
ax2.set_ylabel("SE")
plt.show()

# Plot Ratio against SE for total real data
fig3, ax3 = plt.subplots(figsize=(6, 6))
sns.scatterplot(
    y=df_high_age_10["SE"], x=df_high_age_10["Ratio"], ax=ax3, color="green"
)
ax3.set_title("Ratio vs. SE for Total Real Data")
ax3.set_xlabel("Ratio")
ax3.set_ylabel("SE")
plt.show()

# Plot Ratio against SE for total fake data
fig4, ax4 = plt.subplots(figsize=(6, 6))
sns.scatterplot(y=fake_total["SE"], x=fake_total["ratio"], ax=ax4, color="green")
ax4.set_title("Ratio vs. SE for Total Fake Data")
ax4.set_xlabel("Ratio")
ax4.set_ylabel("SE")
plt.show()



# Calculate AL for each dataframe
group_1_df_1['AL'] = group_1_df_1['ratio'] * group_1_df_1['CR']
group_2_df_1['AL'] = group_2_df_1['ratio'] * group_2_df_1['CR']
group_1_df_2['AL'] = group_1_df_2['ratio'] * group_1_df_2['CR']
group_2_df_2['AL'] = group_2_df_2['ratio'] * group_2_df_2['CR']

# Combine the male dataframes and the female dataframes
male_total = pd.concat([group_1_df_1, group_2_df_1], ignore_index=True)
female_total = pd.concat([group_1_df_2, group_2_df_2], ignore_index=True)

# Calculate percentiles for each gender
percentiles = [5, 10, 25, 50, 75, 90, 95]
import numpy as np

# Calculate percentiles for each gender using numpy
male_percentiles = np.percentile(male_total['AL'], percentiles)
female_percentiles = np.percentile(female_total['AL'], percentiles)

male_percentiles, female_percentiles

# Plotting the comparison of provided and calculated percentiles for both males and females
plt.figure(figsize=(12, 6))

# Plot for males
provided_percentiles_m = np.array([22.49, 22.77, 23.24, 23.78, 24.34, 24.86, 25.18])
plt.subplot(1, 2, 1)
plt.plot(percentiles, male_percentiles, label='Calculated Male Percentiles', marker='o')
plt.plot(percentiles, provided_percentiles_m , label='Provided Male Percentiles', marker='x')
# Calculate the Root Mean Square Error (RMSE)
rmse = np.sqrt(np.mean((male_percentiles - provided_percentiles_m)**2))

plt.title('Comparison of Male AL Percentiles, RMSE: {:.4f}'.format(rmse))
plt.xlabel('Percentiles')
plt.ylabel('Axial Length (AL)')
plt.legend()

# Plot for females
provided_percentiles_f = np.array([22.12, 22.37, 22.81, 23.33, 23.89, 24.43, 24.78])
plt.subplot(1, 2, 2)
plt.plot(percentiles, female_percentiles, label='Calculated Female Percentiles', marker='o')
plt.plot(percentiles, provided_percentiles_f, label='Provided Female Percentiles', marker='x')


# Calculate the difference between the calculated and provided percentiles
difference = female_percentiles - provided_percentiles_f

# Calculate the Root Mean Square Error (RMSE)
rmse = np.sqrt(np.mean(difference**2))

print(f"Root Mean Square Error: {rmse}")

plt.title('Comparison of Female AL Percentiles, RMSE: {:.4f}'.format(rmse))
plt.xlabel('Percentiles')
plt.ylabel('Axial Length (AL)')
plt.legend()

plt.tight_layout()
plt.show()

