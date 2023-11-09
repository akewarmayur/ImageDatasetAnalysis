import pandas as pd
import matplotlib.pyplot as plt

# Create a sample DataFrame
# data = {
#     'Column1': ['A', 'B', 'A', 'C', 'B', 'A', 'D'],
#     'Column2': ['X', 'Y', 'Z', 'X', 'Y', 'X', 'Z']
# }

# df = pd.DataFrame(data)
df = pd.read_csv("Results/DataAnalysis.csv")
# column1 = df["age"].tolist()
# print(column1)
# cc1 = []
# for i in column1:
#     d = i.split(",")
#     for a in d:
#         cc1.append(a)
# cc1 = [i if i != "others" else "unidentified" for i in cc1]
# column2 = df["gender"]
#
# cc2 = []
# for i in column2:
#     d = i.split(",")
#     for a in d:
#         cc2.append(a)
# cc2 = [i if i != "others" else "unidentified" for i in cc2]
# column3 = df["race"]
#
# cc3 = []
# for i in column1:
#     d = i.split(",")
#     for a in d:
#         cc3.append(a)
#
# cc3 = [i if i != "others" else "unidentified" for i in cc3]
# data = {
#     'Age': cc1,
#     'Gender': cc2,
#     'Race': cc3,
# }
#
# df = pd.DataFrame(data)
print(df.columns)
# Specify the names of columns to plot
df.rename(columns={"settings1":"setting", "settings2":"Time or Lighting", "settings3":"camera angle"}, inplace=True)
columns_to_plot = ['people count', "quality"]

# Create a subplot for each specified column
fig, axes = plt.subplots(nrows=1, ncols=len(columns_to_plot), figsize=(12, 4))

# Plot the frequency of unique values for each specified column
for i, col in enumerate(columns_to_plot):
    value_counts = df[col].value_counts()
    ax = value_counts.plot(kind='bar', ax=axes[i])
    ax.set_title(f'{col}')
    ax.set_xlabel('Unique Values')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()
