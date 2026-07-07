import pandas as pd

# Load the datasets
train = pd.read_csv("dataset/KDDTrain+.txt", header=None)
test = pd.read_csv("dataset/KDDTest+.txt", header=None)

# Display dataset dimensions
print("Train:", train.shape)
print("Test:", test.shape)

# Display the first five rows
print(train.head())