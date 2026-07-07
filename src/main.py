from preprocess import load_dataset, analyze_dataset, clean_dataset


TRAIN_PATH = "../dataset/KDDTrain+.txt"
TEST_PATH = "../dataset/KDDTest+.txt"


# Load data
train, test = load_dataset(TRAIN_PATH, TEST_PATH)


# Analyze dataset before cleaning
analyze_dataset(train, "Training dataset")


# Cleaning
train = clean_dataset(train)


# Analyze result
print("\nAfter cleaning:")
print(train.shape)