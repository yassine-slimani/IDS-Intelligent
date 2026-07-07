from src.data_cleaning import load_dataset, analyze_dataset, clean_dataset
from src.data_preprocessing import (
    add_column_names,
    split_features_target,
    encode_features,
    normalize_features
)


TRAIN_PATH = "dataset/KDDTrain+.txt"
TEST_PATH = "dataset/KDDTest+.txt"


# ==========================
# 1. Load dataset
# ==========================
train, test = load_dataset(TRAIN_PATH, TEST_PATH)


# ==========================
# 2. Analyze dataset
# ==========================
analyze_dataset(train, "Training dataset")


# ==========================
# 3. Cleaning
# ==========================
train = clean_dataset(train)

print("\nAfter cleaning:")
print(train.shape)


# ==========================
# 4. Add column names
# ==========================
train = add_column_names(train)


# ==========================
# 5. Split features and target
# ==========================
X, y = split_features_target(train)

print("\nBefore encoding:")
print(X.shape)


# ==========================
# 6. Encoding
# ==========================
X = encode_features(X)

print("\nAfter encoding:")
print(X.shape)


# ==========================
# 7. Normalization
# ==========================
X_scaled, scaler = normalize_features(X)

print("\nAfter normalization:")
print(X_scaled.shape)


# Target information
print("\nTarget classes:")
print(y.value_counts())