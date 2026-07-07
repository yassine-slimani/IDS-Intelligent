import pandas as pd


def load_dataset(train_path, test_path):
    """
    Load NSL-KDD dataset
    """

    train_data = pd.read_csv(train_path, header=None)
    test_data = pd.read_csv(test_path, header=None)

    return train_data, test_data



def analyze_dataset(data, name="Dataset"):
    """
    Basic dataset analysis
    """

    print(f"\n===== {name} Information =====")

    print("\nShape:")
    print(data.shape)

    print("\nMissing values:")
    print(data.isnull().sum().sum())

    print("\nDuplicate rows:")
    print(data.duplicated().sum())

    print("\nData types:")
    print(data.dtypes.value_counts())

    print("\nClass distribution:")
    print(data[41].value_counts())



def clean_dataset(data):
    """
    Clean dataset
    """

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Reset index
    data = data.reset_index(drop=True)

    return data