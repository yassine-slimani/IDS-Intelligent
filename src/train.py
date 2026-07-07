from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle


def split_train_test(X, y):
    """
    Split data into training and validation sets
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test



def train_random_forest(X_train, y_train):
    """
    Train Random Forest model
    """

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model



def save_model(model, path="models/random_forest.pkl"):
    """
    Save trained model
    """

    with open(path, "wb") as file:
        pickle.dump(model, file)