import pandas as pd
from sklearn.preprocessing import StandardScaler


# NSL-KDD column names
COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty"
]


def add_column_names(data):
    """
    Add column names to NSL-KDD dataset
    """
    data.columns = COLUMNS
    return data



def split_features_target(data):
    """
    Separate features (X) and target (y)
    """

    X = data.drop(["label", "difficulty"], axis=1)
    y = data["label"]

    return X, y



def encode_features(X):
    """
    Encode categorical features
    """

    categorical_features = [
        "protocol_type",
        "service",
        "flag"
    ]

    X = pd.get_dummies(
        X,
        columns=categorical_features
    )

    return X



def normalize_features(X):
    """
    Normalize numerical features
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    return X_scaled, scaler