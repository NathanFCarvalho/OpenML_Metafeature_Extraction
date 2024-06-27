import json
import openml
import os
import csv
import pandas as pd
from pymfe.mfe import MFE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def load_feature_names_mapping(filename):
    """Load feature name mapping from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def save_to_csv(dataset_id, feature_names, values, filename, feature_names_mapping_file):
    """Append data to a CSV file."""
    file_exists = os.path.isfile(filename)
    feature_names_mapping = load_feature_names_mapping(feature_names_mapping_file)

    # Substitute feature names with prettier names if mapping exists
    prettier_feature_names = [feature_names_mapping.get(f, f) for f in feature_names]

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['dataset_id'] + prettier_feature_names)
        writer.writerow([dataset_id] + values)


def save_to_jsonl(data, filename):
    """Append data to a JSONL file."""
    with open(filename, 'a') as file:
        json.dump(data, file)
        file.write('\n')


def subsample_dataset(X, y, subsample=False, subsample_size=5000):
    """Reduce the dataset size to subsample_size if it is too large."""
    if len(X) > subsample_size and subsample is True:
        X, _, y, _ = train_test_split(X, y, stratify=y, test_size=1 - subsample_size / len(X), random_state=42)
    return X, y


def handle_nan(X):
    """Impute missing values: mean for numeric columns, mode for categorical columns."""
    X = X.dropna(axis=1, how='all')

    # Impute numerical columns
    num_cols = X.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='mean')
        X[num_cols] = num_imputer.fit_transform(X[num_cols])

    # Impute categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if X[col].isnull().any():
            X[col].fillna(X[col].mode()[0], inplace=True)

    return X


def fetch_and_prepare_data(dataset_id, subsample, target=None):
    """Fetch dataset from OpenML and handle missing values."""
    data = openml.datasets.get_dataset(dataset_id, download_data=False, download_qualities=False,
                                       download_features_meta_data=False, force_refresh_cache=True)
    X, y, _, _ = data.get_data(target=target)
    X = handle_nan(X)
    if target:
        X, y = subsample_dataset(X, y, subsample)
    return X, y


def extract_metafeatures(dataset_id, subsample=False, target=None):
    """Extract metafeatures for a dataset, with or without a target column."""
    try:
        X, y = fetch_and_prepare_data(dataset_id, subsample, target)
        mfe = MFE()
        if y is not None:
            mfe.fit(X.to_numpy(), y.to_numpy())
        else:
            mfe.fit(X.to_numpy())
        ft = mfe.extract()

        # Prepare data for saving to CSV
        feature_names = ft[0]
        values = [float(value) for value in ft[1]]

        # Save to CSV
        save_to_csv(dataset_id, feature_names, values, 'metafeatures.csv', 'feature_names.json')
    except Exception as e:
        save_to_jsonl({dataset_id: str(e)}, 'errors.jsonl')


def process_datasets(datasets_list, datasets_target):
    """Process a list of datasets and extract their metafeatures."""
    for dataset_id, target in zip(datasets_list, datasets_target):
        if pd.isna(target):
            extract_metafeatures(dataset_id)
        else:
            extract_metafeatures(dataset_id, target)

extract_metafeatures(dataset_id=2)