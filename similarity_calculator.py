import random
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def select_random_keys(mf, num_keys=50):
    return random.sample(list(mf.keys()), num_keys)


def create_dataframe_from_dict(subsample_dict):
    data = []
    for key, (features, values) in subsample_dict.items():
        row = {feature: value for feature, value in zip(features, values)}
        data.append(row)
    return pd.DataFrame(data)


def preprocess_dataframe(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[df.abs() > 1e9] = np.nan

    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)

    return df_normalized


def find_nearest_neighbors(df_normalized, k=20):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean').fit(df_normalized)
    distances, indices = nbrs.kneighbors([df_normalized.iloc[0]])
    return distances, indices


def main(mf, num_keys=50, k=20):
    random_keys = select_random_keys(mf, num_keys)
    subsample_dict = {key: mf[key] for key in random_keys}

    df = create_dataframe_from_dict(subsample_dict)
    df_normalized = preprocess_dataframe(df)

    distances, indices = find_nearest_neighbors(df_normalized, k)
    return distances, indices

