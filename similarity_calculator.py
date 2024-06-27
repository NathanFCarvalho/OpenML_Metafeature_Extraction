import random
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def preprocess_dataframe(df):
    """
    Preprocesses the dataframe by replacing infinities with NaN, imputing missing values with the mean,
    and normalizing the data using standard scaling.
    """
    # Replace infinities with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Replace very large or very small values with NaN
    df[df.abs() > 1e9] = np.nan

    # Impute missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Normalize the data using standard scaling
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)

    return df_normalized


def find_top_similar_features(df, dataset_index, x = 5):
    """
    Finds the top 10 features that are most similar to the target dataset before handling missing values.
    """
    target_dataset = df.iloc[dataset_index]
    similarities = df.corrwith(target_dataset, axis=0)
    top_10_features = similarities.abs().sort_values(ascending=False).head(x).index.tolist()
    return top_10_features


def find_nearest_neighbors(metafeature_dataframe, dataset_id, k=20):
    """
    Finds k of the closest datasets to a given dataset based on its metafeatures.
    """
    # Convert the dictionary to a DataFrame
    df = metafeature_dataframe
    dataset_ids = list(df['dataset id'])
    dataset_index = dataset_ids.index(dataset_id)
    df = df.drop('dataset id', axis=1)

    # Select the top x most similar features before handling NaN values
    top_10_features = find_top_similar_features(df, dataset_index)
    df = df[top_10_features]

    # Preprocess the DataFrame
    df_normalized = preprocess_dataframe(df)

    # Initialize, fit and calculate distances through the NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='cosine').fit(df_normalized)
    distances, indices = nbrs.kneighbors([df_normalized.iloc[dataset_index]])
    ids = [dataset_ids[index] for index in indices[0]]

    return distances, ids

