import random
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

info_theoretic = ["attribute concentration mean", "attribute concentration standard deviation", "attribute entropy mean",
                  "attribute entropy standard deviation", "class concentration mean", "class concentration standard deviation", "class entropy"
                  "equivalent number of attributes", "joint entropy mean", "joint entropy standard deviation", "mutual information mean", 
                  "mutual information standard deviation", "noisiness ratio"]
general = ["attribute to instance ratio", "categorical to numerical ratio", "frequency of classes mean", "frequency of classes standard deviation",
          "instance to attribute ratio", "number of attributes", "number of binary attributes", "number of categorical attributes", "number of classes",
          "number of instances", "number of numerical attributes", "numerical to categorical ratio"]
landmarking = ["best node mean performance", "best node performance standard deviation", "elite nearest neighbor mean performance", 
               "elite nearest neighbor performance standard deviation", "linear discriminant mean performance", "linear discriminant performance standard deviation",
              "naive bayes mean performance", "naive bayes performance standard deviation", "one nearest neighbor mean performance",
              "one nearest neighbor performance standard deviation", "random node mean performance", "random node performance standard deviation",
              "worst node mean performance", "worst node performance standard deviation"]
statistical = ["canonical correlation mean",  "canonical correlation standard deviation", "correlation mean", "correlation standard deviation",
              "covariance mean", "covariance standard deviation", "eigenvalues mean", "eigenvalues standard deviation", "geometric mean of attributes",
              "geometric mean standard deviation", "class gravity", "harmonic mean of attributes", "harmonic mean standard deviation", "interquartile range mean",
              "interquartile range standard deviation", "kurtosis mean", "kurtosis standard deviation", "Lawley-Hotelling trace", 
              "median absolute deviation mean", "median absolute deviation standard deviation", "maximum value mean", "maximum value standard deviation",
              "mean value mean", "mean value standard deviation", "median value mean", "median value standard deviation", "minimum value mean",
              "minimum value standard deviation", "number of correlated attributes", "number of discrete attributes", 
              "number of normally distributed attributes", "number of outliers", "Pillai's trace","range mean", "range standard deviation", 
              "Roy's largest root", "standard deviation mean", "standard deviation standard deviation", "standard deviation ratio",
              "skewness mean", "skewness standard deviation", "sparsity mean", "sparsity standard deviation", "trimmed mean", "trimmed mean standard deviation",
              "variance mean", "variance standard deviation", "Wilks' Lambda", ]
model_based = ["number of leaves in decision tree", "leaves per branch mean",  "leaves per branch standard deviation", "leaves corroboration mean",
              "leaves corroboration standard deviation", "leaves homogeneity mean", "leaves homogeneity standard deviation", "leaves per class mean",
              "leaves per class standard deviation", "number of nodes in decision tree", "nodes per attribute ratio", "nodes per instance ratio",
              "nodes per level mean", "nodes per level standard deviation", "repeated nodes mean", "repeated nodes standard deviation",
              "tree depth mean", "tree depth standard deviation", "tree imbalance mean", "tree imbalance standard deviation", "tree shape mean",
              "tree shape standard deviation", "variable importance mean", "variable importance standard deviation", ]

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

def find_nearest_neighbors(metafeature_dataframe, dataset_id, k=20):
    """
    Finds k of the closest datasets to a given dataset based on its metafeatures.
    """
    # Convert the dictionary to a DataFrame
    df = metafeature_dataframe
    dataset_ids = list(df['dataset id'])
    dataset_index = dataset_ids.index(dataset_id)
    df = df.drop('dataset id', axis=1)

    # Select the desired features before handling NaN values
    df = df[general]

    # Preprocess the DataFrame
    df_normalized = preprocess_dataframe(df)

    # Initialize, fit and calculate distances through the NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='cosine').fit(df_normalized)
    distances, indices = nbrs.kneighbors([df_normalized.iloc[dataset_index]])
    ids = [dataset_ids[index] for index in indices[0]]

    return distances, ids

