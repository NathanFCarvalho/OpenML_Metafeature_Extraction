# Metafeature Extraction

This repository contains scripts for extracting metafeatures from 
datasets using the `pymfe` library and handling missing values using
`scikit-learn`. The code also includes functionality for saving the
extracted metafeatures and errors to JSONL files. 

This repository is designed to facilitate **large scale** metadata fetching 
for datasets from OpenML.


## Overview

### Metafeature Extraction

The `metafeature_extraction.py` script is designed to:
- Fetch datasets from OpenML.
- Handle missing values by imputing numerical columns with their mean and categorical columns with their mode.
- Extract metafeatures using the `pymfe` library.
- Save the extracted metafeatures to a CSV file and any errors encountered to a JSONL file.
- For big datasets, the `subsample` parameter, if set to True, creates a subsample of dataset that preserves
stratification and randomly selects a specified number of rows.

### Recommendation System

The `similarity_calculator.py` script is based on the sklearn 
nearest neighbor algorithm using the euclidean metric and calculates similar datasets based on general metafeatures. The following steps are taken:
- Preprocess the metafeatures DataFrame by replacing infinite values, 
imputing missing values, and normalizing the data.
- Find the nearest neighbors using the `sklearn` library.

## Examples

### Metafeature Extraction

In order to extract metafeatures for a dataset in openml, 
you most use the `extract_metafeatures()` function, as it is shown:
```python
extract_metafeatures(dataset_id=2)
```
If the extraction is succesful, a csv file called `metafeatures.csv`
is created or updated. If an error occurs, a file called `errors.jsonl` containing the occured error and in which dataset is created or updated.


### Recommendation System
In order to see the ``k`` most similar datasets to a given one, the ``find_nearest_neighbors()`` function must be used. 
It takes as obligatory parameters a dataframe, such as the one generated by the metafeature extraction tool and the dataset id.
```python
df = pd.read_csv('files/openml_metafeatures.csv')
find_nearest_neighbors(metafeature_dataframe=df, dataset_id=26)
```
In this example, the dataset nursery was used 
and the meta-dataset dataframe was called df. This function returns two lists,
one list with the k closest datasets and one list with the
corresponding distances:
```python
(array([[0.00000000e+00, 0.00000000e+00, 4.17110790e-13, 1.89441041e-09,
         4.45518136e-03, 7.42795367e-03, 7.42795367e-03, 7.42795367e-03,
         7.42795367e-03, 7.50481640e-03, 7.50481640e-03, 7.57172779e-03,
         9.05445198e-03, 9.06429615e-03, 9.06435283e-03, 9.06435283e-03,
         9.06435283e-03, 9.06435283e-03, 9.06435283e-03, 9.06435283e-03]]),
 [26, 43923, 959, 1568, 44990, 44086, 44071, 43972, 44074, 44117, 44126,
  45558, 1459, 1414, 44073, 44116, 43971, 44125, 44085, 44070])
  ```

In this example, the first 4 datasets (distances smaller than 1e-8) are, in fact, variations present in OpenML for the
dataset nursery. This is an utilisation of the similarity function for finding variations of the same dataset inside OpenML.

## Creating the `openml_metafeatures.csv` file

In order to create this file, datasets from OpenML were divised into small ones (< 50,000 instances) 
and big ones (>= 50,000). This division was done mainly for computational power reasons.

For datasets which had a target column, the `extract_metafeatures()` runs PyMFE normally.
If a dataset does not have a target column, it runs PyMFE without specifying a target column, which results in 
fewer metafeatures.

There are some datasets on OpenML which have multiple target columns. For them, 
I chose to run as if they had no target column.

The csv file contains only small datasets, with and without target columns. Bigger datasets were not
included.
