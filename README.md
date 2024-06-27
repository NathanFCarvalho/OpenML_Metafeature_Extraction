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

### Recommendation System

The `similarity_calculator.py` script is based on the sklearn 
nearest neighbor algorithm and provides functionality to:
- Preprocess the metafeatures DataFrame by replacing infinite values, 
imputing missing values, and normalizing the data.
- Find the nearest neighbors using the `sklearn` library.

## Examples


