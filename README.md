Item recommender

Processes your dataset of assessment responses and recommends screener items for each output variable. 

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Options](#options)
- [License](#license)

## Installation
To install item-recommender, follow these steps:

1. Clone the repository to your local machine:
`git clone https://github.com/charlie42/item_recommender.git`

2. Navigate to the project directory:
`cd item_recommender`

3. Install item-recommender as a global package:
`pip install .`

## Usage
To use item-recommender, run the following command:
`item-recommender --data-path [DATA_PATH] --config-path [CONFIG_PATH] --mode [MODE]`

Or, from the project directory:
`make run DATA=[DATA_PATH]`

Replace [DATA_PATH] with the path to your dataset in csv format, and [CONFIG_PATH] with the path to the config file. The example of the config file format is in the example_config subdirectory.

Dataset format:
- Output column names should start with the output column prefix from the config ("Diag." by default), and contain 0 or 1. Rows with a missing value in an output column are skipped for that output.
- All other columns are used as items, and should be numeric. Missing item values are filled with the item's median, computed on the training data of each fold. Items with a larger fraction of missing values than `max missing fraction` are dropped.

The name of the dataset file will be used as the name of the directory where the output will be stored. It contains:
- `recommended_items.csv`: the recommended items for each output, each item's stability (the fraction of cross-validation folds that selected it) and coefficient on standardized responses, performance on the test set, the recommended probability threshold, and test sensitivity and specificity at that threshold.
- `results.joblib`: the recommended items, cross-validation and test set performance for every number of items, and the final model trained on all data with the recommended items.

## Options
- `--mode`: DEV (fast, for testing), DEBUG (default), or PROD. Each mode overrides the config settings listed under `modes` in the config file.

Config settings:
- `n features to check`: the maximum number of items to recommend.
- `percentage for opt n features`: the recommended number of items is the smallest one that reaches this percentage of the best performance improvement over chance.
- `target sensitivity`: the recommended probability threshold is the median across outer folds of the highest threshold that reaches this sensitivity. Sensitivity and specificity on the test set are reported at that threshold.
- `split percentage`: fraction of the data used as the test set.
- `outer cv n splits`, `inner cv n splits`: number of cross-validation folds.
- `hp search n iter`: number of hyperparameter combinations to try.
- `output column prefix`: prefix of the output column names.
- `min class size`: outputs with fewer positive (or negative) samples than this are skipped.
- `max missing fraction`: items with a larger fraction of missing values are dropped.
- `exclude columns`: columns that are neither items nor outputs, such as IDs or total scores. They are ignored.

## License
This project is licensed under the MIT License.
