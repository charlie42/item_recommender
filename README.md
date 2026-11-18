# Item recommender

[![tests](https://github.com/charlie42/item_recommender/actions/workflows/tests.yml/badge.svg)](https://github.com/charlie42/item_recommender/actions/workflows/tests.yml)

Item recommender takes a dataset of questionnaire responses and binary outcomes, such as diagnoses, and recommends a short screener for each outcome. It reports the screener's items, its AUC on held-out data, the probability threshold that reaches the target sensitivity (80% by default) in cross-validation, and the sensitivity and specificity at that threshold on held-out data.

## Method

The numbers below can be changed in the config file. The setting for each one is given in brackets (see [Options](#options)).

For each outcome, 20% of the data (`split percentage`) is held out as the test set. On the rest:

1. Elastic net logistic regression is tuned on all items with a random hyperparameter search and inner cross-validation.
2. Recursive feature elimination narrows the items down to N (`n features to check`, 30 by default), then sequential forward floating selection builds subsets of 1 to N items.
3. Steps 1 and 2 are run in each of 5 outer cross-validation folds (`outer cv n splits`). Each subset is evaluated on the fold left out. The recommended number of items is the smallest one whose mean AUC reaches 95% (`percentage for opt n features`) of the best improvement over chance (AUC 0.5). The probability threshold is the median across folds of the highest threshold that reaches the target sensitivity (`target sensitivity`, 80% by default).
4. Steps 1 and 2 are run on the whole training set, then the subset of the recommended size is evaluated on the test set (AUC, and sensitivity and specificity at the threshold). A model with all items is also evaluated on the test set for comparison. The final model is trained on all data with those items.

The stability of each recommended item (the fraction of outer folds whose subset of the same size includes it) is reported.

Items with more than 40% missing values (`max missing fraction`) are dropped. Other missing item values are filled with the item's median from the training data of each fold.

Only binary outcomes are supported. Selecting a shared item set for multiple outcomes is not supported.

## Examples

### NHIS child (2019 and 2022)

- Participants: 13,090 children aged 4 to 17, reported by a parent
- Items (68, 50 with at most 40% missing values):
  - Strengths and Difficulties Questionnaire (SDQ) with its impact items
  - how often the child seems sad or anxious
  - difficulties with seeing, hearing, mobility, self care, communication, learning, memory, behavior
  - general health, asthma, diabetes
  - adverse childhood experiences
  - food insecurity
  - school days missed
- Outcomes (self-reported diagnoses): ADHD, autism, learning disability, developmental delay, intellectual disability

![Test AUC by number of items for each NHIS child outcome](results/nhis_child/auc_by_n_items.png)

In each figure, the blue line is the cross-validated AUC for each number of items, the dashed line is the AUC with all items, the vertical line is the recommended number of items, and the orange dot is the test set AUC of the recommended items. In each table, cases are the participants with the diagnosis, and each item's stability is in parentheses.

| Outcome | Cases | N items | Items (stability) | Test AUC | Test AUC, all items (50) | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| Intellectual disability | 236 (1.8%) | 2 | overall difficulties (80%), difficulty learning (100%) | 0.91 | 0.91 | 0.72 | 0.94 |
| Autism | 450 (3.4%) | 3 | overall difficulties (100%), difficulty learning (100%), difficulty accepting changes in routine (100%) | 0.90 | 0.93 | 0.81 | 0.84 |
| Learning disability | 1,022 (7.8%) | 3 | easily distracted (100%), overall difficulties (60%), difficulty learning (100%) | 0.89 | 0.91 | 0.86 | 0.76 |
| ADHD | 1,425 (10.9%) | 3 | easily distracted (100%), good attention span (100%), overall difficulties (100%) | 0.86 | 0.88 | 0.76 | 0.83 |
| Developmental delay | 651 (5.0%) | 4 | overall difficulties (100%), difficulty being understood outside the household (100%), difficulty learning (100%), school days missed (60%) | 0.86 | 0.87 | 0.84 | 0.68 |

Item wording: [NHIS 2022 child codebook](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2022/child-codebook.pdf). Variable names are in `results/nhis_child/recommended_items.csv`.

### NHIS adult (2019 and 2022)

- Participants: 59,648 adults
- Items (56, 49 with at most 40% missing values):
  - PHQ-8, GAD-7
  - frequency and level of depression and anxiety
  - general health, BMI category
  - 13 chronic conditions
  - difficulties with seeing, hearing, walking, communication, memory, self care, daily activities
  - smoking, vaping
  - food insecurity
- Outcomes (self-reported diagnoses): depression, anxiety

![Test AUC by number of items for each NHIS adult outcome](results/nhis_adult/auc_by_n_items.png)

| Outcome | Cases | N items | Items (stability) | Test AUC | Test AUC, all items (49) | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| Depression | 10,618 (17.8%) | 3 | how often depressed (100%), how often anxious (100%), difficulty remembering or concentrating (100%) | 0.87 | 0.89 | 0.76 | 0.81 |
| Anxiety | 9,579 (16.1%) | 4 | general health (60%), GAD-7 feeling nervous or on edge (100%), how often depressed (100%), how often anxious (100%) | 0.86 | 0.87 | 0.82 | 0.75 |

Item wording: [NHIS 2022 adult codebook](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2022/adult-codebook.pdf). Variable names are in `results/nhis_adult/recommended_items.csv`.

#### Without mental health items

The same analysis without the mental health items, to find which other health items predict depression and anxiety.

![Test AUC by number of items for each NHIS adult outcome without mental health items](results/nhis_adult_health/auc_by_n_items.png)

| Outcome | Cases | N items | Items (stability) | Test AUC | Test AUC, all items (27) | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| Depression | 10,618 (17.8%) | 6 | general health (100%), asthma (100%), arthritis (100%), difficulty seeing (100%), ever vaped (100%), worried food would run out (100%) | 0.72 | 0.73 | 0.76 | 0.54 |
| Anxiety | 9,579 (16.1%) | 6 | general health (100%), asthma (100%), arthritis (100%), difficulty seeing (100%), ever vaped (100%), couldn't afford balanced meals (80%) | 0.70 | 0.71 | 0.74 | 0.53 |

Item wording: [NHIS 2022 adult codebook](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2022/adult-codebook.pdf). Variable names are in `results/nhis_adult_health/recommended_items.csv`.

### NHANES (2017 to March 2020)

- Participants: 9,232 adults
- Items (154, 93 with at most 40% missing values): depression, sleep, general health, weight history, physical activity, alcohol, smoking, cardiovascular and diet questionnaires, age, sex
- Outcomes: 14 self-reported diagnoses

![Test AUC by number of items for each NHANES outcome](results/nhanes/auc_by_n_items.png)

| Outcome | Cases | N items | Items (stability) | Test AUC | Test AUC, all items (93) | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| Heart failure | 361 (3.9%) | 4 | age (100%), general health (100%), greatest weight (80%), chest pain (100%) | 0.91 | 0.91 | 0.86 | 0.79 |
| Angina | 240 (2.6%) | 3 | age (100%), general health (100%), chest pain (100%) | 0.90 | 0.90 | 0.92 | 0.75 |
| Heart attack | 432 (4.7%) | 4 | age (100%), sex (100%), general health (100%), chest pain (100%) | 0.90 | 0.90 | 0.84 | 0.79 |
| Coronary heart disease | 423 (4.6%) | 3 | age (100%), sex (100%), chest pain (100%) | 0.88 | 0.89 | 0.84 | 0.78 |
| COPD | 846 (9.2%) | 4 | age (100%), general health (100%), smoked 100 cigarettes (100%), short of breath on stairs (100%) | 0.83 | 0.85 | 0.85 | 0.66 |
| Arthritis | 2,812 (30.5%) | 4 | age (100%), trouble sleeping (100%), general health (100%), greatest weight (100%) | 0.82 | 0.84 | 0.81 | 0.70 |
| Diabetes | 1,420 (15.8%) | 3 | age (100%), general health (100%), weight 10 years ago (100%) | 0.82 | 0.84 | 0.78 | 0.72 |
| Cancer | 1,006 (10.9%) | 2 | age (100%), trouble sleeping (80%) | 0.80 | 0.80 | 0.81 | 0.63 |
| Hypertension | 3,563 (38.6%) | 3 | age (100%), general health (100%), greatest weight (100%) | 0.79 | 0.80 | 0.77 | 0.68 |
| Stroke | 487 (5.3%) | 2 | age (100%), general health (100%) | 0.78 | 0.81 | 0.77 | 0.64 |
| High cholesterol | 3,261 (35.6%) | 4 | age (100%), trouble sleeping (100%), ate fewer carbohydrates (100%), chest pain (100%) | 0.76 | 0.78 | 0.77 | 0.63 |
| Thyroid problem | 1,080 (11.7%) | 3 | age (100%), sex (100%), trouble sleeping (100%) | 0.72 | 0.73 | 0.80 | 0.50 |
| Liver condition | 462 (5.0%) | 7 | age (100%), moving or speaking slowly or too fast (60%), trouble sleeping (80%), general health (100%), ate fewer carbohydrates (60%), 4/5 drinks every day (100%), chest pain (80%) | 0.71 | 0.72 | 0.77 | 0.49 |
| Asthma | 1,442 (15.6%) | 4 | age (100%), trouble sleeping (100%), sleepy during the day (60%), short of breath on stairs (100%) | 0.68 | 0.69 | 0.82 | 0.44 |

Item wording: [NHANES 2017 to March 2020 questionnaires](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&Cycle=2017-2020). Variable names are in `results/nhanes/recommended_items.csv`.

### Reproducing the examples

```
pip install -r requirements.txt
make nhis MODE=PROD
make nhanes MODE=PROD
```

The survey files are downloaded to `data/`. The full results (survey variable names, item stability, coefficients) and the figures are written to `results/`. Only items asked in both NHIS years are used.

## Installation

```
git clone https://github.com/charlie42/item_recommender.git
cd item_recommender
pip install .
```

## Usage

```
item-recommender --data-path [DATA_PATH] --config-path [CONFIG_PATH] --mode [MODE]
```

Or, from the project directory:

```
make run DATA=[DATA_PATH]
```

[DATA_PATH] is your dataset in csv format. [CONFIG_PATH] is the config file. There is an example config in `example_config/`.

Dataset format:
- Output column names start with the output column prefix from the config ("Diag." by default) and contain 0 or 1. Rows with a missing value in an output column are skipped for that output.
- All other columns are items and should be numeric. Items with a larger fraction of missing values than `max missing fraction` are dropped.

The output goes to a directory named after the dataset file. It contains:
- `recommended_items.csv`: the number of participants and cases for each output, the recommended items, each item's stability (the fraction of cross-validation folds that selected it) and coefficient on standardized responses, performance on the test set with the recommended items and with all items, the recommended probability threshold, and test sensitivity and specificity at that threshold.
- `results.joblib`: the recommended items, cross-validation and test set performance for every number of items, and the final model trained on all data with the recommended items.

The sign of a coefficient depends on how the item is coded. In NHIS, yes is coded 1 and no 2, so a negative coefficient means the predicted probability is higher for a yes answer. Such items can be recoded before running the recommender.

To run the tests:

```
pip install pytest
pytest
```

The tests plant items in a synthetic dataset and check that the recommender finds them.

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
- `exclude columns`: columns that are neither items nor outputs, such as IDs or total scores.

## License
This project is licensed under the MIT License.
