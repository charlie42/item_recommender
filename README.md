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
`git clone https://github.com/charlie42/item-recommender.git`

2. Navigate to the project directory:
`cd item-recommender`

3. Install the required dependencies:
`pip install -r requirements.txt`

4. Install item-recommender as a global package:
`pip install .`

## Usage
To use item-recommender, run the following command:
`item-recommender --data-path [DATASET_PATH] --config-path [CONFIG PATH]`

Replace [DATA_PATH] with the path to your dataset in csv format, and [CONFIG_PATH] with the path to the config file. The example of the config file format is in the example_config subdirectory. The name of the dataset file will be used as the name of the directory where the output will be stored. Output columns named should start with "Output."

## License
This project is licensed under the MIT License.