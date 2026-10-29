import os
import time
import multiprocessing
import joblib
import pandas as pd
from item_recommender.data_loader import DataLoader
from item_recommender.config import Config
from item_recommender.data_splitter import DataSplitter
from item_recommender.cross_validator import CrossValidator
from item_recommender.argument_parser import ArgumentParser

def process_output(args):
    """Process a single output column."""

    output_column, output_columns, data, config = args

    # Split the data
    split_data = DataSplitter(data, output_columns, output_column).split(
        config.get_item("split percentage")
    )

    # Perform cross-validation, then select items on the whole training set
    cross_validator = CrossValidator(split_data, output_column, config)
    cv_performances = cross_validator.cross_validate()
    return output_column, cross_validator.recommend(cv_performances)

def save_results(results, output_dir):
    """Save the results."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(results, os.path.join(output_dir, "results.joblib"))
    pd.DataFrame([
        {
            "output": output,
            "n items": len(result["items"]),
            "test performance": result["test_performance"],
            "items": ", ".join(result["items"]),
        }
        for output, result in results.items()
    ]).to_csv(os.path.join(output_dir, "recommended_items.csv"), index=False)


def main():
    start_time = time.time() # Start timer for measuring total time of script
    args = ArgumentParser().args

    # Load the config
    config = Config()
    config.load_config(args.config_path, args.mode)
    print("Config loaded")

    # Load the dataset
    data_loader = DataLoader(args.data_path, config)
    data = data_loader.load_data()
    output_columns = data_loader.get_output_columns()
    print(f"Data loaded, {len(output_columns)} output columns found")

    # Create a multiprocessing Pool to process output columns in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(
            process_output,
            [(column, output_columns, data, config) for column in output_columns]
    )

    output_dir = os.path.splitext(os.path.basename(args.data_path))[0]
    save_results(dict(results), output_dir)

    print(f"Done! Total time: {time.time() - start_time} seconds")

if __name__ == "__main__":
    main()
