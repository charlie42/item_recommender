import argparse
import time
import multiprocessing
import joblib
from item_recommender.data_loader import DataLoader
from item_recommender.config import Config
from item_recommender.data_splitter import DataSplitter
from item_recommender.cross_validator import CrossValidator
from item_recommender.argument_parser import ArgumentParser

def process_output(args):
    """Process a single output column."""

    output_column, data, config = args
    
    # Split the data
    split_data = DataSplitter(data, output_column).split(config["outer_cv"])

    # Perform cross-validation
    output, fitted_objects, perforamance = CrossValidator(
        split_data, 
        output_column, 
        config
    ).cross_validate()
    return output, fitted_objects, perforamance

def aggregate_results(results):
    """
    Aggregate the results from all output columns.
    Format of perforamnce: {
        "perf_all_features": [0.8, 0.9, 0.7, 0.8, 0.9]
        "perf_all_checked_features": [0.7, 0.8, 0.6, 0.7, 0.8],
        "opt_ns_of_features": [2, 3, 4, 5, 6],
        "perf_on_features": {
            1:[0.6, 0.7, 0.5, 0.6, 0.7],
            2:[0.7, 0.8, 0.6, 0.7, 0.8],]
        }
    }
    """
    # Aggregate the results
    fitted_objects_dict = {}
    performances_dict = {}

    for result in results:
        output, fitted_objects, perforamances = result
        
        # Aggregate the fitted objects
        if output not in fitted_objects_dict:
            fitted_objects_dict[output] = []
        fitted_objects_dict[output].append(fitted_objects)

        # Aggregate the performances
        if output not in performances_dict:
            performances_dict[output] = []
        performances_dict[output].append(perforamances)

    return fitted_objects_dict, performances_dict

def save_results(fitted_objects_dict, performances_dict):
    """Save the results."""
    joblib.dump(fitted_objects_dict, "fitted_objects_dict.joblib")
    joblib.dump(performances_dict, "performances_dict.joblib")


def main():
    start_time = time.time() # Start timer for measuring total time of script
    args = ArgumentParser().args

    # Load the config
    config = Config().load_config(args.config_path, args.mode)
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
            [(column, data, config) for column in output_columns]
    )

    fitted_objects_dict, performances_dict = aggregate_results(results)
    save_results(fitted_objects_dict, performances_dict)

    print(f"Done! Total time: {time.time() - start_time} seconds")

if __name__ == "__main__":
    main()