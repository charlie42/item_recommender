import argparse
import time
import multiprocessing
from data_loader import DataLoader
from config import Config
from data_splitter import DataSplitter
from cross_validator import CrossValidator
from argument_parser import ArgumentParser

def process_output(output_column, data, config):
    """Process a single output column."""
    # Split the data
    split_data = DataSplitter(data).split_data(config)
    output, fitted_objects, perforamance = CrossValidator.cross_validate(
        split_data, 
        output_column, 
        config
    )
    return output, fitted_objects, perforamance

    # Save the best model and selected features
    # Implement saving logic here

def aggregate_results(results):
    """Aggregate the results from all output columns."""
    for result in results:
        output, fitted_objects, perforamance = result
        # Implement aggregation logic here

def main():
    start_time = time.time() # Start timer for measuring total time of script
    args = ArgumentParser.parse_arguments()

    try:
        # Load the dataset
        data_loader = DataLoader(args.dataset_path)
        data = data_loader.load_data()
        output_columns = data_loader.get_output_columns()

        # Load the config
        config = Config(args.config_path, args.mode).load_config()

        # Create a multiprocessing Pool to process output columns in parallel
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(
                process_output, 
                [(column, data, config) for column in output_columns]
        )

        results = aggregate_results(results)

        print(f"Don! Total time: {time.time() - start_time} seconds")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()