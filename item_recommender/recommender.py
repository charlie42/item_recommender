import argparse
import time
import multiprocessing
from src.data_processing.dataset_loader import load_dataset
from src.model_training.train import train_model
from src.model_training.cross_validator import cross_validate
from src.model_training.hyperparameter_search import hyperparameter_search
from src.model_training.config_reader import Config

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path", 
        type=str, 
        help="Path to the dataset file")
    parser.add_argument(
        "--config-path", 
        type=str, 
        help="Path to the config file")
    parser.add_argument(
        "--mode", 
        type=str, 
        help="Mode of the script: DEV, DEBUG, or PROD")
    return parser.parse_args()

def process_output(output_column, data, config):
    """Process a single output column."""
    
    
    

    # Save the best model and selected features
    # Implement saving logic here

def aggregate_results(results):
    """Aggregate the results from all output columns."""
    # Implement aggregation logic here

def main():
    start_time = time.time() # Start timer for measuring total time of script
    args = parse_arguments()

    try:
        # Load the dataset
        dataset_loader = DatasetLoader(args.dataset_path)
        dataset_loader.load_data()
        data = dataset_loader.get_data()
        output_columns = dataset_loader.get_output_columns()

        # Load the config
        config = Config(args.config_path, args.mode)

        # Create a multiprocessing Pool to process output columns in parallel
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(process_output, [(column, data, config) for column in output_columns])

        results = aggregate_results(results)

        print(f"Don! Total time: {time.time() - start_time} seconds")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()