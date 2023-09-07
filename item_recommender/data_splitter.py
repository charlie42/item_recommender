from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, data, output_column):
        self.data = data
        self.output_column = output_column
        
    def split_data(self, config):
        """Split the data into train and test sets."""
        split_percentage = config["split_percentage"]

        # Split train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, 
            self.data[self.output_col], 
            test_size=split_percentage, 
            stratify=self.data[self.output_col], 
            random_state=1)
        
        return X_train, X_test, y_train, y_test
        
