from sklearn.model_selection import train_test_split

class DataSplitter:
    """
    Split the data into train and test sets.
    """

    def __init__(self, data, output_column):
        self.data = data
        self.output_column = output_column
        self.split_data = None
        
    def split(self, split_percentage):
        """
        Split the data into train and test sets.
        Returns a dictionary with the split data (X_train, X_test, y_train, y_test)
        """
        
        # Split train and test sets
        X_train, X_test, y_train, y_test = self._split_data(split_percentage)

        # Save the split data
        self.split_data = self._save_split_data(X_train, X_test, y_train, y_test)

        return self.split_data
    
    def _split_data(self, split_percentage):
        """Split the data into train and test sets."""

        # Split train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, 
            self.data[self.output_column], 
            test_size=split_percentage, 
            stratify=self.data[self.output_column], 
            random_state=1)
        
        return X_train, X_test, y_train, y_test
    
    def _save_split_data(self, X_train, X_test, y_train, y_test):
        """Save the split data."""
        return {
            "X_train": X_train, "X_test": X_test, 
            "y_train": y_train, "y_test": y_test
        }
        
