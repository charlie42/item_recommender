class ModeManager:
    def __init__(self, config_dict, mode):
        self.config_dict = config_dict
        self.update_dict = config_dict["modes"][mode]

    def update_config(self):
        """"
        Update the config based on the mode.
        """
        # Replace items from the update dict in the config dict
        print("DEBUG self.update_dict", self.update_dict, "\n")
        for key, value in self.update_dict.items():
            # Find all instances of the key in the config dict
            print("DEBUG key", key, "\n")
            self.config_dict = self._replace_item(self.config_dict, key, value)

        return self.config_dict
    
    def _replace_item(self, obj, key, replace_value):
        for k, v in obj.items():
            if isinstance(v, dict):
                obj[k] = self._replace_item(v, key, replace_value)
        if key in obj:
            obj[key] = replace_value
        return obj



        