import sys
import importlib

class Parameters:
    def __init__(self):
        self.dict = {}

    def read_parameters(self, path):
        try:
           spec = importlib.util.spec_from_file_location("general_parameters", path)
           module = importlib.util.module_from_spec(spec)
           sys.modules["general_parameters"] = module
           spec.loader.exec_module(module)
           self.dict = module.general_parameters
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
