import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.Machine_Recommendation.exception import customexception
from src.Machine_Recommendation.logger import logging
from src.Machine_Recommendation.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Performs basic data validation checks.
        
        Args:
            data: The DataFrame to validate.
        
        Raises:
            customexception: If data validation fails.
        """
        try:
            # Check for missing values
            if data.isnull().values.any():
                raise customexception("Missing values found in the data.")
            
            # Add other validation checks as needed

        except Exception as e:
            logging.info(f"Data validation failed: {e}")
            raise customexception(e, sys) from e

    def transform_data(self, train_path: str, test_path: str) -> tuple:
        try:
            logging.info('Data Transformation initiated')
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train 1  and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            # Validate the data
            self._validate_data(train_df)
            self._validate_data(test_df)

            # Convert to dense arrays (if necessary) 
            # - This might be removed if your data is always dense
            if hasattr(train_df, 'toarray'):
                train_df = train_df.toarray()
            if hasattr(test_df, 'toarray'):
                test_df = test_df.toarray()

            return train_df, test_df

        except Exception as e:
            logging.info(f"Exception occurred in data transformation: {e}")
            raise customexception(e, sys) from e