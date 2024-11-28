import os
import sys
import pandas as pd
from src.Machine_Recommendation.exception import customexception
from src.Machine_Recommendation.logger import logging
from src.Machine_Recommendation.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            scaled_data=preprocessor.transform(features)
            
            pred=model.predict(scaled_data)
            
            return pred
            
            
        
        except Exception as e:
            raise customexception(e,sys)
    
    
    
class CustomData:
    def __init__(self,
                 Ratings:float,
                 RAM:float,
                 ROM:float,
                 Mobile_Size:float,
                 Primary_Cam:int,
                 Selfi_Cam:float,
                 Battery_Power:int,
                 ):
        
        self.Ratings=Ratings
        self.RAM=RAM
        self.ROM=ROM
        self.Mobile_Size=Mobile_Size
        self.Primary_Cam=Primary_Cam
        self.Selfi_Cam=Selfi_Cam
        self.Battery_Power = Battery_Power
       
            
                
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'Ratings':[self.Ratings],
                    'RAM':[self.RAM],
                    'ROM':[self.ROM],
                    'Mobile_Size':[self.Mobile_Size],
                    'Primary_Cam':[self.Primary_Cam],
                    'Selfi_Cam':[self.Selfi_Cam],
                    'Battery_Power':[self.Battery_Power],
                  
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)