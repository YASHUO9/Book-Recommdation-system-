
import pandas as pd
from src.Machine_Recommendation.logger import logging
from src.Machine_Recommendation.exception import customexception

import os
import sys
from sklearn.model_selection import train_test_split
from pathlib import Path

class DataIngestionConfig:
    
    raw_data_path:str=os.path.join("artifacts","book_pivoted.csv")
    raw_data_path_rating:str=os.path.join("artifacts","ratings.csv")       
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")
    final_rating_path:str=os.path.join("artifacts","final_rating.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    
    def initiate_data_ingestion(self):
        logging.info("data ingestion started")
        
        try:
            data=pd.read_csv(Path(os.path.join("notebooks/data","book.csv")))
            data2=pd.read_csv(Path(os.path.join("notebooks/data","ratings.csv")), sep=";", on_bad_lines='skip', encoding='latin-1')
            final_rating = pd.read_csv(Path(os.path.join("notebooks/data","final_rating.csv")), sep=";", on_bad_lines='skip', encoding='latin-1')
            
            
            logging.info(" i have read dataset as a df")
            
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            data2.to_csv(self.ingestion_config.raw_data_path_rating,index=False)  
            final_rating.to_csv(self.ingestion_config.final_rating_path,index=False)          
            logging.info(" i have saved the raw dataset in artifact folder")
            
            logging.info("here i have performed train test split")
            
            train_data,test_data=train_test_split(data,test_size=0.25)
            logging.info("train test split completed")
            
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            
            logging.info("data ingestion part completed")
            
         
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
            
        except Exception as e:
           logging.info("exception during occured at data ingestion stage")
           raise customexception(e,sys)


data_ingestion=DataIngestion()
train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
print(train_data_path)
print(test_data_path)