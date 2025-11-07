
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.feature_engineering.NYC_Air_2019 import NYCFeatureEngineer
from src.components.data_transformation import DataTransformation
from dataclasses import dataclass
import os
import sys
from sklearn.model_selection import train_test_split
from src.components.model_trainer import ModelTrainer


# Create the config class for the datasets we want to work with their respective folders and file path 
@dataclass
class DataIngestionConfig():
    raw_data_path:str = os.path.join('artifacts', 'raw_data.csv')
    train_data_path: str = os.path.join('artifacts', 'train_data.csv')
    test_data_path: str = os.path.join('artifacts', 'test_data.csv')


# Create the dataIngestion class 

class DataIngestion():
    def __init__(self):

        self.ingestion_config = DataIngestionConfig() # we instantiate the config class with our class level variable of DataIngestion

    # Creating the another function for initiating the DataIngestion:

    def initiate_ingestion(self):

        logging.info('DataIngestion Process Initiated:')
        try:
            df = pd.read_csv('notebook/EDA/NYC_Data.csv' )
            logging.info('Dataset succesfully read, entering the next process')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            logging.info('Initializing the train_test_split')
            train_set, test_set = train_test_split(df, test_size = 0.3, random_state = 42)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info('Sucessfully splited and save in respective directores: Ingestion of the Data is completed')
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e, sys)
        # Completed that's it

if __name__=='__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_ingestion()

    # Instantiate the feature engineer

    feature_engineer = NYCFeatureEngineer()

    # Engineered features
    train_data_read = pd.read_csv(train_data)
    test_data_read = pd.read_csv(test_data)
    train_csv_path, test_csv_path = feature_engineer.process_and_save(train_data_read, test_data_read)
    print(train_csv_path, test_csv_path)

    transformed_object = DataTransformation()
    train_arr, test_arr, path = transformed_object.initiate_transformation(train_csv_path, test_csv_path)

    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_arr, test_arr)









# python -m src.components.data_ingestion