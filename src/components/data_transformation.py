'''
In data transformation, we basically vectorized the data, into numbers.
Creating the pipelines for this. 

'''
import os
import sys
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_and_load
import pandas as pd
import numpy as np



# Create the transformation config class

@dataclass
class DataTransformConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

# Create the data_transformation class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformConfig()

    
    def get_data_transformer_object(self):
         # this file will basically create or return the preprocessor obj file:
        logging.info("Started get_data_transformer_object")
        try:
            numerical_columns = ['latitude', 'longitude', 'number_of_reviews', 'reviews_per_month',
                                    'calculated_host_listings_count', 'availability_365', 'last_review_day',
                                    'neighbourhood_freq',  'minimum_nights_log']

    
            categorical_columns= ['neighbourhood_group', 'room_type']            
            # Create the numerical pipeline:



            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = 'median')),
                    ("scaler", StandardScaler())

                ]
            )

            cat_pipeline = Pipeline(
                steps = 
                [
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ("onehot", OneHotEncoder())
                ]
            )

            preprocessor = ColumnTransformer(
                 
                [
                    ("numerical_columns",  num_pipeline, numerical_columns),
                    ("categorical_columns", cat_pipeline, categorical_columns)
                ]
            )
            logging.info("Preprocessor object returned:")
            return preprocessor
            

        except Exception as e:
            raise CustomException(e, sys)
            
    def initiate_transformation(self, train_path, test_path):
        try:

            logging.info("Started data transformation")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Completed reading the test and train data")
            logging.info("Obtaining the preprocesor object.")

            preprocessor_obj = self.get_data_transformer_object()
            print(f"Train_df columns: {train_df.columns}")
            print(f"Test_df_columns: {test_df.columns}")

            target_column_name = 'price_log'

            input_feature_train_df = train_df.drop(columns = [target_column_name], axis = 1)
            target_feature_train_df = train_df['price_log']

            input_feature_test_df = test_df.drop(columns = [target_column_name], axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying the preprocesing object on training dataframe and testing dataframe."
            )
            print("Columns expected by the preprocessor:")
            print("Numerical columns:", ['latitude', 'longitude', 'number_of_reviews', 'reviews_per_month',
                             'calculated_host_listings_count', 'availability_365', 'last_review_day',
                             'neighbourhood_freq', 'price_log', 'minimum_nights_log'])
            
            print("Categorical columns:", ['neighbourhood_group', 'room_type'])

            print("\nActual columns in input_feature_train_df:")
            print(input_feature_train_df.columns.tolist())

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            print(f"Return DataType after applying preprocesor object.")
            print(f"input_feature_train_arr: {type(input_feature_train_arr)}")
            print(f"Input feature test arr: {type(input_feature_test_arr)}")

            print(f"Data Type of the train and test(target columns): {type(target_feature_test_df)}")

            # Now concatenating the columns: 

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_and_load(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )





        except Exception as e:
            raise CustomException(e, sys)
        
    
    
    