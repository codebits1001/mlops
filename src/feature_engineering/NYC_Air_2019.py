from .base_feature_engineer import BaseFeatureEngineer
import pandas as pd
import numpy as np
import os

from src.exception import CustomException
from src.logger import logging

class FeatureConfig:
    train_df_file_path: str = os.path.join('artifacts', 'train_featured_df.csv')
    test_df_file_path: str = os.path.join('artifacts', 'test_featured_df.csv')

class NYCFeatureEngineer( BaseFeatureEngineer):
    def __init__(self):
        self.preprocessed_file_df = FeatureConfig()
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
          # creates the seperate copy of the dataframe, prevents accidental changes to the orginal dataframe outside the function
        
        logging.info("Initiating the feature engineering:")
        
        df["last_review_year"] = pd.to_datetime(df["last_review"]).dt.year
        df["last_review_month"] = pd.to_datetime(df["last_review"]).dt.month
        df["last_review_day"] = pd.to_datetime(df["last_review"]).dt.day       
        df.drop("last_review", axis=1, inplace=True)

        logging.info("last review columns done:")


            # High cardinality categorical column (frequency encoding) ----

        
        freq = df["neighbourhood"].value_counts(normalize = True)
        df["neighbourhood_freq"] = df["neighbourhood"].map(freq)
        df.drop("neighbourhood", axis =1, inplace = True)

        logging.info("frequency target encoding done for the neighbourhood.")
        

        df["price_log"] = np.log1p(df["price"])
        df["minimum_nights_log"] = np.log1p(df["minimum_nights"])
        df.drop(["id","name","host_id","host_name","last_review_month", "last_review_year", "price", "minimum_nights"], axis=1, inplace=True)

        logging.info("Log transformation for price and minimum_nights.")
        df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
        df["last_review_day"] = df["last_review_day"].fillna(0)

        logging.info("All the columns nan values are filled. \n All the original columns are dropped, which are transformed or engineered.")
        
       
       
        
        return df
    

    def process_and_save(self, train_df: pd.DataFrame, test_df: pd.DataFrame):

        # engineer features
        logging.info("process and save function called: (NYC_Air file, line: 59)")
        train_df_eng = self.engineer_features(train_df)
        test_df_eng = self.engineer_features(test_df)

        # make directories
        os.makedirs(os.path.dirname(self.preprocessed_file_df.train_df_file_path), exist_ok = True)
        os.makedirs(os.path.dirname(self.preprocessed_file_df.test_df_file_path), exist_ok = True)

        # save csvs 
        
        logging.info("Saving the engineered data frame(test and train)")
        train_df_eng.to_csv(self.preprocessed_file_df.train_df_file_path, header = True, index = False)
        test_df_eng.to_csv(self.preprocessed_file_df.test_df_file_path, header = True, index = False)

        #return file path 
        logging.info("Returning the file path.")
        return (
            self.preprocessed_file_df.train_df_file_path,
            self.preprocessed_file_df.test_df_file_path
        )