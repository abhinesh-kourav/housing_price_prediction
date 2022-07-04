import os
import sys
import pandas as pd
from housing.constants import SCHEMA_CATEGORICAL_COLUMNS_KEY, SCHEMA_COLUMNS_KEY, SCHEMA_DOMAIN_VALUE_KEY, SCHEMA_NUMERICAL_COLUMNS_KEY, SCHEMA_TARGET_COLUMN_KEY
from housing.logger import logging
from housing.exception import HousingException
from housing.entity.config_entity import DataValidationConfig
from housing.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from housing.util.util import read_yaml_file


class DataValidation:
    def __init__(self,
                data_validation_config: DataValidationConfig,
                data_ingestion_artifact: DataIngestionArtifact) -> None:
        try:
            logging.info(f"{'='*20} Data Validation Log Started {'='*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise HousingException(e,sys) from e

    def do_train_test_files_exist(self)-> True:
        try:
            logging.info("Checking if test and trai file exists.")
            does_train_file_exist = False
            does_test_file_exist = True
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            does_train_file_exist = os.path.exists(train_file_path)
            does_test_file_exist = os.path.exists(test_file_path)
            if not does_train_file_exist:
                raise Exception(f"Train file: [{train_file_path}] does not exists.")
            if not does_test_file_exist:
                raise Exception(f"Test file: [{test_file_path}] does not exists.")
            if does_test_file_exist and does_train_file_exist:
                logging.info(f"Both train file: [{train_file_path}] and test file: [{test_file_path} exist.")
                return True
        except Exception as e:
            raise HousingException(e,sys) from e

    def validate_dataset_schema(self)-> True:
        try:
            test_file_path = self.data_ingestion_artifact.test_file_path
            train_file_path = self.data_ingestion_artifact.train_file_path
            schema = read_yaml_file(self.data_validation_config.schema_file_path)
            schema_columns = schema[SCHEMA_COLUMNS_KEY]
            schema_domain_value = schema[SCHEMA_DOMAIN_VALUE_KEY]
            schema_numerical_columns = schema[SCHEMA_NUMERICAL_COLUMNS_KEY]
            schema_categorical_columns = schema[SCHEMA_CATEGORICAL_COLUMNS_KEY]
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            """
            Validation
            1. No. of columns
            2. Check the values of ocean proximity
            3. Check column names
            4. Try to check the datatypes
            """
            logging.info("Checking no. of columns in train and test dataset")
            check_no_of_columns = len(schema_columns) == len(train_df.columns) and len(schema_columns) == len(test_df.columns)
            if not check_no_of_columns:
                raise Exception("Train and/or test dataset does not have columns given in schema.")
            else:
                logging.info('No. of columns are same in train and test dataset and in schema file.')
                logging.info("Checking columns names")
                for column in schema_columns.keys():
                    if column not in train_df.columns:
                        raise Exception(f"Train dataset does not have column '{column}' required in schema file")
                    if column not in test_df.columns:
                        raise Exception(f"Test dataset does not have column '{column}' required in schema file")
                else:
                    logging.info(f"Train and test dataset have column required in schema file")
            logging.info("Checking the datatypes of numerical columns")
            for column in schema_numerical_columns:
                if schema_columns[column] not in f'{train_df[column].dtype}':
                    raise Exception(f"Column '{column}' in train dataset does not have required dtype.")
                if schema_columns[column] not in f'{test_df[column].dtype}':
                    raise Exception(f"Column '{column}' in test dataset does not have required dtype.")
            else:
                logging.info("Both train and test datasets columns have required datatypes")
            logging.info("Checking the domain values of categorical columns")
            for column, cats in schema_domain_value.items():
                logging.info(f"Checking domain values of column '{column}'")
                for cat in train_df[column].unique():
                    if cat not in schema_domain_value[column]:
                        raise Exception(f"category '{cat}' is an unwanted value in column '{column}'")
                    else:
                        logging.info(f"column '{column}' has all the required categories and no extra category")
            return True
        except Exception as e:
            raise HousingException(e,sys) from e

    def initiate_data_validation(self)-> DataValidationArtifact:
        try:
            train_test_exist = self.do_train_test_files_exist()
            validation_status =self.validate_dataset_schema()

        except Exception as e:
            raise HousingException(e,sys) from e