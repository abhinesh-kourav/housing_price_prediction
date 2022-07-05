import os
import sys
import pandas as pd
from housing.constants import SCHEMA_CATEGORICAL_COLUMNS_KEY, SCHEMA_COLUMNS_KEY, SCHEMA_DOMAIN_VALUE_KEY, SCHEMA_NUMERICAL_COLUMNS_KEY, SCHEMA_TARGET_COLUMN_KEY
from housing.logger import logging
from housing.exception import HousingException
from housing.entity.config_entity import DataValidationConfig
from housing.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from housing.util.util import read_yaml_file
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import json

class DataValidation:
    def __init__(self,
                data_validation_config: DataValidationConfig,
                data_ingestion_artifact: DataIngestionArtifact) -> None:
        try:
            logging.info(f"{'='*20} Data Validation Log Started {'='*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.train_file_path = self.data_ingestion_artifact.train_file_path
            self.test_file_path = self.data_ingestion_artifact.test_file_path
            self.train_df = pd.read_csv(self.train_file_path)
            self.test_df = pd.read_csv(self.test_file_path)
        except Exception as e:
            raise HousingException(e,sys) from e

    def do_train_test_files_exist(self)-> True:
        try:
            logging.info("Checking if test and trai file exists.")
            does_train_file_exist = False
            does_test_file_exist = True
            does_train_file_exist = os.path.exists(self.train_file_path)
            does_test_file_exist = os.path.exists(self.test_file_path)
            if not does_train_file_exist:
                raise Exception(f"Train file: [{self.train_file_path}] does not exists.")
            if not does_test_file_exist:
                raise Exception(f"Test file: [{self.test_file_path}] does not exists.")
            if does_test_file_exist and does_train_file_exist:
                logging.info(f"Both train file: [{self.train_file_path}] and test file: [{self.test_file_path} exist.")
                return True
        except Exception as e:
            raise HousingException(e,sys) from e

    def validate_dataset_schema(self)-> True:
        try:
            schema = read_yaml_file(self.data_validation_config.schema_file_path)
            schema_columns = schema[SCHEMA_COLUMNS_KEY]
            schema_domain_value = schema[SCHEMA_DOMAIN_VALUE_KEY]
            schema_numerical_columns = schema[SCHEMA_NUMERICAL_COLUMNS_KEY]
            schema_categorical_columns = schema[SCHEMA_CATEGORICAL_COLUMNS_KEY]
            logging.info("Checking no. of columns in train and test dataset")
            check_no_of_columns = len(schema_columns) == len(self.train_df.columns) and len(schema_columns) == len(self.test_df.columns)
            if not check_no_of_columns:
                raise Exception("Train and/or test dataset does not have columns given in schema.")
            else:
                logging.info('No. of columns are same in train and test dataset and in schema file.')
                logging.info("Checking columns names")
                for column in schema_columns.keys():
                    if column not in self.train_df.columns:
                        raise Exception(f"Train dataset does not have column '{column}' required in schema file")
                    if column not in self.test_df.columns:
                        raise Exception(f"Test dataset does not have column '{column}' required in schema file")
                else:
                    logging.info(f"Train and test dataset have column required in schema file")
            logging.info("Checking the datatypes of numerical columns")
            for column in schema_numerical_columns:
                if schema_columns[column] not in f'{self.train_df[column].dtype}':
                    raise Exception(f"Column '{column}' in train dataset does not have required dtype.")
                if schema_columns[column] not in f'{self.test_df[column].dtype}':
                    raise Exception(f"Column '{column}' in test dataset does not have required dtype.")
            else:
                logging.info("Both train and test datasets columns have required numerical datatypes")
            logging.info("Checking datatypes of categorical columns")
            for column in schema_categorical_columns:
                if self.train_df[column].dtype != 'object':
                    raise Exception(f"Column {column} in train dataset is does not have categorical values")
                if self.test_df[column].dtype != 'object':
                    raise Exception(f"Column {column} in test dataset is does not have categorical values")
            else:
                logging.info("Both training and test datasets have required categorical columns")
            logging.info("Checking the domain values of categorical columns")
            for column, cats in schema_domain_value.items():
                logging.info(f"Checking domain values of column '{column}'")
                for cat in self.train_df[column].unique():
                    if cat not in schema_domain_value[column]:
                        raise Exception(f"category '{cat}' is an unwanted value in column '{column}' of test dataset")
                for cat in self.test_df[column].unique():
                    if cat not in schema_domain_value[column]:
                        raise Exception(f"category '{cat}' is an unwanted value in column '{column}' of test dataset")
                else:
                    logging.info(f"column '{column}' has all the required categories and no extra category")
            logging.info("Data Validation Successful!")
            return True
        except Exception as e:
            raise HousingException(e,sys) from e

    def get_and_save_data_drift_report(self):
        try:
            profile = Profile(sections = [DataDriftProfileSection()])
            profile.calculate(self.train_df, self.test_df)
            report = json.loads(profile.json())
            report_file_path = self.data_validation_config.report_file_path
            os.makedirs(os.path.dirname(report_file_path), exist_ok=True)
            with open(self.data_validation_config.report_file_path, "w") as report_file:
                json.dump(report, report_file, indent=4)
                return report
        except Exception as e:
            raise HousingException(e,sys) from e

    def save_data_drift_report_page(self):
        try:
            dashboard = Dashboard(tabs= [DataDriftTab()])
            dashboard.calculate(self.train_df, self.test_df)
            dashboard.save(self.data_validation_config.report_page_file_path)
        except Exception as e:
            raise HousingException(e,sys) from e

    def does_data_drift_occur(self)-> bool:
        try:
            report = self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()
            return True
        except Exception as e:
            raise HousingException(e,sys) from e

    def initiate_data_validation(self)-> DataValidationArtifact:
        try:
            train_test_exist = self.do_train_test_files_exist()
            validation_status =self.validate_dataset_schema()
            self.does_data_drift_occur()
            data_validation_artifact = DataValidationArtifact(schema_file_path=self.data_validation_config.schema_file_path,
                                                            report_file_path=self.data_validation_config.report_page_file_path,
                                                            report_page_file_path=self.data_validation_config.report_page_file_path,
                                                            is_validated=True,
                                                            message="Data Validation performed sucessfully.")
            logging.info(f"Data Validation Artifact : {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise HousingException(e,sys) from e

    def __del__(self):
        logging.info(f"{'='*20}Data Validation log ended{'='*20} \n\n")