import os, sys
from housing.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact
from housing.entity.config_entity import DataTransformationConfig
from housing.exception import HousingException
from housing.logger import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from housing.constants import *
import numpy as np, pandas as pd
from housing.util.util import read_yaml_file,save_object,save_numpy_array_data,load_data

class FeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True,
                 total_rooms_ix=3,
                 population_ix=5,
                 households_ix=6,
                 total_bedrooms_ix=4, columns=None):
        """
        FeatureGenerator Initialization
        add_bedrooms_per_room: bool
        total_rooms_ix: int index number of total rooms columns
        population_ix: int index number of total population columns
        households_ix: int index number of  households columns
        total_bedrooms_ix: int index number of bedrooms columns
        """
        try:
            self.columns = columns
            if self.columns is not None:
                total_rooms_ix = self.columns.index(COLUMN_TOTAL_ROOMS)
                population_ix = self.columns.index(COLUMN_POPULATION)
                households_ix = self.columns.index(COLUMN_HOUSEHOLDS)
                total_bedrooms_ix = self.columns.index(COLUMN_TOTAL_BEDROOM)

            self.add_bedrooms_per_room = add_bedrooms_per_room
            self.total_rooms_ix = total_rooms_ix
            self.population_ix = population_ix
            self.households_ix = households_ix
            self.total_bedrooms_ix = total_bedrooms_ix
        except Exception as e:
            raise HousingException(e, sys) from e

    def fit(self, X, y=None):
        try:
            return self
        except Exception as e:
            raise HousingException(e,sys) from e

    def transform(self, X, y=None):
        try:
            room_per_household = X[:, self.total_rooms_ix] / \
                                 X[:, self.households_ix]
            population_per_household = X[:, self.population_ix] / \
                                       X[:, self.households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, self.total_bedrooms_ix] / \
                                    X[:, self.total_rooms_ix]
                generated_feature = np.c_[
                    X, room_per_household, population_per_household, bedrooms_per_room]
            else:
                generated_feature = np.c_[
                    X, room_per_household, population_per_household]

            return generated_feature
        except Exception as e:
            raise HousingException(e, sys) from e


class DataTransformation:
    def __init__(self, data_transformation_config:DataTransformationConfig,
                 data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_artifact:DataValidationArtifact ) -> None:
        try:
            logging.info(f"{'='*20} Data Transformation Log Started {'='*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise HousingException(e,sys) from e

    def get_transformer_object(self)-> ColumnTransformer:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path
            schema = read_yaml_file(schema_file_path)
            numerical_columns = schema[SCHEMA_NUMERICAL_COLUMNS_KEY]
            categorical_columns = schema[SCHEMA_CATEGORICAL_COLUMNS_KEY]
            num_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                           ('feature_generator', FeatureGenerator(add_bedrooms_per_room=self.data_transformation_config.add_bedroom_per_room,
                                                                                  columns=numerical_columns)),
                                           ('scaling',StandardScaler())
                                          ])
            cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                           ('one_hot_encoder', OneHotEncoder()),
                                           ('scaling',StandardScaler(with_mean=False))
                                          ])            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            preprocessing = ColumnTransformer(transformers=[('num_pipeline', num_pipeline, numerical_columns),
                                                            ('cat_pipeline', cat_pipeline, categorical_columns),
                                                            ])
            return preprocessing
        except Exception as e:
            raise HousingException(e,sys) from e

    def initiate_data_transformation(self)-> DataTransformationArtifact:
        try:
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_transformer_object()
            schema_file_path = self.data_validation_artifact.schema_file_path
            schema = read_yaml_file(schema_file_path)
            logging.info("Obtaining train and test dataset")
            train_df = load_data(self.data_ingestion_artifact.train_file_path, schema_file_path)
            test_df = load_data(self.data_ingestion_artifact.test_file_path, schema_file_path)
            target_column = schema[SCHEMA_TARGET_COLUMN_KEY]
            logging.info("Splitting the datasets into input and output features")
            X_train = train_df.drop(target_column, axis=1)
            y_train = train_df[[target_column]]
            X_test = test_df.drop(target_column, axis=1)
            y_test = test_df[[target_column]]
            logging.info("Transforming input features using preprocessing object file.")
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)
            logging.info("Concatenating transformed input features with output features")
            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]
            logging.info("Saving transformed train and test datasets")
            transformed_test_dir = self.data_transformation_config.transformed_test_dir
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_file_path = os.path.join(transformed_test_dir,
                                                      os.path.basename(self.data_ingestion_artifact.test_file_path).replace('.csv','.npz'))
            transformed_train_file_path = os.path.join(transformed_train_dir,
                                                      os.path.basename(self.data_ingestion_artifact.train_file_path).replace('.csv','.npz'))       
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)
            preprocessed_object_file_path = self.data_transformation_config.preprocessed_object_file_path
            logging.info("Saving preprocesing object file")
            save_object(preprocessed_object_file_path, preprocessing_obj)
            data_transformation_artifact = DataTransformationArtifact(transformed_test_file_path=transformed_test_file_path,
                                                                    transformed_train_file_path=transformed_train_file_path,
                                                                    preprocessed_object_file_path=preprocessed_object_file_path,
                                                                    is_transformed=True,
                                                                    message="Data Transformation completed successfully.")
            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise HousingException(e,sys) from e

    def __del__(self):
        logging.info(f"{'='*20}Data Transformation log completed.{'='*20} \n\n") 