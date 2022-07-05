from housing.entity.config_entity import DataInjestionConfig
from housing.exception import HousingException
import sys, os
from housing.logger import logging
from housing.entity.artifact_entity import DataIngestionArtifact
import tarfile
from six.moves import urllib
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


class DataIngestion:
    def __init__(self, data_ingestion_config: DataInjestionConfig) -> None:
        try:
            logging.info(f"{'='*20} Data Ingestion Log Started {'='*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise HousingException(e,sys) from e

    def download_housing_data(self)-> str:
        try:
            #Extracting remote url to download dataset
            download_url = self.data_ingestion_config.dataset_download_url
            #downloaded file's directory
            tgz_download_dir = self.data_ingestion_config.tgz_download_dir
            os.makedirs(tgz_download_dir, exist_ok=True) 
            tgz_file_name = os.path.basename(download_url)
            tgz_file_path = os.path.join(tgz_download_dir,
                                tgz_file_name)
            logging.info(f"Downloading [{tgz_file_name}] from [{download_url}] to [{tgz_download_dir}]")
            urllib.request.urlretrieve(download_url, tgz_file_path)
            logging.info(f"Downloaded [{tgz_file_name}] successfully.")
            return tgz_file_path
        except Exception as e:
            raise HousingException(e,sys) from e

    def extract_tgz_file(self, tgz_file_path:str):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            os.makedirs(raw_data_dir, exist_ok=True)
            logging.info(f"Extracting [{tgz_file_path}]to [{raw_data_dir}]")
            with tarfile.open(tgz_file_path) as tgz_file_obj:
                tgz_file_obj.extractall(path=raw_data_dir)
            logging.info('Extraction finished!')

        except Exception as e:
            raise HousingException(e,sys) from e

    def split_data_as_train_test(self):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            file_name = os.listdir(raw_data_dir)[0]
            housing_file_path = os.path.join(raw_data_dir,
                                file_name)
            logging.info(f'Reading csv file: [{housing_file_path}]')
            housing_df = pd.read_csv(housing_file_path)
            housing_df['income_cat'] = pd.cut(housing_df['median_income'],
                                       bins= [0.0,1.5,3,4.5,6.0,np.inf],
                                       labels= [1,2,3,4,5])
            logging.info(f"Splitting the dataset into train and test")
            strat_train_set = None
            strat_test_set = None
            split = StratifiedShuffleSplit(n_splits=1,
                    test_size=0.2, 
                    random_state=13)
            for train_index, test_index in split.split(housing_df,housing_df['income_cat']):
                strat_train_set = housing_df.loc[train_index].drop(['income_cat'], axis=1)
                strat_test_set = housing_df.loc[test_index].drop(['income_cat'], axis=1)
            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,
                              file_name)
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,
                             file_name)
            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,
                            exist_ok=True)
                logging.info(f"Exporting training dataset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path,
                                       index=False)
            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir,
                            exist_ok=True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path,
                                      index=False)
            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                                            test_file_path=test_file_path,
                                                            is_ingested=True,
                                                            message=f"Data Ingestion Completed sucessfully")
            logging.info(f'Data Ingestion Artifact: {data_ingestion_artifact}')
            return data_ingestion_artifact
        except Exception as e:
            raise HousingException(e,sys) from e

    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            tgz_file_path = self.download_housing_data()
            self.extract_tgz_file(tgz_file_path)
            return self.split_data_as_train_test()
        except Exception as e:
            raise HousingException(e,sys) from e

    def __del__(self):
        logging.info(f"{'='*20}Data Ingestion log Ended{'='*20} \n\n")