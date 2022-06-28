from tkinter import E
from housing.entity.config_entity import DataInjectionConfig
from housing.exception import HousingException
import sys, os
from housing.logger import logging



class DataIngestion:
    def __init__(self, data_ingestion_config: DataInjectionConfig) -> None:
        try:
            logging.info(f"{'='*20} Data Ingestion Log Started {'='*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise HousingException(e,sys) from e

    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            pass
        except Exception as e:
            raise HousingException(e,sys) from e