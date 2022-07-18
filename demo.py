import sys, os
from housing.config.configuration import Configuration
from housing.exception import HousingException
from housing.pipeline.pipeline import Pipeline
from housing.logger import logging


def main():
    try:
        config_path = os.path.join("config","config.yaml")
        pipeline = Pipeline(Configuration(config_file_path=config_path))
        #pipeline = pipeline.run_pipeline()
        pipeline.start()
        logging.info("main function execution completed.")
    except Exception as e:
        logging.error(f"{e}")
        print(e)


if __name__ == "__main__":
    main()