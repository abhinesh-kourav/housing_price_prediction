from collections import namedtuple


DataInjectionConfig = namedtuple('DataIngestionConfig',
                                ['dataset_download_url',
                                'tgz_download_dir',
                                'raw_data_dir',
                                'ingested_train_dir', 
                                'ingested_test_dir'])

DataValidationConfig = namedtuple('DataValidationConfig', ['schema_file_path'])

DataTransformationConfig = namedtuple('DataTransformationConfig',
                                    ['add_bedroom_per_room',
                                    'transformed_train_dir',
                                    'transform_test_dir',
                                    'preprocessed_object_file_path']) #pickle file path

ModelTrainConfig = namedtuple('ModelTrainConfig',
                            ['trained_model_file_path', #pickle file path
                            'base_accuracy'])

ModelEvaluationConfig = namedtuple('ModelEvaluationConfig',
                                ['model_evaluation_file_path', 'time_stamp'])

ModelPusherCOnfig = namedtuple('ModelPusherConfig',
                                ['export_dir_path'])

