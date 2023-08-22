import pandas as pd
import sys
import os

from ludwig.datasets.loaders.dataset_loader import DatasetLoader
from typing import Dict, List, Optional, Set, Tuple, Union
from sklearn.model_selection import train_test_split

class PRIMELoader(DatasetLoader):


    def load(self, split=False, kaggle_username=None, kaggle_key=None) -> pd.DataFrame:
        data_file = os.path.join(self.config.download_urls[0], self.config.train_filenames)
        self.df = self.load_files_to_dataframe([data_file])
        self.verify()
        return self.split(self.df)
        
    def download(self, kaggle_username=None, kaggle_key=None) -> List[str]:
        raise ValueError ("Unimplemented")

    def verify(self) -> None:
        if list(self.df.columns.values) is []:
          raise ValueError ("Verifcation failed")
        
    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        raise ValueError ("Unimplemented")

    def transform_files(self, file_paths: List[str]) -> List[str]:
        raise ValueError ("Unimplemented")

    def load_files_to_dataframe(self, file_paths: List[str], root_dir=None) -> pd.DataFrame:
        df_list = (pd.read_csv(file) for file in file_paths)
        df_concat  = pd.concat(df_list, ignore_index=True)
        return df_concat

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        raise ValueError ("Unimplemented")        

    def export(self, output_directory: str) -> None:
        raise ValueError ("Unimplemented")        

    def load_unprocessed_dataframe(self, file_paths: List[str]) -> pd.DataFrame:
        raise ValueError ("Unimplemented")        

    def save_processed(self, dataframe: pd.DataFrame) -> None:
        raise ValueError ("Unimplemented")        

    def load_transformed_dataset(self) -> pd.DataFrame:
        raise ValueError ("Unimplemented")        

    def split(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df, test_df = train_test_split(dataset, test_size=0.2)
        train_df, val_df = train_test_split(train_df, test_size=0.2)
        return train_df, val_df, test_df
