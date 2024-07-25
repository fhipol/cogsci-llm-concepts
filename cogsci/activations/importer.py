import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import time
import pandas as pd
import pyarrow.parquet as pq
import os

from cogsci.common import MISTRAL_COLS_ACTIVATIONS


class ExperimentDataImporter:
    cols_activations = MISTRAL_COLS_ACTIVATIONS
    path_activations = "/content/drive/MyDrive/mistral/data/activations"

    def __init__(self,
                 layer_name: str,
                 n_experiment: int,
                 model_name="mistral"):
        self.df = None
        self.df_metadata = None
        self.layer_name = layer_name
        self.n_experiment = n_experiment
        self.model_name = model_name
        self.n_layers = [0, 3, 10, 17, 24, 31]
        self.n_max = 50
        self.path_parquets_per_word = f"{self.path_activations}/{self.model_name}/{self.n_experiment}/{self.layer_name}"
        self.temperature = 0

        df_dask = dd.read_parquet(self.path_parquets_per_word)
        mask = (df_dask['n_layer'].isin(self.n_layers)) & \
               (df_dask['n'] < self.n_max)

        self.df_dask = df_dask[mask].reset_index(drop=True)

    def export_df_as_gathered_data(self):
        path = f"{self.path_activations}/{self.model_name}/gathered/"
        filename = f"tmp={self.n_experiment}_layer={self.layer_name}_model={self.model_name}_t=0.parquet"
        full_path = os.path.join(path, filename)
        self.import_from_parquets()
        print("parquets imported into Memory, now write them to single parquet")
        self.df.to_parquet(full_path)
        print("done")

    def load_df_metadata(self) -> pd.DataFrame:
        path = self.path_parquets_per_word
        parquet_files = [f for f in os.listdir(path)
                         if f.endswith('.parquet')
                         and not "whore" in f
                         # < this file cannot be processed.
                         # my theory, GDrive restricts reading cause the slung
                         ]

        metadata_list = []

        total_files = len(parquet_files)

        for i, file in enumerate(parquet_files, 1):
            file_path = os.path.join(path, file)

            # TODO: if this takes more than one minute, retry it:
            metadata = pq.read_metadata(file_path)
            metadata_list.append(metadata)
            print(f'Processed {i}/{total_files} files: {file}')

        df = pd.DataFrame([
            {
                "template": metadata.metadata[b"template"],
                "output": metadata.metadata[b"output"],
                "word": metadata.metadata[b"word"],
                "prompt": metadata.metadata[b"prompt"]
            }
            for metadata in metadata_list
        ])

        df = df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)
        df["layer_name"] = self.layer_name
        df["experiment"] = self.n_experiment
        df["model_name"] = self.model_name

        self.df_metadata = df
        return df

    def import_df_from_gathered(self) -> pd.DataFrame:
        """
        It imports the data from the gathered parquet
        """
        path = f"{self.path_activations}/{self.model_name}/gathered/tmp={self.n_experiment}_layer={self.layer_name}_model={self.model_name}_t={self.temperature}.parquet"
        df = pd.read_parquet(path)
        return df

    def import_from_parquets(self) -> pd.DataFrame:
        start_time = time.time()
        self.df = self.df_dask.compute().reset_index(drop=True)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        return self.df
