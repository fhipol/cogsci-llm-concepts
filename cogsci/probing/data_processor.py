import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from cogsci.common import MISTRAL_COLS_ACTIVATIONS


class DataProocessor:

    cols_acts = MISTRAL_COLS_ACTIVATIONS

    def __init__(self,
                 df,
                 n_layer: int,
                 n_components=None,
                 standarization="standard",
                 ):

        self.n_layer = n_layer
        self.df_layer = df[df.n_layer == self.n_layer]
        self.standarization = standarization

        if n_components:
            do_all_components = n_components == "all"
            self.pca = PCA() if do_all_components else PCA(n_components)
            self.n_components = len(
                self.cols_acts) if do_all_components else n_components
        else:
            self.n_components = False

    def fit_transform_with_scaler(self, df_data, inplace=True) -> pd.DataFrame:
        """
    Scales the data wrapping the scaled data in a Dataframe with same shape than
    Original one
    """
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df_data)
        df = pd.DataFrame(scaled_array,
                          index=df_data.index,
                          columns=df_data.columns)

        # if inplace,substitute the original self.df_layer values by scaled vals
        if inplace:
            self.df_layer[df_data.columns] = df

        return df

    def get_X(self):

        df_X = self.df_layer.loc[:, self.cols_acts]

        if self.standarization:
            print("Running Standarization for X...")
            X = self.fit_transform_with_scaler(df_X)

        if self.n_components:
            # keep observations idx but cols aren't the same (useful 4 aligning)
            print("Runnning PCA...")
            X = pd.DataFrame(self.pca.fit_transform(X), index=X.index)
            explained_variance: np.ndarray = self.pca.explained_variance_ratio_
            print("Explained variance", explained_variance.sum())

        return X

    def get_y(self, col_name: str):

        df_y = self.df_layer[[col_name]]

        if self.standarization:
            print("Running Standarization for Y...")
            y = self.fit_transform_with_scaler(df_y)

        return y

    def plot_scree_plot(self):

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.pca.explained_variance_ratio_) + 1),
                 self.pca.explained_variance_ratio_, marker='o')
        plt.title('Scree Plot')
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance Ratio')
        plt.show()

    def plot_explained_variance(self):

        explained_variance = self.pca.explained_variance_ratio_

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(explained_variance) + 1), explained_variance,
                alpha=0.5, align='center')
        plt.ylabel('Explained Variance Ratio')
        plt.xlabel('Principal Components')
        plt.title('Explained Variance by Components')

        plt.tight_layout()
        plt.show()

    def plot_cumulative_variance(self):
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                 marker='o', linestyle='--')
        plt.ylabel('Cumulative Explained Variance')
        plt.xlabel('Principal Components')
        plt.title('Cumulative Explained Variance')

        plt.tight_layout()
        plt.show()

    def get_n_components_explaining_var(self, p_explained_var: float) -> int:
        """
    get how many components explain as much as p_explained_var ∈ [0, 1]
    """
        cumulative_explained_variance = np.cumsum(
            self.pca.explained_variance_ratio_)
        n_components = np.argmax(
            cumulative_explained_variance >= p_explained_var) + 1
        return n_components

    def get_explained_var_for_n_components(self, n_components: int) -> float:
        """
    return the explained variance by n_components
    """
        assert not n_components > self.n_components
        explained_variance = np.sum(
            self.pca.explained_variance_ratio_[:n_components])
        return explained_variance
