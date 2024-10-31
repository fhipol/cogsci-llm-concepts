import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

from cogsci.probing.data_processor import DataProocessor

PSY_DIMS = ['AROU_M',
            'VAL_M',
            'DOM_M',
            'CNC_M',
            'IMAG_M',
            'FAM_M',
            'AOA_M',
            'SIZE_M',
            'GEND_M'
            ]

# used for consistent mapping between each psy_dim and series graph color
COLOR_MAP = {dim: plt.cm.get_cmap('tab10', len(PSY_DIMS))(i)
             for i, dim in enumerate(PSY_DIMS)
             }


def _plot_predictions_probe(df_plot,
                            psy_dim,
                            title,
                            y_lim_min=1,
                            y_lim_max=9,
                            ):
    # Group by 'word' & calculate the mean & std for 'IMAG_M' and 'IMAG_M_pred'
    group_by = 'word'

    df_grouped = df_plot.groupby(group_by).agg({
        f'{psy_dim}': ['mean', 'std'],
        f'{psy_dim}_pred': ['mean', 'std']
    }).reset_index()

    df_grouped.columns = ['word',
                          f'{psy_dim}_mean',
                          f'{psy_dim}_std',
                          f'{psy_dim}_pred_mean',
                          f'{psy_dim}_pred_std'
                          ]

    df_grouped = df_grouped.sort_values(by=f'{psy_dim}_mean')

    plt.figure(figsize=(12, 6))

    # Plot actual PSY_DIM mean values as a smooth line
    plt.plot(df_grouped['word'],
             df_grouped[f'{psy_dim}_mean'],
             label=f'{psy_dim}',
             color=COLOR_MAP[psy_dim],
             linewidth=2,
             )

    # Shaded prediction std zone
    plt.fill_between(df_grouped['word'],
                     df_grouped[f'{psy_dim}_pred_mean'] - df_grouped[
                         f'{psy_dim}_pred_std'],
                     df_grouped[f'{psy_dim}_pred_mean'] + df_grouped[
                         f'{psy_dim}_pred_std'],
                     color='silver',
                     alpha=0.5,
                     label=f'{psy_dim}_pred_std')

    # Scatter plot for prediction mean values for TRAIN AND VAL
    train_data = df_grouped[df_grouped['set'] == 'train']
    test_data = df_grouped[df_grouped['set'] == 'test']

    plt.scatter(train_data['word'],
                train_data[f'{psy_dim}_pred_mean'],
                label=f'{psy_dim}_pred (Train)',
                color='red',
                marker='o',
                s=20)

    plt.scatter(test_data['word'],
                test_data[f'{psy_dim}_pred_mean'],
                label=f'{psy_dim}_pred (Test)',
                color='green',
                marker='x',
                s=20)

    plt.xlabel('Words')
    plt.ylabel('Values')
    plt.title(title)

    # Show every nth word on the x-axis (because they are thousands of words)
    n = 20
    plt.xticks(range(0, len(df_grouped['word']), n), df_grouped['word'][::n],
               rotation=90, fontsize=8)
    plt.gca().set_xticklabels(df_grouped['word'][::n], rotation=90, ha='center')

    plt.ylim(y_lim_min, y_lim_max)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_predictions_probe(df_plot, psy_dim, title, y_lim_min=1, y_lim_max=9):
    # Group by 'word' and keep 'set' information
    df_grouped = df_plot.groupby(['word', 'set']).agg({
        f'{psy_dim}': ['mean', 'std'],
        f'{psy_dim}_pred': ['mean', 'std']
    }).reset_index()

    # Flatten the multi-level columns after grouping
    df_grouped.columns = ['word', 'set',
                          f'{psy_dim}_mean',
                          f'{psy_dim}_std',
                          f'{psy_dim}_pred_mean',
                          f'{psy_dim}_pred_std']

    # Sorting by the psychological dimension mean for better visualization
    df_grouped = df_grouped.sort_values(by=f'{psy_dim}_mean')

    plt.figure(figsize=(12, 6))

    # Plot actual PSY_DIM mean values as a smooth line
    plt.plot(df_grouped['word'],
             df_grouped[f'{psy_dim}_mean'],
             label=f'{psy_dim}',
             color='blue',  # Use a fixed color or COLOR_MAP if defined
             linewidth=2)

    # Shaded prediction std zone
    plt.fill_between(df_grouped['word'],
                     df_grouped[f'{psy_dim}_pred_mean'] - df_grouped[
                         f'{psy_dim}_pred_std'],
                     df_grouped[f'{psy_dim}_pred_mean'] + df_grouped[
                         f'{psy_dim}_pred_std'],
                     color='silver', alpha=0.5, label=f'{psy_dim}_pred_std')

    # Separate scatter plots for train and test sets
    train_data = df_grouped[df_grouped['set'] == 'train']
    test_data = df_grouped[df_grouped['set'] == 'test']

    plt.scatter(train_data['word'],
                train_data[f'{psy_dim}_pred_mean'],
                label=f'{psy_dim}_pred (Train)',
                color='red',
                marker='o',
                s=20)

    plt.scatter(test_data['word'],
                test_data[f'{psy_dim}_pred_mean'],
                label=f'{psy_dim}_pred (Test)',
                color='green',
                marker='x',
                s=20)

    plt.xlabel('Words')
    plt.ylabel('Values')
    plt.title(title)

    # Efficiently handling word labels on x-axis
    n = 20
    plt.xticks(range(0, len(df_grouped['word']), n), df_grouped['word'][::n],
               rotation=90, fontsize=8)
    plt.gca().set_xticklabels(df_grouped['word'][::n], rotation=90, ha='center')

    plt.ylim(y_lim_min, y_lim_max)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_results_from_probes(df_results,
                             n_exp: int,
                             y_lim: tuple = None,
                             show_legend: bool = False):
    plt.figure(figsize=(10, 6))

    for psy_dim in df_results['psy_dim'].unique():
        subset = df_results[df_results['psy_dim'] == psy_dim]
        plt.plot(subset['n_layer'],
                 subset['R2_val'],
                 marker='o',
                 label=f'psy_dim {psy_dim}',
                 color=COLOR_MAP[psy_dim]
                 )

    plt.xlabel('Number of Layers')
    plt.ylabel('R2 Score')
    plt.title(
        f'R2 Validation Score vs nth layer by psy_dim for experiment {n_exp}')

    if show_legend:
        plt.legend(title='psy_dim')

    plt.grid(True)

    if y_lim:
        plt.ylim(y_lim)

    plt.show()


class ProbeExecutor:
    ML_MODELS = {
        "ridge": Ridge(alpha=1.0),
        "multiperceptron": MLPRegressor(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=200),
        "gradient_regressor": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3)
    }

    def __init__(self,
                 df_acts_with_norms,
                 n_experiment: int,
                 ml_model_key: str,
                 n_components=None,
                 ):

        self.data = []
        self.df_acts_with_norms = df_acts_with_norms
        self.n_layers = df_acts_with_norms.n_layer.unique()
        self.ml_model_key = ml_model_key
        self.X = None

        # the number of components used in the pre-PCA, None if no PCA performed
        self.n_components = n_components
        self.n_experiment = n_experiment
        pd.options.mode.chained_assignment = None

    def run_probe_for_psy_dim(self,
                              data_processor: DataProocessor,
                              psy_dim: str,
                              n_layer: str
                              ) -> dict:

        y = data_processor.get_y(col_name=psy_dim)
        # y_scaled = (df_n_layer[psy_dim] - 1) / (9 - 1)

        # TODO: pass X in a more elegant way
        X_train, X_val, y_train, y_val = train_test_split(self.X,
                                                          y,
                                                          test_size=0.2,
                                                          random_state=11)

        # Label the training/test sets
        data_processor.df_layer.loc[X_train.index, 'set'] = 'train'
        data_processor.df_layer.loc[X_val.index, 'set'] = 'test'
        model = self.ML_MODELS[self.ml_model_key]
        model.fit(X_train, y_train)

        # predictions over the output variables
        # TODO: OPTIMIZE!!!!
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        data_processor.df_layer[f"{psy_dim}_pred"] = model.predict(self.X)

        # Calculate metrics for train data
        R2_train = r2_score(y_train, y_train_pred)
        R2_val = r2_score(y_val, y_val_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_val = mean_squared_error(y_val, y_val_pred)

        print(f"R-squared for Layer-{n_layer} & {psy_dim}: {R2_val}")
        print(f"Mean Squared Error: {mse_val}")
        print("---------------------")

        data_record = {
            "n_layer": n_layer,
            "psy_dim": psy_dim,
            "R2_train": R2_train,
            "mse_train": mse_train,
            "R2_val": R2_val,
            "mse_val": mse_val,
            "ml_model": self.ml_model_key
        }

        self.data.append(data_record)
        return data_record

    @property
    def df_results(self) -> pd.DataFrame:
        df = pd.DataFrame(self.data)
        df["n_experiment"] = self.n_experiment
        return df

    def run(self):

        for n_layer in self.n_layers:
            data_processor = DataProocessor(df=self.df_acts_with_norms,
                                            n_layer=n_layer,
                                            n_components=self.n_components,
                                            )
            self.X = data_processor.get_X()

            for psy_dim in PSY_DIMS:
                data_record = self.run_probe_for_psy_dim(data_processor,
                                                         psy_dim,
                                                         n_layer)

                title = f'Layer {n_layer} {psy_dim} Prediction - ' \
                        f'Train R²: {data_record["mse_train"]:.2f}, ' \
                        f'MSE: {data_record["R2_train"]:.2f} | ' \
                        f'Val R²: {data_record["mse_val"]:.2f}, ' \
                        f'MSE: {data_record["R2_val"]:.2f} ' \
                    # f'(Model: {self.ml_model_key})'

                plot_predictions_probe(df_plot=data_processor.df_layer,
                                       psy_dim=psy_dim,
                                       title=title,
                                       y_lim_min=-5,
                                       y_lim_max=5
                                       )

        plot_results_from_probes(self.df_results,
                                 n_exp=self.n_experiment,
                                 y_lim=[0, 1])
