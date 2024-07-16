import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from cogsci.probing.data_processor import DataProocessor


def plot_predictions_probe(df_plot,
                           psy_dim,
                           n_layer,
                           y_lim_min=1,
                           y_lim_max=9
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

    # Plot actual values as a smooth line
    plt.plot(df_grouped['word'],
             df_grouped[f'{psy_dim}_mean'],
             label=f'{psy_dim}',
             color='darkgreen')

    # Shaded prediction std zone
    plt.fill_between(df_grouped['word'],
                     df_grouped[f'{psy_dim}_pred_mean'] - df_grouped[
                         f'{psy_dim}_pred_std'],
                     df_grouped[f'{psy_dim}_pred_mean'] + df_grouped[
                         f'{psy_dim}_pred_std'],
                     color='lightblue',
                     alpha=0.5,
                     label='IMAG_pred_std')

    # Scatter plot for prediction mean values
    plt.scatter(df_grouped['word'],
                df_grouped[f'{psy_dim}_pred_mean'],
                label=f'{psy_dim}_pred',
                color='darkblue',
                marker='o',
                s=2)

    plt.xlabel('Words')
    plt.ylabel('Values')
    plt.title(f'{psy_dim} vs {psy_dim}_pred for Layer {n_layer}')

    # Show every nth word on the x-axis (because they are thousands of words)
    n = 20
    plt.xticks(range(0, len(df_grouped['word']), n), df_grouped['word'][::n],
               rotation=90, fontsize=8)
    plt.gca().set_xticklabels(df_grouped['word'][::n], rotation=90, ha='center')

    plt.ylim(y_lim_min, y_lim_max)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_results_from_probes(data):
    df_results = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))

    for psy_dim in df_results['psy_dim'].unique():
        subset = df_results[df_results['psy_dim'] == psy_dim]
        plt.plot(subset['n_layer'],
                 subset['R2_val'],
                 marker='o',
                 label=f'psy_dim {psy_dim}')

    plt.xlabel('Number of Layers')
    plt.ylabel('R2 Score')
    plt.title('R2 Score vs Number of Layers by psy_dim')
    plt.legend(title='psy_dim')
    plt.grid(True)
    plt.show()


class ProbeExecutor:
    psy_dims = ['AROU_M',
                'VAL_M',
                'DOM_M',
                'CNC_M',
                'IMAG_M',
                'FAM_M',
                'AOA_M',
                'SIZE_M',
                'GEND_M'
                ]

    ML_MODELS = {
        "ridge": Ridge(alpha=1.0),
        "multi": MLPRegressor(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=200),
        "GradientBoostingRegressor": {"n_estimators": 100,
                                      "learning_rate": 0.1,
                                      "max_depth": 3}
    }

    def __init__(self,
                 df_acts_with_norms,
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
        pd.options.mode.chained_assignment = None

    def run_probe_for_psy_dim(self,
                              data_processor,
                              psy_dim: str,
                              n_layer: str
                              ):

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

        self.data.append({
            "n_layer": n_layer,
            "psy_dim": psy_dim,
            "R2_train": R2_train,
            "mse_train": mse_train,
            "R2_val": R2_val,
            "mse_val": mse_val,
            "ml_model": self.ml_model_key
        })

    def run(self):
        for n_layer in self.n_layers:
            data_processor = DataProocessor(df=self.df_acts_with_norms,
                                            n_layer=n_layer,
                                            n_components=self.n_components,
                                            )
            self.X = data_processor.get_X()

            for psy_dim in self.psy_dims:
                self.run_probe_for_psy_dim(data_processor, psy_dim, n_layer)
                plot_predictions_probe(df_plot=data_processor.df_layer,
                                       psy_dim=psy_dim,
                                       n_layer=n_layer,
                                       y_lim_min=-5,
                                       y_lim_max=5
                                       )

        plot_results_from_probes(self.data)
