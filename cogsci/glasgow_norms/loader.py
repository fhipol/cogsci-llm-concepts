import pandas as pd
import spacy
from wordfreq import word_frequency


class GlasgowNormsLoader:
    filepath = '/content/drive/MyDrive/mistral/data/13428_2018_1099_MOESM2_ESM.csv'
    filepath = '/Users/fhipola/Documents/github/cogsci_thesis/cosgci/data/13428_2018_1099_MOESM2_ESM.csv'

    def __init__(self, filepath=filepath) -> None:
        # Load SpaCy English language model
        self.nlp = spacy.load("en_core_web_sm")
        self.df = None
        self.filepath

    def load_df_glasgow_norms(self, collapse_cols: bool = True) -> pd.DataFrame:
        # Define the file path
        # Read the CSV file

        df = pd.read_csv(self.filepath, header=[0, 1])
        sr_words = df.droplevel(1, axis=1)["Words"]
        df = df.set_index(sr_words).drop(["Words", "Length"], axis=1)

        first_level_cols_fixed = [last_named
                                  if 'Unnamed' in col
                                  else (last_named := col)
                                  for col in df.columns.get_level_values(0)
                                  ]

        df.columns = [first_level_cols_fixed,
                      df.columns.get_level_values(1)
                      ]

        if collapse_cols:
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        return df

    def get_pos(self, word) -> str:
        return self.nlp(word)[0].pos_

    def enrich_df(self, df) -> pd.DataFrame:
        df['category'] = [self.get_pos(word) for word in df.index]
        df['frequency'] = [word_frequency(word, 'en') for word in df.index]

        # mark polisemic words
        df['base_word'] = df.index.to_series().str.extract(r'([^()]+)')[
            0].str.strip()
        df['sense'] = df.index.to_series().str.extract(r'\((.*?)\)')
        base_word_counts = df['base_word'].value_counts()

        # df['is_polisemic'] = df['base_word'].map(lambda x: base_word_counts[x] > 1)
        df['is_polisemic'] = 0
        df.loc[df['sense'].notna() & (df['base_word'].map(
            base_word_counts) > 1), 'is_polisemic'] = 1
        df.loc[df['sense'].isna() & (df['base_word'].map(
            base_word_counts) > 1), 'is_polisemic'] = 2

        return df

    def load_df(self) -> pd.DataFrame:
        df = self.load_df_glasgow_norms()
        df = self.enrich_df(df)
        self.df = df.sort_values("frequency", ascending=False)

    @property
    def df_non_polisemic_nouns(self) -> pd.DataFrame:
        mask = (self.df.category == "NOUN") & ~(self.df.is_polisemic)
        return self.df[mask]

if __name__ == "__main__":
    glasgow_norms_loader = GlasgowNormsLoader()
    glasgow_norms_loader.load_df()
    glasgow_norms_loader.df.category.value_counts()