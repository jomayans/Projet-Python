from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class Scaler_DEScaler:
    def __init__(self):
        self.scalers = {}
        self.min_ = {}
        self.max_ = {}
        self.cols = []
        self.num_cols = []

    def normalizer(self, unnormalized_df):
        df = unnormalized_df.copy()

        self.num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        for col in self.num_cols:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler
            self.min_[col] = scaler.data_min_
            self.max_[col] = scaler.data_max_

        self.cols = df.columns

        return df

    def DE_normalizer(self, normalized_df):
        DEnormalize_df = normalized_df[self.cols].copy()

        for col in self.num_cols:
            DEnormalize_df[col] = (self.max_[col] - self.min_[col]) * DEnormalize_df[col] + self.min_[col]

        return DEnormalize_df
