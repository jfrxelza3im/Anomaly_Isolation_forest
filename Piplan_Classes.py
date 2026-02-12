from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.columns_to_drop)

#%%
class Drop_na(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X=X.copy()
        return X.dropna()
#%%
class Melt_data(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.melt(
            id_vars=["utc_timestamp"],
            value_vars= ["machine_1","machine_2","machine_3","machine_4","machine_5"],
            var_name="machine_col",
            value_name="power",
        )
#%%
class Sort(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.sort_values("utc_timestamp").reset_index(drop=True)
#%%
class Sort_For_Machine(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.sort_values(["machine_id", "utc_timestamp"]).reset_index(drop=True)
#%%
class Extract_machine_id(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["machine_id"] = X["machine_col"].str.replace("machine_", "").astype(int)
        # keep a copy so we can preserve the numeric id after one-hot encoding
        return X
#%%
class Calculate_power_diff(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["power_diff"] = X.groupby("machine_id")["power"].diff()/0.25
        X["power_diff"] = X["power_diff"].fillna(0)
        return X
#%%
class Parse_data(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['utc_timestamp'] = pd.to_datetime(X['utc_timestamp'], errors='coerce', utc=True)
        return X


if __name__ == "__main__":
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    # Define your columns
    cols_to_drop = ["cet_cest_timestamp", "area_offices", "area_room_1",
                    "area_room_2", "area_room_3", "area_room_4", "compressor",
                    "cooling_aggregate", "cooling_pumps", "dishwasher", "ev",
                    "grid_import", "pv_facade", "pv_roof", "refrigerator", "ventilation"]

    # Build the pipeline
    First_piplan = Pipeline([
        ('drop_cols', ColumnDropper(cols_to_drop)),
        ('parse', Parse_data()),
        ('melt', Melt_data()),
        ('drop_na', Drop_na()),
        ('extract_id', Extract_machine_id()),
        ('sort_machine', Sort_For_Machine()),
        ('calc_diff', Calculate_power_diff()),
        ('sort_final', Sort())
    ])
    ct = ColumnTransformer(
        transformers=[
            ('passthrough', 'passthrough', ['utc_timestamp', 'power_diff']),
            ('machine', OneHotEncoder(
                handle_unknown='ignore', sparse_output=False  # Dense for sanity
            ), ['machine_id'])
        ],
        remainder='drop'
    )

    Final_piplan = Pipeline([
        ("first_pipeline", First_piplan),
        ("column_transformer", ct),
    ])
    joblib.dump(Final_piplan, "preprocessing_pipeline_new2.joblib")
    print("Pipeline saved successfully with clean module references!")