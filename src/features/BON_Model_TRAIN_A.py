import sys
sys.path.append('/home/onyxia/work/cine-insights/src/features')
import   BUILD10_1 as Build10a1
from sklearn.preprocessing import MinMaxScaler
from time import time
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor
from time import time
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import make_scorer


def get_clean_df(raw_df):
    (x_train, y_train), (x_test, y_test), (x_val, y_val), ct = Build10a1.preprocessing_data(raw_df)
    return (x_train, y_train), (x_test, y_test), (x_val, y_val), ct


# define custom scorer
def rmsle(y_true, y_pred):
    # rmsle = np.sqrt(((np.log1p(y_true)-np.log1p(y_pred))**2).mean())
    rmsle = np.sqrt((y_true-y_pred)**2).mean()
    return rmsle


scorer = make_scorer(rmsle)


def mean_absolute_percentage_error(y_test, y_pred): 
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100


class XGBRegressorWrapper:
    def __init__(self, Raw_data):
        (X_train, y_train), (X_test, y_test), (_, _), ct = get_clean_df(Raw_data)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_params_ = None
        self.model = None
        self.ct_data_transformer = ct    # u attribut importnat lors de la prediction de nouvelle de données "
        self.train_time = None
        self.train_rmsle = None
        self.test_rmsle = None
        self.test_mape = None

        # Ignore all warnings
        warnings.filterwarnings('ignore')

        self._train_model()

    def _train_model(self):
        start = time()
        xgb = XGBRegressor()
        parameters = {'objective': ['reg:squarederror'],
                      'learning_rate': [0.03, 0.05, 0.07],
                      'max_depth': [5, 6, 7],
                      'min_child_weight': [3, 4],
                      'silent': [1],
                      'subsample': [0.7, 0.8, 0.9],
                      'colsample_bytree': [0.7],
                      'n_estimators': [500, 1000, 2800]}

        grid = GridSearchCV(xgb,
                            parameters,
                            cv=2,
                            n_jobs=-1,
                            verbose=True)

        grid.fit(self.X_train, self.y_train)
        self.best_params_ = grid.best_params_
        self.model = grid.best_estimator_
        self.train_time = np.round(time() - start, 4)
        self.train_rmsle = grid.best_score_

        # Calculate metrics on test set
        y_pred_test = self.model.predict(self.X_test)
        self.test_rmsle = rmsle(self.X_test, self.y_test)
        self.test_mape = mean_absolute_percentage_error(self.y_test, y_pred_test)

    def display_results(self):
        data = {
            "Best parameters": [self.best_params_],
            "Training RMSLE": [self.train_rmsle],
            "Test RMSLE": [self.test_rmsle],
            "Test MAPE": [self.test_mape],
            "Training time": [self.train_time]
        }
        return pd.DataFrame(data)

    def save_best_model(self, filename):
        import joblib
        joblib.dump(self.model, filename)
        return pd.DataFrame({"Message": ["Model saved successfully."]})
    def predict_(self, X):
        return self.model.predict(X)
    
