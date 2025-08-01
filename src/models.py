import lightgbm as lgb
from datetime import datetime
import yfinance as yf
import pandas as pd
from features import prepare_features

today = datetime.today().strftime('%Y-%m-%d')

def load_lightgbm(target="^GSPC", starting_year="2000", **kwargs):
    X = yf.download(target, 
                        start=starting_year + "-01-01", # first day of starting year
                        end=today, 
                        interval="1d")

    # Create target column BEFORE renaming columns
    X["target"] = X["Close"].shift(-1)
    
    # Split train/test : tout jusqu'à 2024 pour train, 2025 pour test
    train_end_date = '2024-12-31'

    X.columns = X.columns.get_level_values(0) 
    X = X.dropna()
    # X.columns = X.columns.str.replace(r'[^0-9A-Za-z_]', '_', regex=True)

    X_train = X.loc[:train_end_date].drop(['target'], axis=1)
    y_train = X.loc[:train_end_date, "target"]
    X_test = X.loc[train_end_date:].drop(['target'], axis=1)
    y_test = X.loc[train_end_date:, "target"]

    model = lgb.LGBMRegressor(**kwargs)
    model.fit(X_train, y_train)
    
    return model, X_train, y_train, X_test, y_test

#-------------------------------------------------------# 
#----------------- Train & Test Model ------------------#
#-------------------------------------------------------#

if __name__ == "__main__":
    params = {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "verbose": -1
    }
    model, X_train, y_train, X_test, y_test = load_lightgbm(**params)
    print("Model loaded and trained successfully.")
    print(model)

    sp500 = yf.download("^GSPC", start="2020-01-01", interval="1d")
    print(sp500.head())
    print(sp500.dtypes)

#-------------------------------------------------------# 
#-------------- Prediction & Evaluation ----------------#
#-------------------------------------------------------#

    # Prédictions
    y_pred = model.predict(X_test)

    # Métriques de régression
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)
    print("r2 Score:", r2)