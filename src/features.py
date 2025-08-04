import pandas as pd
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import xgboost as xgb
import optuna 

def prepare_features(data):
    """Prépare les features avec shift() directement"""
    features = pd.DataFrame(index=data.index)
    
    # Prix OHLC du jour précédent
    features['Open_yesterday'] = data['Open'].shift(1)
    features['High_yesterday'] = data['High'].shift(1) 
    features['Low_yesterday'] = data['Low'].shift(1)
    features['Volume_yesterday'] = data['Volume'].shift(1)
    features['Close_yesterday'] = data['Close'].shift(1)

    # Returns du jour précédent
    features['Return_1d'] = data['Close'].pct_change().shift(1)
    features['Return_5d_'] = data['Close'].pct_change(5).shift(1)
    
    # Moyennes mobiles du jour précédent
    features['SMA_20'] = data['Close'].shift(1).rolling(20).mean()
    features['Volatility_20'] = data['Close'].pct_change().shift(1).rolling(20).std()

    return features

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "random_state": 42,
    }
    if model_type == "LightGBM":
        params["num_leaves"] = trial.suggest_int("num_leaves", 20, 200)
        model = lgb.LGBMRegressor(**params)
    elif model_type == "XGBoost":
        params["gamma"] = trial.suggest_float("gamma", 0.0, 5.0)
        model = xgb.XGBRegressor(**params)
    else:
        raise Exception("Not implemented")
    
    score = cross_val_score(model, 
                            X_train, 
                            y_train, 
                            cv=3, 
                            scoring="neg_root_mean_squared_error").mean()
    return -score

def optimize_model(X_train_data, y_train_data, model_type_name, n_trials=100):
    """Lance l'optimisation Optuna"""
    global X_train, y_train, model_type
    
    # Variables globales pour objective()
    X_train = X_train_data
    y_train = y_train_data
    model_type = model_type_name
    
    # Créer l'étude Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params, study.best_value