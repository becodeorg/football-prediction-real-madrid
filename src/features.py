import pandas as pd
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import xgboost as xgb
import optuna 
import numpy as np

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.volatility import BollingerBands

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "random_state": 42,
    }
    if model_type == "LightGBM":
        params["num_leaves"] = trial.suggest_int("num_leaves", 10, 500)
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

def create_features(df):
    # Daily returns (percentage change)
    df['Daily_Return'] = df['Close'].pct_change()

    # Price changes
    df['Price_Change'] = df['Close'] - df['Open']
    df['Price_Change_Pct'] = ((df['Close'] - df['Open']) / df['Open']) * 100

    # High-Low spread
    df['HL_Spread'] = df['High'] - df['Low']
    df['HL_Spread_Pct'] = (df['HL_Spread'] / df['Close']) * 100

    # 2. MOVING AVERAGES
    # print("📊 Creating moving averages...")

    # Simple Moving Averages
    df['SMA_5'] = SMAIndicator(close=df['Close'], window=5).sma_indicator()
    df['SMA_10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()

    # Exponential Moving Averages
    df['EMA_5'] = EMAIndicator(close=df['Close'], window=5).ema_indicator()
    df['EMA_10'] = EMAIndicator(close=df['Close'], window=10).ema_indicator()
    df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()

    # Moving Average Crossovers (important for trading signals)
    df['SMA_Cross_5_20'] = np.where(df['SMA_5'] > df['SMA_20'], 1, 0)
    df['EMA_Cross_5_20'] = np.where(df['EMA_5'] > df['EMA_20'], 1, 0)

    # 3. MOMENTUM INDICATORS
    # print("⚡ Creating momentum indicators...")

    # RSI (Relative Strength Index) - Measures overbought/oversold conditions
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()

    # MACD (Moving Average Convergence Divergence)
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()

    # 4. VOLATILITY INDICATORS
    # print("📈 Creating volatility indicators...")

    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    # Historical Volatility (rolling standard deviation of returns)
    df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std()
    df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()

    # 5. VOLUME INDICATORS
    # print("📊 Creating volume indicators...")

    # Volume moving averages
    df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()

    # Volume ratio
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

    # 6. LAG FEATURES (Previous day values)
    # print("⏰ Creating lag features...")

    # Previous day values (important for prediction)
    for lag in [1, 2, 3, 5]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)

    return df