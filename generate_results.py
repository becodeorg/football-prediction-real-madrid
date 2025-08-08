#!/usr/bin/env python3
"""
Script to generate project results in CSV format with 3 columns.
This script will run the models and export predictions to a CSV file.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def create_basic_features(df):
    """Create basic technical features"""
    df = df.copy()
    
    # Price-based features
    df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
    df['Price_Change'] = df['Close'].pct_change()
    
    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    
    # RSI (simplified)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Volume features
    df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=10).std()
    
    return df

def setup_simple_data(target="^GSPC", start_year=2020):
    """Setup data similar to the original setup_data function"""
    today = datetime.today().strftime('%Y-%m-%d')
    
    # Download data
    print(f"Downloading {target} data from {start_year}...")
    X = yf.download(target, 
                   start=f"{start_year}-01-01", 
                   end=today, 
                   interval="1d")
    
    # Fix column names if multi-level
    if isinstance(X.columns, pd.MultiIndex):
        X.columns = X.columns.get_level_values(0)
    
    # Create target column BEFORE creating features
    X["target"] = X["Close"].shift(-1) - X["Close"]
    
    # Create features
    X = create_basic_features(X)
    
    # Get yesterday's data
    yesterday = X.iloc[[-1]].drop(['target'], axis=1)
    X = X[:-1]  # drop last row with NaN target
    
    # Remove NaN values
    X = X.dropna()
    
    # Split data
    train_end_date = '2024-12-31'
    X_train = X.loc[:train_end_date].drop(['target'], axis=1)
    y_train = X.loc[:train_end_date, 'target']
    X_test = X.loc[train_end_date:].drop(['target'], axis=1)
    y_test = X.loc[train_end_date:, 'target']
    
    return X_train, y_train, X_test, y_test, yesterday

def generate_project_results():
    """
    Generate project results CSV with 3 columns:
    - Date: The date for the prediction
    - Actual: The actual price/return
    - Predicted: The predicted price/return
    """
    
    print("Generating project results...")
    
    # Set up parameters
    target = "^GSPC"  # S&P 500
    start_year = 2020
    
    # Get data
    print("Loading and preparing data...")
    X_train, y_train, X_test, y_test, yesterday = setup_simple_data(
        target=target,
        start_year=start_year
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Train LightGBM model (best performing model)
    print("Training LightGBM model...")
    lgb_params = {
        "n_estimators": 1100,
        "max_depth": 14,
        "learning_rate": 0.016,
        "num_leaves": 31,
        "subsample": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0,
        "reg_lambda": 0.7,
        "verbose": -1
    }
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Create results dataframe
    print("Creating results dataframe...")
    
    # Get the dates from the test set index
    dates = X_test.index
    
    # Convert predictions and actual values to absolute prices
    # Since y_test and y_pred are price changes, we need to convert them back to actual prices
    initial_price = X_test['Close'].iloc[0]
    actual_prices = initial_price + y_test.cumsum()
    predicted_prices = X_test['Close'] + y_pred - y_test
    
    # Create the results dataframe
    results_df = pd.DataFrame({
        'Date': dates,
        'Actual': actual_prices,
        'Predicted': predicted_prices
    })
    
    # Reset index to make Date a regular column
    results_df = results_df.reset_index(drop=True)
    
    # Ensure Date column is properly formatted
    results_df['Date'] = pd.to_datetime(results_df['Date']).dt.strftime('%Y-%m-%d')
    
    # Round numerical columns to 2 decimal places
    results_df['Actual'] = results_df['Actual'].round(2)
    results_df['Predicted'] = results_df['Predicted'].round(2)
    
    # Show some statistics
    print(f"\nResults Summary:")
    print(f"Number of predictions: {len(results_df)}")
    print(f"Date range: {results_df['Date'].iloc[0]} to {results_df['Date'].iloc[-1]}")
    print(f"Actual price range: {results_df['Actual'].min():.2f} to {results_df['Actual'].max():.2f}")
    print(f"Predicted price range: {results_df['Predicted'].min():.2f} to {results_df['Predicted'].max():.2f}")
    
    # Calculate performance metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])
    rmse = np.sqrt(mean_squared_error(results_df['Actual'], results_df['Predicted']))
    r2 = r2_score(results_df['Actual'], results_df['Predicted'])
    
    print(f"\nModel Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Save to CSV
    output_file = "project_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Show first few rows
    print(f"\nFirst 10 rows of results:")
    print(results_df.head(10))
    
    return results_df

if __name__ == "__main__":
    # Generate the results
    results = generate_project_results()
    
    print("\n" + "="*50)
    print("PROJECT RESULTS GENERATED SUCCESSFULLY!")
    print("="*50)
    print(f"File saved as: project_results.csv")
    print(f"Columns: Date, Actual, Predicted")
    print(f"Total rows: {len(results)}")
