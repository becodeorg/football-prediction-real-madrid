from datetime import datetime
import yfinance as yf

from features import create_features

today = datetime.today().strftime('%Y-%m-%d')

def setup_data(target="^GSPC", 
               time_type="day",
               value="2000"):
    
    if time_type == "5 min":
        X = yf.download(target, period=str(value) + "d", interval="5m")
    else: 
        X = yf.download(target, 
                        start=str(value) + "-01-01", 
                        end=today, 
                        interval="1d")

    # Create target column BEFORE renaming columns
    X["target"] = X["Close"].shift(-1) - X["Close"]
    # df['target'] = df['Close'].pct_change().shift(-1)
    # df['target'] = np.log(df['Close']).diff().shift(-1)

    X.columns = X.columns.get_level_values(0) 
    X = create_features(X)

    yesterday = X.iloc[[-1]].drop(['target'], axis=1)
    X = X[:-1]  # drop last row with NaN target

    # X.columns = X.columns.str.replace(r'[^0-9A-Za-z_]', '_', regex=True)

    if time_type == "5 min":
        # Use an 80/20 ratio split for intraday data
        train_size = int(len(X) * 0.8)
        X_train = X.iloc[:train_size].drop(['target'], axis=1)
        y_train = X.iloc[:train_size]['target']
        X_test = X.iloc[train_size:].drop(['target'], axis=1)
        y_test = X.iloc[train_size:]['target']
    else:
        # Use fixed date split for daily data
        train_end_date = '2024-12-31'
        X_train = X.loc[:train_end_date].drop(['target'], axis=1)
        y_train = X.loc[:train_end_date, 'target']
        X_test = X.loc[train_end_date:].drop(['target'], axis=1)
        y_test = X.loc[train_end_date:, 'target']

    return X_train, y_train, X_test, y_test, yesterday

if __name__ == "__main__":
    X = yf.download("^GSPC", period="40d", interval="5m")
    print(X.head())
    print(X.tail())