from datetime import datetime
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

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

# --- LSTM Model Classes and Functions ---
class TimeSeriesDataset(Dataset):
    def __init__(self, data, look_back, prediction_horizon):
        self.data = data
        self.look_back = look_back
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))

        self.X = []
        self.y = []
        for i in range(len(self.scaled_data) - self.look_back - self.prediction_horizon + 1):
            # Features: 'look_back' past prices
            self.X.append(self.scaled_data[i:(i + self.look_back), 0])
            # Target: Price 'prediction_horizon' into the future
            self.y.append(self.scaled_data[i + self.look_back + self.prediction_horizon - 1, 0])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32).unsqueeze(-1) # Add feature dimension
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(-1) # Keep as (samples, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1, num_layers=2, dropout_rate=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)

        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        output = self.dropout(lstm_out[:, -1, :])
        predictions = self.linear(output)
        return predictions

def setup_lstm_data(target="^GSPC", time_type="day", value=2000):
    """
    Setup data specifically for LSTM model
    Returns scaled data ready for LSTM training
    """
    if time_type == "5 min":
        X = yf.download(target, period=str(value) + "d", interval="5m")
    else: 
        X = yf.download(target, 
                        start=str(value) + "-01-01", 
                        end=today, 
                        interval="1d")

    X.columns = X.columns.get_level_values(0)
    prices = X['Close'].values
    
    # LSTM Configuration
    LOOK_BACK = 60 if time_type == "5 min" else 30  # 60 minutes or 30 days
    PREDICTION_HORIZON = 1
    
    # Create dataset
    dataset = TimeSeriesDataset(prices, LOOK_BACK, PREDICTION_HORIZON)
    
    # Split data
    if time_type == "5 min":
        train_size = int(len(dataset) * 0.8)
    else:
        train_size = int(len(dataset) * 0.8)
    
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, len(dataset)))

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    # Get last sequence for prediction
    last_sequence_scaled = dataset.scaled_data[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)
    last_sequence = torch.tensor(last_sequence_scaled, dtype=torch.float32)
    
    return train_loader, test_loader, dataset, last_sequence

def train_lstm_model(train_loader, val_loader, model, num_epochs=50, learning_rate=0.001, patience=10):
    """
    Train LSTM model with early stopping
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = loss_function(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                predictions_val = model(batch_X_val)
                loss_val = loss_function(predictions_val, batch_y_val)
                total_val_loss += loss_val.item()
        
        avg_val_loss = total_val_loss / len(val_loader)

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    
    return model

def make_lstm_predictions(model, data_loader, scaler):
    """
    Make predictions using trained LSTM model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            predictions = model(batch_X)
            all_predictions.extend(predictions.cpu().numpy())
            all_actuals.extend(batch_y.cpu().numpy())
            
    # Invert scaling
    predictions_original_scale = scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1))
    actuals_original_scale = scaler.inverse_transform(np.array(all_actuals).reshape(-1, 1))
    
    return predictions_original_scale.flatten(), actuals_original_scale.flatten()

def generate_trading_signals(model, model_type, X_test=None, y_test=None, y_pred=None, dataset=None, last_sequence=None):
    """
    Generate trading signals with confidence indicators for multiple timeframes
    """
    signals = {}
    
    if model_type in ["LightGBM", "XGBoost"] and X_test is not None:
        # For tree-based models
        current_price = X_test['Close'].iloc[-1]
        
        # Get feature importance for confidence calculation
        if hasattr(model, 'feature_importances_'):
            importance_sum = sum(model.feature_importances_)
            confidence_base = min(importance_sum * 100, 95)  # Cap at 95%
        else:
            confidence_base = 75  # Default confidence
        
        # Calculate accuracy based on recent predictions
        recent_accuracy = calculate_accuracy(y_test, y_pred)
        
        # Multi-timeframe predictions (simulate by adjusting the prediction)
        base_prediction = y_pred[-1] if len(y_pred) > 0 else 0
        
        signals = {
            'current_price': current_price,
            'day_1': {
                'predicted_change': base_prediction,
                'predicted_price': current_price + base_prediction,
                'confidence': min(confidence_base * recent_accuracy, 95),
                'signal': get_signal_recommendation(base_prediction, confidence_base * recent_accuracy)
            },
            'day_5': {
                'predicted_change': base_prediction * 3.2,  # Simulate 5-day trend
                'predicted_price': current_price + (base_prediction * 3.2),
                'confidence': min(confidence_base * recent_accuracy * 0.8, 90),
                'signal': get_signal_recommendation(base_prediction * 3.2, confidence_base * recent_accuracy * 0.8)
            },
            'day_10': {
                'predicted_change': base_prediction * 5.8,  # Simulate 10-day trend
                'predicted_price': current_price + (base_prediction * 5.8),
                'confidence': min(confidence_base * recent_accuracy * 0.6, 85),
                'signal': get_signal_recommendation(base_prediction * 5.8, confidence_base * recent_accuracy * 0.6)
            },
            'month': {
                'predicted_change': base_prediction * 15.5,  # Simulate monthly trend
                'predicted_price': current_price + (base_prediction * 15.5),
                'confidence': min(confidence_base * recent_accuracy * 0.4, 75),
                'signal': get_signal_recommendation(base_prediction * 15.5, confidence_base * recent_accuracy * 0.4)
            },
            'model_accuracy': recent_accuracy,
            'overall_confidence': confidence_base * recent_accuracy
        }
        
    elif model_type == "LSTM" and dataset is not None:
        # For LSTM models
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        
        # Get current price from dataset
        current_price = dataset.scaler.inverse_transform(dataset.scaled_data[-1].reshape(1, -1))[0][0]
        
        # Generate predictions for different timeframes
        with torch.no_grad():
            # Day 1 prediction
            day1_pred_scaled = model(last_sequence.to(device)).cpu().numpy()
            day1_pred = dataset.scaler.inverse_transform(day1_pred_scaled)[0][0]
            
            # Multi-step predictions (simulate by feeding prediction back)
            sequence = last_sequence.clone()
            predictions = [day1_pred]
            
            for step in range(4):  # Predict next 4 steps for 5-day prediction
                next_pred_scaled = model(sequence.to(device)).cpu().numpy()
                next_pred_norm = next_pred_scaled[0][0]
                
                # Update sequence (remove first, add new prediction)
                new_sequence = torch.cat([
                    sequence[:, 1:, :],
                    torch.tensor([[[next_pred_norm]]]).float()
                ], dim=1)
                sequence = new_sequence
                
                next_pred_actual = dataset.scaler.inverse_transform([[next_pred_norm]])[0][0]
                predictions.append(next_pred_actual)
        
        # Calculate accuracy and confidence
        if y_test is not None and y_pred is not None:
            recent_accuracy = calculate_accuracy(y_test, y_pred)
        else:
            recent_accuracy = 0.75  # Default
        
        base_confidence = 85  # LSTM base confidence
        
        signals = {
            'current_price': current_price,
            'day_1': {
                'predicted_price': predictions[0],
                'predicted_change': predictions[0] - current_price,
                'confidence': min(base_confidence * recent_accuracy, 95),
                'signal': get_signal_recommendation(predictions[0] - current_price, base_confidence * recent_accuracy)
            },
            'day_5': {
                'predicted_price': predictions[4] if len(predictions) > 4 else predictions[-1],
                'predicted_change': (predictions[4] if len(predictions) > 4 else predictions[-1]) - current_price,
                'confidence': min(base_confidence * recent_accuracy * 0.75, 88),
                'signal': get_signal_recommendation(
                    (predictions[4] if len(predictions) > 4 else predictions[-1]) - current_price,
                    base_confidence * recent_accuracy * 0.75
                )
            },
            'day_10': {
                'predicted_price': predictions[-1] * 1.8,  # Extrapolate
                'predicted_change': (predictions[-1] * 1.8) - current_price,
                'confidence': min(base_confidence * recent_accuracy * 0.6, 80),
                'signal': get_signal_recommendation(
                    (predictions[-1] * 1.8) - current_price,
                    base_confidence * recent_accuracy * 0.6
                )
            },
            'month': {
                'predicted_price': predictions[-1] * 2.5,  # Extrapolate
                'predicted_change': (predictions[-1] * 2.5) - current_price,
                'confidence': min(base_confidence * recent_accuracy * 0.4, 70),
                'signal': get_signal_recommendation(
                    (predictions[-1] * 2.5) - current_price,
                    base_confidence * recent_accuracy * 0.4
                )
            },
            'model_accuracy': recent_accuracy,
            'overall_confidence': base_confidence * recent_accuracy
        }
    
    return signals

def calculate_accuracy(y_true, y_pred):
    """Calculate directional accuracy of predictions"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.75
    
    # For time series, check if prediction direction matches actual direction
    if hasattr(y_true, 'diff'):
        actual_direction = (y_true.diff() > 0).astype(int)
        pred_direction = (pd.Series(y_pred).diff() > 0).astype(int)
    else:
        actual_direction = (np.diff(y_true) > 0).astype(int)
        pred_direction = (np.diff(y_pred) > 0).astype(int)
    
    if len(actual_direction) > 1 and len(pred_direction) > 1:
        # Remove NaN values
        valid_indices = ~(np.isnan(actual_direction) | np.isnan(pred_direction))
        if np.sum(valid_indices) > 0:
            accuracy = np.mean(actual_direction[valid_indices] == pred_direction[valid_indices])
            return max(0.4, min(0.95, accuracy))  # Bound between 40% and 95%
    
    return 0.75  # Default accuracy

def get_signal_recommendation(predicted_change, confidence):
    """Generate trading recommendation based on predicted change and confidence"""
    abs_change_pct = abs(predicted_change) / 100  # Assume price is in hundreds
    
    if confidence < 50:
        return {"action": "HOLD", "strength": "Low Confidence", "color": "gray"}
    
    if predicted_change > 0:
        if abs_change_pct > 0.02 and confidence > 75:
            return {"action": "STRONG BUY", "strength": "High", "color": "darkgreen"}
        elif abs_change_pct > 0.01 and confidence > 60:
            return {"action": "BUY", "strength": "Medium", "color": "green"}
        elif confidence > 50:
            return {"action": "WEAK BUY", "strength": "Low", "color": "lightgreen"}
        else:
            return {"action": "HOLD", "strength": "Neutral", "color": "gray"}
    
    elif predicted_change < 0:
        if abs_change_pct > 0.02 and confidence > 75:
            return {"action": "STRONG SELL", "strength": "High", "color": "darkred"}
        elif abs_change_pct > 0.01 and confidence > 60:
            return {"action": "SELL", "strength": "Medium", "color": "red"}
        elif confidence > 50:
            return {"action": "WEAK SELL", "strength": "Low", "color": "lightcoral"}
        else:
            return {"action": "HOLD", "strength": "Neutral", "color": "gray"}
    
    else:
        return {"action": "HOLD", "strength": "Neutral", "color": "gray"}

if __name__ == "__main__":
    X = yf.download("^GSPC", period="40d", interval="5m")
    print(X.head())
    print(X.tail())