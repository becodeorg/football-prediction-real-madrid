
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_predictions(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.6)

    ax.plot([y_test.min(), y_test.max()], 
            [y_test.min(), y_test.max()], 'r--', lw=2)
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Prediction vs Actual')
    return fig

def plot_real_vs_predicted_a(X_test, y_test, y_pred, 
                             yesterday=None, last_pred=None):
    
    # Handle LSTM case where X_test might be None
    if X_test is None or not hasattr(X_test, 'Close'):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(y_test[:100], label='Actual', alpha=0.7)
        ax.plot(y_pred[:100], label='Predicted', alpha=0.7)
        ax.set_title('LSTM Model: Actual vs Predicted (First 100 Points)')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        return fig

    # Original logic for tree-based models
    y_test_abs = X_test['Close'].iloc[0] + y_test.cumsum()
    y_pred_abs = X_test['Close'] + y_pred - y_test

    if last_pred is not None and len(last_pred) > 0:
        tomorrow = y_test.index[-1] + pd.Timedelta(days=1)
        tomorrow_price = y_test_abs.iloc[-1] + last_pred[0]
        
        y_pred_extended = pd.concat([
            pd.Series(y_pred_abs, index=X_test.index),
            pd.Series([tomorrow_price], index=[tomorrow])
        ])
    else:
        y_pred_extended = pd.Series(y_pred_abs, index=X_test.index)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=X_test.index, y=y_test_abs, label='Actual', ax=ax)
    sns.lineplot(x=y_pred_extended.index, y=y_pred_extended, label='Predicted', ax=ax)

    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Actual vs Predicted Values')
    ax.legend()
    return fig

def plot_real_vs_predicted_b(y_test, y_pred, 
                             yesterday=None, last_pred=None):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Handle case where y_test might not have index (LSTM case)
    if hasattr(y_test, 'index'):
        # Tree-based models with pandas index
        sns.lineplot(x=y_test.index, y=y_test, label='Actual', ax=ax)
        sns.lineplot(x=y_test.index, y=y_pred, label='Predicted', ax=ax)
        ax.set_xlabel('Date')
    else:
        # LSTM models with numpy arrays
        ax.plot(y_test, label='Actual', alpha=0.7)
        ax.plot(y_pred, label='Predicted', alpha=0.7)
        ax.set_xlabel('Time Steps')

    ax.set_ylabel('Value')
    ax.set_title('Actual vs Predicted Values')
    ax.legend()
    ax.grid(True)
    return fig

def plot_train_test_pred(X_train, X_test, y_train, y_test,  y_pred):
    
    y_test_abs = X_test['Close']
    y_pred_abs = X_test['Close'] + y_pred - y_test

    y_train_abs = X_train['Close'] + y_train

    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(
        y_train.index, y_train_abs,
        label='Train Actual',
        color='tab:blue',
        linewidth=2
    )
    ax.plot(
        y_test.index, y_test_abs,
        label='Test Actual',
        color='tab:red',
        linewidth=2
    )
    ax.plot(
        y_test.index, y_pred_abs,
        label='Test Predicted',
        color='tab:green',
        linestyle='--'
    )

    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Actual vs Predicted — Train & Test')
    ax.legend(loc='best')
    plt.tight_layout()
    return fig

