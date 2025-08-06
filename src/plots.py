
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

    y_pred = np.append(y_pred, last_pred)
    y_test = np.append(y_test, 0)

    X_test = pd.concat([X_test, yesterday])

    y_test_abs = X_test['Close']
    y_pred_abs = X_test['Close'] + y_pred - y_test

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(x=X_test.index, y=y_test_abs, label='Actual', ax=ax)
    sns.lineplot(x=X_test.index, y=y_pred_abs, label='Predicted', ax=ax)

    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Actual vs Predicted Values')
    ax.legend()
    return fig

def plot_real_vs_predicted_b(y_test, y_pred, 
                             yesterday=None, last_pred=None):

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=y_test.index, y=y_test, label='Actual', ax=ax)
    sns.lineplot(x=y_test.index, y=y_pred, label='Predicted', ax=ax)

    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Actual vs Predicted Values')
    ax.legend()
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

