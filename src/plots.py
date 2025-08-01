
import matplotlib.pyplot as plt
import seaborn as sns


def plot_predictions(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.6)

    ax.plot([y_test.min(), y_test.max()], 
            [y_test.min(), y_test.max()], 'r--', lw=2)
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Prediction vs Actual')
    return fig

def plot_real_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=y_test.index, y=y_test, label='Actual', ax=ax)
    sns.lineplot(x=y_test.index, y=y_pred, label='Predicted', ax=ax)

    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Actual vs Predicted Values')
    ax.legend()
    return fig