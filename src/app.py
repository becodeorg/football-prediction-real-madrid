import pandas as pd
import numpy as np

import streamlit as st

import lightgbm as lgb
import xgboost as xgb

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

from models import setup_data #, load_lstm
from plots import plot_real_vs_predicted_a
from plots import plot_real_vs_predicted_b
from plots import plot_train_test_pred
from features import optimize_model

#-------------------------------------------------------# 
#--------------- Global Config & Constants -------------#
#-------------------------------------------------------#

available_tickers = [
    "^gspc",  # S&P 500
    "^NDX",  # NASDAQ 100
    "^ftse",  # FTSE 100
    "^n225",  # Nikkei 225
    ]

st.set_page_config(layout="wide")
params = {}

#-------------------------------------------------------# 
#------------ Model & Config by default ----------------#
#-------------------------------------------------------#

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model = "LightGBM"
    st.session_state.ticker = "^gspc"
    st.session_state.X_train = None
    st.session_state.y_train = None
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.y_pred = None
    st.session_state.new_model = False
    st.session_state.optimized = False

# Load default model on first run
if not st.session_state.model_trained:
    with st.spinner("Loading default model..."):

        default_params = {
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
        # Load the default model
        X_train, y_train, X_test, y_test, yesterday = setup_data(
            target=st.session_state.ticker, 
            time_type="day", 
            value=2000)
        
        model = lgb.LGBMRegressor(**default_params)
        model.fit(X_train, y_train)
        
        # Store in session state
        st.session_state.model = model
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.y_pred = model.predict(X_test)
        st.session_state.yesterday = yesterday
        st.session_state.last_pred = model.predict(yesterday)
        st.session_state.model_trained = True

#-------------------------------------------------------# 
#--------------- Sidebar & Model Config ----------------#
#-------------------------------------------------------#

with st.sidebar:
    model_type = st.selectbox("Select Model Type:", ["LightGBM", "XGBoost", "LSTM"])

    tab1, tab2, tab3 = st.tabs(["Basic", "Advanced", "Graphs"])

    #-------------------------------------------------------# 
    #------------------ Tab 1: Basic Config ----------------#
    #-------------------------------------------------------#

    with tab1:
        if model_type == "XGBoost" or model_type == "LightGBM":
            params["n_estimators"] = st.slider(
                "n_estimators", min_value=50, max_value=3000, value=1000, step=50)
            params["max_depth"] = st.slider(
                "max_depth", min_value=1, max_value=20, value=15, step=1)
            params["learning_rate"] = st.slider(
                "learning_rate", min_value=0.001, max_value=0.5, value=0.015, step=0.001)


            if model_type == "LightGBM":
                params["num_leaves"] = st.slider(
                    "num_leaves", min_value=10, max_value=300, value=250, step=5)
            if model_type == "XGBoost":
                params["gamma"] = st.slider(
                    "gamma", min_value=0.0, max_value=5.0, value=0.0, step=0.1)

    #-------------------------------------------------------# 
    #---------------- Tab 2: Advanced Config ---------------#
    #-------------------------------------------------------#

    with tab2:
        if model_type == "XGBoost" or model_type == "LightGBM":
            params["subsample"] = st.slider(
                "subsample", min_value=0.5, max_value=1.0, value=1.0, step=0.05)
            params["colsample_bytree"] = st.slider(
                "colsample_bytree", min_value=0.3, max_value=1.0, value=0.7, step=0.05)
            params["reg_alpha"] = st.slider(
                "reg_alpha (L1)", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
            params["reg_lambda"] = st.slider(
                "reg_lambda (L2)", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

    #-------------------------------------------------------# 
    #--------------------- Training Model ------------------#
    #-------------------------------------------------------#

    time_type = st.selectbox("Select Time Type:", ["1 day", "5 min"])
    if time_type == "5 min":
        time_value = st.number_input("Number of Days: (max: 60)", value=40, min_value=1, max_value=60, step=1)
    else:
        time_value = st.number_input("Starting Year:", value=2000, min_value=1933, max_value=2025, step=1)
    # time_value = st.date_input("Starting Date:", value=pd.to_datetime("2020-01-01"))

    target = st.selectbox("Target Ticker:", options=available_tickers)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Train Model"):
            if model_type in ["LightGBM", "XGBoost"]:
                X_train, y_train, X_test, y_test, yesterday = setup_data(
                    target=target,
                    time_type=time_type,
                    value=time_value)

                if model_type == "LightGBM":
                    model = lgb.LGBMRegressor(**params)
                elif model_type == "XGBoost":
                    model = xgb.XGBRegressor(**params)
            else:
                st.error("LSTM model is not implemented yet.")

            st.session_state.model = model
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            model.fit(X_train, y_train)
            st.session_state.y_pred = model.predict(X_test)
            st.session_state.last_pred = model.predict(yesterday)
            st.session_state.model_trained = True
            st.session_state.new_model = True
            st.session_state.optimized = False

    if st.session_state.new_model and not st.session_state.optimized:
        st.success("Model trained successfully!")

    with col2:
        if st.button("Optimize Model"):
            if model_type in ["LightGBM", "XGBoost"]:
                st.session_state.optimized = True
                st.session_state.new_model = False
                X_train, y_train, X_test, y_test, yesterday = setup_data(
                    target=target,
                    time_type=time_type,
                    value=time_value)
            else:
                st.error("LSTM model is not implemented yet.")

               
            with st.spinner("Searching..."):

                best_params, best_score = optimize_model(X_train, 
                                                            y_train, 
                                                            "LightGBM", 
                                                            n_trials=50)
                st.session_state.best_params = best_params
                
                if model_type == "LightGBM":
                    model = lgb.LGBMRegressor(**best_params)
                elif model_type == "XGBoost":
                    model = xgb.XGBRegressor(**best_params)

                model.fit(X_train, y_train)
                st.session_state.model = model
                st.session_state.y_pred = model.predict(X_test)
                st.session_state.last_pred = model.predict(yesterday)
                st.session_state.model_trained = True

    if st.session_state.optimized: 
        st.success("Optimized model trained!")

#-------------------------------------------------------# 
#------------------------- Graphs ----------------------#
#-------------------------------------------------------#

st.title("PREDICTION DASHBOARD")

# if st.session_state.model_trained:
model = st.session_state.model
X_train = st.session_state.X_train
X_test = st.session_state.X_test
y_train = st.session_state.y_train
y_test = st.session_state.y_test
y_pred = st.session_state.y_pred

try:
    col1, col2 = st.columns(2)
    with col1:
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        st.write("Mean Absolute Error:", mae)
        st.write("Root Mean Squared Error:", rmse)
        st.write("r2 Score:", r2)
        st.write("---")

    with col2:
        yesterday = st.session_state.yesterday
        last_pred = st.session_state.last_pred
        latest_real_price = yesterday["Close"].iloc[-1]
        latest_predicted_price = latest_real_price + last_pred[-1]
        st.write("Latest Real Price:", latest_real_price)
        st.write("Tomorrow's Prediction:", latest_predicted_price)
        st.write("Tomorrow's Difference:", latest_predicted_price - latest_real_price)
        st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        last_pred = st.session_state.last_pred
        yesterday = st.session_state.yesterday

        fig = plot_real_vs_predicted_a(X_test, y_test, y_pred,
                                       yesterday, last_pred)
        st.pyplot(fig)

    with col2:
        fig = plot_real_vs_predicted_b(y_test, y_pred,
                                       yesterday, last_pred)
        st.pyplot(fig)

    fig = plot_train_test_pred(X_train, X_test, 
                               y_train, y_test, y_pred)
    st.pyplot(fig)

except Exception as e:
    st.write("Error occurred while evaluating the model:", e)