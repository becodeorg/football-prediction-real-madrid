import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from datetime import datetime
import pathlib as path

import streamlit as st
import yfinance

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

from models import load_lightgbm # , load_xgboost, load_lstm
from plots import plot_predictions, plot_real_vs_predicted

#-------------------------------------------------------# 
#--------------- Global Config & Constants -------------#
#-------------------------------------------------------#

available_tickers = [
    # Stock Tickers
    "^gspc",  # S&P 500
    "^NDX",  # NASDAQ 100
    "^ftse",  # FTSE 100
    "^n225",  # Nikkei 225
    "^cac",   # CAC 40

## company: Tickers
    "^aapl",  # Apple Inc.
    "^msft",  # Microsoft Corporation
    "^googl", # Alphabet Inc. (Google)
    "^amzn",  # Amazon.com Inc.
    "^fb",    # Meta Platforms Inc. (Facebook)
    "^tsla",  # Tesla Inc.
    "^vow3",  # Volkswagen AG
    "^nvda",  # NVIDIA Corporation
    "^intc",  # Intel Corporation
    "^ko",    # The Coca-Cola Company
    "^mcd",   # McDonald's Corporation
    "^dis",   # The Walt Disney Company
    "^ba",    # The Boeing Company
    "^air",   # Airbus SE
    "^lvmh",  # LVMH Moët Hennessy Louis Vuitton SE
    "^ora",   # Orange S.A.
]

st.set_page_config(layout="wide")

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

# Load default model on first run
if not st.session_state.model_trained:
    with st.spinner("Loading default model..."):

        default_params = {
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
        # Load the default model
        model, X_train, y_train, X_test, y_test = load_lightgbm(**default_params)
        
        # Store in session state
        st.session_state.model = model
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.y_pred = model.predict(X_test)
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
            n_estimators = st.slider("n_estimators", min_value=50, max_value=1500, value=500, step=50)
            max_depth = st.slider("max_depth", min_value=1, max_value=20, value=5, step=1)
            learning_rate = st.slider("learning_rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)


            if model_type == "LightGBM":
                num_leaves = st.slider("num_leaves", min_value=10, max_value=300, value=31, step=5)
            if model_type == "XGBoost":
                gamma = st.slider("gamma", min_value=0.0, max_value=5.0, value=0.0, step=0.1)

    #-------------------------------------------------------# 
    #---------------- Tab 2: Advanced Config ---------------#
    #-------------------------------------------------------#

    with tab2:
        if model_type == "XGBoost" or model_type == "LightGBM":
            subsample = st.slider("subsample", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
            colsample_bytree = st.slider("colsample_bytree", min_value=0.3, max_value=1.0, value=0.8, step=0.05)
            reg_alpha = st.slider("reg_alpha (L1)", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
            reg_lambda = st.slider("reg_lambda (L2)", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

#-------------------------------------------------------# 
#--------------------- Training Model ------------------#
#-------------------------------------------------------#

    time_type = st.selectbox("Select Time Type:", ["1 day", "5 min"])
    if time_type == "5 min":
        starting_date = st.number_input("Number of Days: (max: 60)", value=40, min_value=1, max_value=60, step=1)
    else:
        starting_date = st.number_input("Starting Year:", value=2000, min_value=2000, max_value=2025, step=1)
    # starting_date = st.date_input("Starting Date:", value=pd.to_datetime("2020-01-01"))

    target = st.text_input("Target Ticker:", value="^GSPC")

    if st.button("Train Model"):
        if model_type in ["XGBoost", "LightGBM"]:
            params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
            }
            if model_type == "XGBoost":
                params["gamma"] = gamma
            if model_type == "LightGBM":
                params["num_leaves"] = num_leaves

        model, X_train, y_train, X_test, y_test = load_lightgbm(
            starting_date=starting_date, **params)
        
        st.session_state.model = model
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.y_pred = model.predict(X_test)
        st.session_state.model_trained = True
        st.success("Model trained successfully!")


    pass

#-------------------------------------------------------# 
#------------------------- Graphs ----------------------#
#-------------------------------------------------------#

st.title("PREDICTION DASHBOARD")

# if st.session_state.model_trained:
model = st.session_state.model
y_test = st.session_state.y_test
y_pred = st.session_state.y_pred

try:
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.write("Mean Absolute Error:", mae)
    st.write("Root Mean Squared Error:", rmse)
    st.write("r2 Score:", r2)
    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        fig = plot_predictions(y_test, y_pred)
        st.pyplot(fig)

    with col2:
        fig = plot_real_vs_predicted(y_test, y_pred)
        st.pyplot(fig)


except Exception as e:
    st.write("Error occurred while evaluating the model:", e)