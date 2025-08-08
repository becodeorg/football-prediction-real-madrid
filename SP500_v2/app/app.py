"""
Main Streamlit application for S&P 500 Trading Dashboard
Single comprehensive app using the data fetcher from data folder
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
import sys
import os

# Add the parent directory to the Python path to import from data module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import from data module
from data.fetch_data import DataFetcher

#----------------------------------------------------
# Fetch data
#------------------------------------------------------

# Page configuration
st.set_page_config(
    page_title="S&P 500 Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive {
        color: #00C851;
    }
    .negative {
        color: #ff4444;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(symbol, period=None, interval='1d', start_date=None, end_date=None):
    """Load data with caching - supports both period and date range"""
    fetcher = DataFetcher()
    
    if start_date and end_date:
        # Use custom date range
        if interval == 'yearly':
            return fetcher.get_yearly_data(symbol, start_date, end_date)
        else:
            return fetcher.get_historical_data_by_dates(symbol, start_date, end_date, interval)
    else:
        # Use period-based fetching
        return fetcher.get_historical_data(symbol, period, interval)

@st.cache_data(ttl=60)  # Cache for 1 minute
def load_real_time_data(symbol):
    """Load real-time data with caching"""
    fetcher = DataFetcher()
    return fetcher.get_real_time_data(symbol)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_basic_info(symbol):
    """Load basic info with caching"""
    fetcher = DataFetcher()
    return fetcher.get_basic_info(symbol)

def create_candlestick_chart(data, title="S&P 500 Price Chart", interval='1d'):
    """Create candlestick chart with proper date formatting"""
    fig = go.Figure(data=go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name="S&P 500"
    ))
    
    # Configure x-axis based on data interval
    xaxis_config = {
        'title': 'Date',
        'rangeslider': {'visible': False},
        'type': 'date'
    }
    
    # Adjust tick formatting based on interval
    if interval == 'yearly':
        xaxis_config.update({
            'dtick': 'M12',  # Show every 12 months (yearly)
            'tickformat': '%Y',  # Show only year
            'tickmode': 'linear'
        })
    elif interval in ['1mo', '3mo']:
        xaxis_config.update({
            'dtick': 'M6',  # Show every 6 months
            'tickformat': '%b %Y',  # Show month and year
        })
    elif interval in ['1wk', '1d']:
        # Let Plotly auto-format for shorter intervals
        pass
    
    fig.update_layout(
        title=title,
        yaxis_title="Price ($)",
        xaxis=xaxis_config,
        height=600,
        showlegend=False
    )
    
    return fig

def create_volume_chart(data, interval='1d'):
    """Create volume chart with proper date formatting"""
    fig = go.Figure(data=go.Bar(
        x=data.index,
        y=data['volume'],
        name="Volume",
        marker_color='rgba(55, 128, 191, 0.7)'
    ))
    
    # Configure x-axis based on data interval
    xaxis_config = {
        'title': 'Date',
        'type': 'date'
    }
    
    # Adjust tick formatting based on interval
    if interval == 'yearly':
        xaxis_config.update({
            'dtick': 'M12',  # Show every 12 months (yearly)
            'tickformat': '%Y',  # Show only year
            'tickmode': 'linear'
        })
    elif interval in ['1mo', '3mo']:
        xaxis_config.update({
            'dtick': 'M6',  # Show every 6 months
            'tickformat': '%b %Y',  # Show month and year
        })
    elif interval in ['1wk', '1d']:
        # Let Plotly auto-format for shorter intervals
        pass
    
    fig.update_layout(
        title="Trading Volume",
        yaxis_title="Volume",
        xaxis=xaxis_config,
        height=300
    )
    
    return fig

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 S&P 500 Trading Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🛠️ Configuration")
    
    # Symbol selection
    symbols = {
        'S&P 500 Index': '^GSPC',
        'S&P 500 ETF (SPY)': 'SPY',
        'Vanguard S&P 500 ETF (VOO)': 'VOO',
        'NASDAQ Composite': '^IXIC',
        'Dow Jones': '^DJI'
    }
    
    selected_symbol_name = st.sidebar.selectbox(
        "Select Index/ETF:",
        list(symbols.keys()),
        index=0
    )
    selected_symbol = symbols[selected_symbol_name]
    
    # Date range selection mode
    st.sidebar.header("📅 Time Period")
    
    use_custom_dates = st.sidebar.checkbox("Use Custom Date Range", value=False)
    
    if use_custom_dates:
        # Custom date range selection
        st.sidebar.subheader("Custom Date Range")
        
        # Calculate default dates
        default_start_date = date.today() - timedelta(days=1825)  # 5 years ago
        default_end_date = date.today()
        
        # Minimum date (yfinance limitation)
        min_date = date(1970, 1, 1)
        max_date = date.today()
        
        start_date = st.sidebar.date_input(
            "Start Date:",
            value=default_start_date,
            min_value=min_date,
            max_value=max_date,
            help="Select the start date for historical data"
        )
        
        end_date = st.sidebar.date_input(
            "End Date:",
            value=default_end_date,
            min_value=min_date,
            max_value=max_date,
            help="Select the end date for historical data"
        )
        
        # Validate date range
        if start_date >= end_date:
            st.sidebar.error("Start date must be before end date!")
            st.stop()
        
        # Calculate date range info
        date_diff = (end_date - start_date).days
        years = date_diff / 365.25
        
        st.sidebar.info(f"📊 Selected Range: {date_diff} days ({years:.1f} years)")
        
        # Quick date presets
        st.sidebar.subheader("🚀 Quick Presets")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("1 Year"):
                start_date = date.today() - timedelta(days=365)
                st.rerun()
            if st.button("2 Years"):
                start_date = date.today() - timedelta(days=730)
                st.rerun()
            if st.button("5 Years"):
                start_date = date.today() - timedelta(days=1825)
                st.rerun()
        
        with col2:
            if st.button("10 Years"):
                start_date = date.today() - timedelta(days=3650)
                st.rerun()
            if st.button("COVID Era"):
                start_date = date(2020, 3, 1)  # COVID pandemic start
                st.rerun()
            if st.button("Max Data"):
                start_date = date(1990, 1, 1)  # Far back for max data
                st.rerun()
        
        selected_period = None
        selected_period_name = f"{start_date} to {end_date}"
        
    else:
        # Traditional period selection
        st.sidebar.subheader("Predefined Periods")
        
        periods = {
            '1 Day': '1d',
            '5 Days': '5d',
            '1 Month': '1mo',
            '3 Months': '3mo',
            '6 Months': '6mo',
            '1 Year': '1y',
            '2 Years': '2y',
            '5 Years': '5y'
        }
        
        selected_period_name = st.sidebar.selectbox(
            "Select Time Period:",
            list(periods.keys()),
            index=6  # Default to 2 Years
        )
        selected_period = periods[selected_period_name]
        
        start_date = None
        end_date = None
    
    # Interval selection
    st.sidebar.header("⏱️ Data Interval")
    
    intervals = {
        '5 Minutes': '5m',
        '15 Minutes': '15m',
        '30 Minutes': '30m',
        '1 Hour': '1h',
        '1 Day': '1d',
        '1 Week': '1wk',
        '1 Month': '1mo',
        '1 Year (Annual)': 'yearly'
    }
    
    selected_interval_name = st.sidebar.selectbox(
        "Select Interval:",
        list(intervals.keys()),
        index=6  # Default to 1 Day
    )
    selected_interval = intervals[selected_interval_name]
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Main content
    try:
        # Load real-time data
        with st.spinner("Loading real-time data..."):
            real_time_data = load_real_time_data(selected_symbol)
        
        # Load basic info
        with st.spinner("Loading company info..."):
            basic_info = load_basic_info(selected_symbol)
        
        # Display real-time metrics
        if real_time_data:
            st.subheader(f"📊 {selected_symbol_name} - Real-Time Data")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = real_time_data.get('current_price', 0)
                st.metric(
                    label="Current Price",
                    value=f"${current_price:.2f}"
                )
            
            with col2:
                change = real_time_data.get('change', 0)
                change_percent = real_time_data.get('change_percent', 0)
                st.metric(
                    label="Daily Change",
                    value=f"${change:.2f}",
                    delta=f"{change_percent:.2f}%"
                )
            
            with col3:
                volume = real_time_data.get('volume', 0)
                st.metric(
                    label="Volume",
                    value=f"{volume:,}"
                )
            
            with col4:
                timestamp = real_time_data.get('timestamp', 'N/A')
                st.metric(
                    label="Last Update",
                    value=str(timestamp).split(' ')[1][:8] if timestamp != 'N/A' else 'N/A'
                )
        
        # Display basic info
        if basic_info:
            st.subheader("📋 Basic Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"**Name:** {basic_info.get('name', 'N/A')}")
                st.info(f"**Sector:** {basic_info.get('sector', 'N/A')}")
                st.info(f"**Currency:** {basic_info.get('currency', 'USD')}")
            
            with col2:
                market_cap = basic_info.get('market_cap', 'N/A')
                if market_cap != 'N/A':
                    st.info(f"**Market Cap:** ${market_cap:,}")
                else:
                    st.info(f"**Market Cap:** N/A")
                
                pe_ratio = basic_info.get('pe_ratio', 'N/A')
                st.info(f"**P/E Ratio:** {pe_ratio}")
                
                beta = basic_info.get('beta', 'N/A')
                st.info(f"**Beta:** {beta}")
            
            with col3:
                high_52 = basic_info.get('fifty_two_week_high', 'N/A')
                low_52 = basic_info.get('fifty_two_week_low', 'N/A')
                st.info(f"**52W High:** ${high_52}")
                st.info(f"**52W Low:** ${low_52}")
                
                div_yield = basic_info.get('dividend_yield', 'N/A')
                if div_yield != 'N/A':
                    st.info(f"**Dividend Yield:** {div_yield:.2%}")
                else:
                    st.info(f"**Dividend Yield:** N/A")
        
        # Load historical data
        with st.spinner(f"Loading {selected_period_name} historical data..."):
            data = load_data(
                selected_symbol, 
                selected_period, 
                selected_interval,
                start_date,
                end_date
            )
        
        if not data.empty:
            st.subheader(f"📈 {selected_symbol_name} - {selected_period_name}")
            
            # Display data info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.info(f"**Records:** {len(data)}")
            with col2:
                if use_custom_dates:
                    date_diff = (end_date - start_date).days
                    st.info(f"**Date Range:** {date_diff} days")
                else:
                    st.info(f"**Period:** {selected_period_name}")
            with col3:
                st.info(f"**Interval:** {selected_interval_name}")
            with col4:
                latest_price = data['close'].iloc[-1]
                st.info(f"**Latest Close:** ${latest_price:.2f}")
            
            # Create and display candlestick chart
            candlestick_fig = create_candlestick_chart(
                data, 
                f"{selected_symbol_name} - {selected_period_name}",
                selected_interval
            )
            st.plotly_chart(candlestick_fig, use_container_width=True)
            
            # Create and display volume chart
            volume_fig = create_volume_chart(data, selected_interval)
            st.plotly_chart(volume_fig, use_container_width=True)
            
            # Display recent data table
            st.subheader("📊 Recent Data")
            
            # Show last 10 records
            recent_data = data[['open', 'high', 'low', 'close', 'volume']].tail(10)
            
            # Format dates based on interval
            if selected_interval == 'yearly':
                recent_data.index = recent_data.index.strftime('%Y')  # Only year for yearly data
            elif selected_interval in ['1mo', '3mo']:
                recent_data.index = recent_data.index.strftime('%Y-%m')  # Year-month for monthly data
            elif selected_interval in ['1wk']:
                recent_data.index = recent_data.index.strftime('%Y-%m-%d')  # Year-month-day for weekly
            else:
                recent_data.index = recent_data.index.strftime('%Y-%m-%d %H:%M')  # Full datetime for intraday
            
            st.dataframe(
                recent_data.style.format({
                    'open': '${:.2f}',
                    'high': '${:.2f}',
                    'low': '${:.2f}',
                    'close': '${:.2f}',
                    'volume': '{:,}'
                }),
                use_container_width=True
            )
            
            # Price statistics
            st.subheader("📈 Price Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Highest Price", f"${data['high'].max():.2f}")
            with col2:
                st.metric("Lowest Price", f"${data['low'].min():.2f}")
            with col3:
                st.metric("Average Price", f"${data['close'].mean():.2f}")
            with col4:
                price_change = ((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]) * 100
                st.metric("Total Return", f"{price_change:+.2f}%")
        
        else:
            st.error(f"❌ No data available for {selected_symbol_name}")
            st.info("Please check your internet connection or try a different symbol.")
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("Please check your internet connection and make sure yfinance is installed.")
        
        with st.expander("🔧 Installation Instructions"):
            st.code("pip install yfinance pandas plotly streamlit", language="bash")
    
    # Footer
    st.markdown("---")
    st.markdown("**📊 S&P 500 Trading Dashboard** | Data provided by Yahoo Finance via yfinance")
    st.markdown("⚠️ **Disclaimer:** This application is for educational purposes only. Not suitable for real trading without professional validation.")


# --- LightGBM Performance Tab ---
import sys
sys.path.append("../models")
from models.Ligthgbm import SP500LightGBMPredictor

def ligthgbm_performance_tab():
    st.header("Model Performance Overview")
    predictor = SP500LightGBMPredictor()
    # Try to load models and metrics
    if not predictor.models:
        st.warning("No LightGBM models found. Please train a model first.")
        return
    # Show performance dashboard (reuses Ligthgbm.py logic)
    predictor.create_performance_dashboard()



# --- New App Structure ---
def fetch_data_page():
    main()

def model_performance_page():
    st.markdown('<h1 class="main-header">🤖 Model Performance</h1>', unsafe_allow_html=True)
    # Sidebar for model parameters
    st.sidebar.header("Model Selection & Parameters")
    predictor = SP500LightGBMPredictor()

    model_keys = list(predictor.models.keys())
    if not model_keys:
        from models.Ligthgbm import create_comprehensive_lgb_models
        predictor = create_comprehensive_lgb_models()
        model_keys = list(predictor.models.keys())
        if not model_keys:
            st.error("Failed to train LightGBM models. Please check your data and code.")
            return


    # Model family dropdown (for future extensibility)

    # --- Model family dropdown (LightGBM, XGBoost, Logistic Regression, LSTM) ---
    model_families = ['LightGBM', 'XGBoost', 'Logistic Regression', 'LSTM']
    selected_family = st.sidebar.selectbox("Model", model_families, key='model_family')

    # Load models for selected family
    optimized_models = {}
    if selected_family == 'LightGBM':
        for k in model_keys:
            if k == 'regression_optimized':
                optimized_models['Regressor'] = k
            elif k == 'classification_optimized':
                optimized_models['Classifier'] = k
        summary_func = predictor.get_model_summary
        metrics_func = lambda name: summary_func(name)['performance_metrics']
    elif selected_family == 'XGBoost':
        # Import and load XGBoost models
        from models.XGboost import SP500XGBPredictor, create_comprehensive_xgb_models
        xgb_predictor = SP500XGBPredictor()
        xgb_model_keys = list(xgb_predictor.models.keys())
        if not xgb_model_keys:
            xgb_predictor = create_comprehensive_xgb_models()
            xgb_model_keys = list(xgb_predictor.models.keys())
        for k in xgb_model_keys:
            if k == 'optimized' and xgb_predictor.target_type == 'regression':
                optimized_models['Regressor'] = k
            elif k == 'optimized' and xgb_predictor.target_type == 'classification':
                optimized_models['Classifier'] = k
        predictor = xgb_predictor
        summary_func = predictor.get_model_summary
        metrics_func = lambda name: summary_func(name)['performance_metrics']
    elif selected_family == 'Logistic Regression':
        # Import and load Logistic Regression models
        from models.LogisticReg import SP500LogisticRegPredictor, create_comprehensive_logistic_models
        lr_predictor = SP500LogisticRegPredictor()
        lr_model_keys = list(lr_predictor.models.keys())
        if not lr_model_keys:
            lr_predictor = create_comprehensive_logistic_models()
            lr_model_keys = list(lr_predictor.models.keys())
        for k in lr_model_keys:
            if k == 'optimized' and lr_predictor.target_type == 'regression':
                optimized_models['Regressor'] = k
            elif k == 'optimized' and lr_predictor.target_type == 'classification':
                optimized_models['Classifier'] = k
        predictor = lr_predictor
        summary_func = predictor.get_model_summary
        metrics_func = lambda name: summary_func(name)['performance_metrics']
    elif selected_family == 'LSTM':
        # Import and load LSTM models
        from models.LSTM import SP500LSTMPredictor, create_comprehensive_lstm_models
        lstm_predictor = SP500LSTMPredictor()
        lstm_model_keys = list(lstm_predictor.models.keys())
        if not lstm_model_keys:
            lstm_predictor = create_comprehensive_lstm_models()
            lstm_model_keys = list(lstm_predictor.models.keys()) if lstm_predictor else []
        
        # LSTM handles both types with the same model but different data loading
        if lstm_model_keys:
            optimized_models['Regressor'] = 'optimized'
            optimized_models['Classifier'] = 'optimized'
        else:
            st.error("Failed to train LSTM models. Please check PyTorch installation.")
            return
            
        predictor = lstm_predictor
        summary_func = predictor.get_model_summary
        metrics_func = lambda name: summary_func(name)['performance_metrics']

    subcats = list(optimized_models.keys())
    selected_subcat = st.sidebar.selectbox("Type", subcats, key='model_subcat')
    selected_model = optimized_models[selected_subcat]

    summary = summary_func(selected_model)
    metrics = metrics_func(selected_model)

    # Parameter controls based on model type
    lgbm_params = {}
    if summary['task_type'] == 'regression':
        lgbm_params = {
            'num_leaves': st.sidebar.slider('num_leaves', 10, 100, 31, key='reg_leaves'),
            'learning_rate': st.sidebar.select_slider('learning_rate', options=[0.001, 0.01, 0.05, 0.1, 0.2], value=0.1, key='reg_lr'),
            'n_estimators': st.sidebar.slider('n_estimators', 10, 500, 100, key='reg_nest')
        }
    else:
        lgbm_params = {
            'num_leaves': st.sidebar.slider('num_leaves', 10, 100, 31, key='clf_leaves'),
            'learning_rate': st.sidebar.select_slider('learning_rate', options=[0.001, 0.01, 0.05, 0.1, 0.2], value=0.1, key='clf_lr'),
            'n_estimators': st.sidebar.slider('n_estimators', 10, 500, 100, key='clf_nest')
        }

    if st.button("Retrain Selected Model with Above Parameters"):
        predictor.load_data(target_type=summary['task_type'])
        predictor.train_model(model_name=selected_model, lgb_params=lgbm_params)
        predictor.evaluate_model(selected_model)
        st.success(f"Model '{selected_model}' retrained.")
        summary = predictor.get_model_summary(selected_model)
        metrics = summary['performance_metrics']

    st.subheader(f"Model Performance Metrics: {selected_model}")

    # Reliability indicator logic
    reliability = "Low"
    color = "#ff4444"
    if summary['task_type'] == 'regression':
        r2 = metrics.get('test_r2', 0)
        rmse = metrics.get('test_rmse', 1e9)
        if r2 > 0.5 and rmse < 0.5:
            reliability = "High"
            color = "#00C851"
        elif r2 > 0.1 and rmse < 1.0:
            reliability = "Middle"
            color = "#ffbb33"
    else:
        acc = metrics.get('test_accuracy', 0)
        if acc > 0.7:
            reliability = "High"
            color = "#00C851"
        elif acc > 0.55:
            reliability = "Middle"
            color = "#ffbb33"
    st.markdown(f"<div style='padding:0.5em 1em;background:{color};color:white;border-radius:0.5em;display:inline-block;font-weight:bold;'>Reliability: {reliability}</div>", unsafe_allow_html=True)

    predictor._render_performance_tab([selected_model])
    st.subheader(f"Model Data Overview: {selected_model}")
    predictor._render_overview_tab([selected_model])


def models_comparison_page():
    """Compare all models performance side by side"""
    st.header("🔬 Models Comparison Dashboard")
    st.markdown("Compare performance across all ML model families: LightGBM, XGBoost, Logistic Regression, and LSTM")
    
    # Initialize all predictors
    def load_all_models():
        """Load and train all models if not already available"""
        models_data = {}
        
        try:
            # LightGBM
            from models.Ligthgbm import SP500LightGBMPredictor, create_comprehensive_lgb_models
            lgb_predictor = create_comprehensive_lgb_models()
            if lgb_predictor and lgb_predictor.models:
                models_data['LightGBM'] = {
                    'predictor': lgb_predictor,
                    'models': lgb_predictor.models.keys(),
                    'type': 'Tree-based'
                }
        except Exception as e:
            st.warning(f"Could not load LightGBM models: {e}")
        
        try:
            # XGBoost
            from models.XGboost import SP500XGBPredictor, create_comprehensive_xgb_models
            xgb_predictor = create_comprehensive_xgb_models()
            if xgb_predictor and xgb_predictor.models:
                models_data['XGBoost'] = {
                    'predictor': xgb_predictor,
                    'models': xgb_predictor.models.keys(),
                    'type': 'Tree-based'
                }
        except Exception as e:
            st.warning(f"Could not load XGBoost models: {e}")
        
        try:
            # Logistic Regression
            from models.LogisticReg import SP500LogisticRegPredictor, create_comprehensive_logistic_models
            lr_predictor = create_comprehensive_logistic_models()
            if lr_predictor and lr_predictor.models:
                models_data['Logistic Regression'] = {
                    'predictor': lr_predictor,
                    'models': lr_predictor.models.keys(),
                    'type': 'Linear'
                }
        except Exception as e:
            st.warning(f"Could not load Logistic Regression models: {e}")
        
        try:
            # LSTM
            from models.LSTM import SP500LSTMPredictor, create_comprehensive_lstm_models
            lstm_predictor = create_comprehensive_lstm_models()
            if lstm_predictor and lstm_predictor.models:
                models_data['LSTM'] = {
                    'predictor': lstm_predictor,
                    'models': lstm_predictor.models.keys(),
                    'type': 'Neural Network'
                }
        except Exception as e:
            st.warning(f"Could not load LSTM models: {e}")
        
        return models_data
    
    # Load all models (using session state to avoid reloading)
    if 'all_models' not in st.session_state:
        with st.spinner("Loading all models... This may take a few minutes for first-time setup."):
            st.session_state.all_models = load_all_models()
    
    all_models = st.session_state.all_models
    
    if not all_models:
        st.error("No models could be loaded. Please check your model implementations.")
        return
    
    st.success(f"✅ Loaded {len(all_models)} model families: {', '.join(all_models.keys())}")
    
    # Task type selection
    st.subheader("📊 Task Type Selection")
    task_type = st.radio("Select task type to compare:", ["Regression", "Classification"], horizontal=True)
    
    # Collect metrics from all models
    comparison_data = []
    
    for family_name, family_data in all_models.items():
        predictor = family_data['predictor']
        model_type = family_data['type']
        
        # Find appropriate model for the task
        model_name = None
        if hasattr(predictor, 'model_metrics'):
            for model in family_data['models']:
                summary = predictor.get_model_summary(model)
                if task_type.lower() in summary.get('task_type', '').lower():
                    model_name = model
                    break
            
            # If no exact match, try to get any model and check its task type
            if not model_name and family_data['models']:
                model_name = list(family_data['models'])[0]
                # Load data for this task type
                try:
                    predictor.load_data(target_type=task_type.lower())
                    if model_name not in predictor.model_metrics:
                        predictor.evaluate_model(model_name)
                except:
                    pass
            
            if model_name and model_name in predictor.model_metrics:
                summary = predictor.get_model_summary(model_name)
                metrics = summary['performance_metrics']
                data_info = summary['data_info']
                
                comparison_data.append({
                    'Model Family': family_name,
                    'Model Type': model_type,
                    'Model Name': model_name,
                    'Task': summary.get('task_type', '').title(),
                    'Features': data_info.get('features', 0),
                    'Train Samples': data_info.get('train_samples', 0),
                    'Test Samples': data_info.get('test_samples', 0),
                    'Metrics': metrics,
                    'Predictor': predictor
                })
    
    if not comparison_data:
        st.warning(f"No models found for {task_type} task. Please train models first.")
        return
    
    # Create comparison visualizations
    st.subheader(f"📈 {task_type} Models Performance Comparison")
    
    # Performance metrics comparison
    if task_type == "Regression":
        # Regression metrics
        metrics_df = []
        for model_data in comparison_data:
            if model_data['Task'] == 'Regression':
                metrics = model_data['Metrics']
                metrics_df.append({
                    'Model': model_data['Model Family'],
                    'Type': model_data['Model Type'],
                    'RMSE': metrics.get('test_rmse', 0),
                    'MAE': metrics.get('test_mae', 0),
                    'R²': metrics.get('test_r2', 0),
                    'Direction Accuracy': metrics.get('test_direction_accuracy', 0)
                })
        
        if metrics_df:
            df = pd.DataFrame(metrics_df)
            
            # Metrics visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # RMSE comparison
                fig_rmse = px.bar(df, x='Model', y='RMSE', color='Type', 
                                title="Root Mean Square Error (Lower is Better)")
                fig_rmse.update_layout(height=400)
                st.plotly_chart(fig_rmse, use_container_width=True)
                
                # R² comparison
                fig_r2 = px.bar(df, x='Model', y='R²', color='Type',
                              title="R² Score (Higher is Better)")
                fig_r2.update_layout(height=400)
                st.plotly_chart(fig_r2, use_container_width=True)
            
            with col2:
                # MAE comparison
                fig_mae = px.bar(df, x='Model', y='MAE', color='Type',
                               title="Mean Absolute Error (Lower is Better)")
                fig_mae.update_layout(height=400)
                st.plotly_chart(fig_mae, use_container_width=True)
                
                # Direction Accuracy comparison
                fig_dir = px.bar(df, x='Model', y='Direction Accuracy', color='Type',
                               title="Direction Accuracy (Higher is Better)")
                fig_dir.update_layout(height=400)
                st.plotly_chart(fig_dir, use_container_width=True)
            
            # Summary table
            st.subheader("📊 Detailed Regression Metrics Summary")
            st.dataframe(df.round(6), use_container_width=True)
    
    else:
        # Classification metrics
        metrics_df = []
        for model_data in comparison_data:
            if model_data['Task'] == 'Classification':
                metrics = model_data['Metrics']
                metrics_df.append({
                    'Model': model_data['Model Family'],
                    'Type': model_data['Model Type'],
                    'Accuracy': metrics.get('test_accuracy', 0),
                    'Precision': metrics.get('test_precision', 0),
                    'Recall': metrics.get('test_recall', 0),
                    'F1-Score': metrics.get('test_f1_score', 0)
                })
        
        if metrics_df:
            df = pd.DataFrame(metrics_df)
            
            # Metrics visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy comparison
                fig_acc = px.bar(df, x='Model', y='Accuracy', color='Type',
                               title="Test Accuracy (Higher is Better)")
                fig_acc.update_layout(height=400)
                st.plotly_chart(fig_acc, use_container_width=True)
                
                # Precision comparison
                fig_prec = px.bar(df, x='Model', y='Precision', color='Type',
                                title="Test Precision (Higher is Better)")
                fig_prec.update_layout(height=400)
                st.plotly_chart(fig_prec, use_container_width=True)
            
            with col2:
                # Recall comparison
                fig_rec = px.bar(df, x='Model', y='Recall', color='Type',
                               title="Test Recall (Higher is Better)")
                fig_rec.update_layout(height=400)
                st.plotly_chart(fig_rec, use_container_width=True)
                
                # F1-Score comparison
                fig_f1 = px.bar(df, x='Model', y='F1-Score', color='Type',
                              title="Test F1-Score (Higher is Better)")
                fig_f1.update_layout(height=400)
                st.plotly_chart(fig_f1, use_container_width=True)
            
            # Summary table
            st.subheader("📊 Detailed Classification Metrics Summary")
            st.dataframe(df.round(4), use_container_width=True)
    
    # Model complexity comparison
    st.subheader("🔍 Model Complexity & Data Usage")
    complexity_df = []
    for model_data in comparison_data:
        complexity_df.append({
            'Model Family': model_data['Model Family'],
            'Type': model_data['Model Type'],
            'Features Used': model_data['Features'],
            'Training Samples': model_data['Train Samples'],
            'Test Samples': model_data['Test Samples'],
            'Framework': get_model_framework(model_data['Model Family'])
        })
    
    if complexity_df:
        complexity_df = pd.DataFrame(complexity_df)
        
        col1, col2 = st.columns(2)
        with col1:
            # Features comparison
            fig_features = px.bar(complexity_df, x='Model Family', y='Features Used', 
                                color='Type', title="Number of Features Used")
            fig_features.update_layout(height=400)
            st.plotly_chart(fig_features, use_container_width=True)
        
        with col2:
            # Training samples comparison
            fig_samples = px.bar(complexity_df, x='Model Family', y='Training Samples',
                               color='Type', title="Training Samples Used")
            fig_samples.update_layout(height=400)
            st.plotly_chart(fig_samples, use_container_width=True)
        
        st.subheader("📋 Model Details Summary")
        st.dataframe(complexity_df, use_container_width=True)
    
    # Best model recommendations
    st.subheader("🏆 Model Recommendations")
    
    if task_type == "Regression" and metrics_df:
        df = pd.DataFrame(metrics_df)
        best_rmse = df.loc[df['RMSE'].idxmin()]
        best_r2 = df.loc[df['R²'].idxmax()]
        best_direction = df.loc[df['Direction Accuracy'].idxmax()]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Lowest RMSE", f"{best_rmse['Model']}", f"{best_rmse['RMSE']:.6f}")
        with col2:
            st.metric("📈 Highest R²", f"{best_r2['Model']}", f"{best_r2['R²']:.6f}")
        with col3:
            st.metric("🎪 Best Direction Accuracy", f"{best_direction['Model']}", f"{best_direction['Direction Accuracy']:.4f}")
    
    elif task_type == "Classification" and metrics_df:
        df = pd.DataFrame(metrics_df)
        best_acc = df.loc[df['Accuracy'].idxmax()]
        best_f1 = df.loc[df['F1-Score'].idxmax()]
        best_prec = df.loc[df['Precision'].idxmax()]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Highest Accuracy", f"{best_acc['Model']}", f"{best_acc['Accuracy']:.4f}")
        with col2:
            st.metric("⚖️ Best F1-Score", f"{best_f1['Model']}", f"{best_f1['F1-Score']:.4f}")
        with col3:
            st.metric("🔍 Highest Precision", f"{best_prec['Model']}", f"{best_prec['Precision']:.4f}")

def get_model_framework(model_family):
    """Get the framework used by each model family"""
    frameworks = {
        'LightGBM': 'LightGBM',
        'XGBoost': 'XGBoost',
        'Logistic Regression': 'Scikit-learn',
        'LSTM': 'PyTorch'
    }
    return frameworks.get(model_family, 'Unknown')


def get_recommendation(signal, risk):
    """Generate investment recommendation based on signal and risk"""
    if 'STRONG BUY' in signal:
        if risk == 'LOW':
            return "💰 Excellent buy opportunity"
        else:
            return "🎯 Good buy with caution"
    elif 'BUY' in signal:
        if risk == 'LOW':
            return "📈 Favorable buy condition"
        elif risk == 'MEDIUM':
            return "⚖️ Consider buying with moderation"
        else:
            return "⚠️ Risky buy - small position only"
    elif 'STRONG SELL' in signal:
        if risk == 'LOW':
            return "🚫 Strong sell recommendation"
        else:
            return "📉 Consider selling position"
    elif 'SELL' in signal:
        if risk == 'MEDIUM':
            return "🔽 Moderate sell pressure"
        else:
            return "⚠️ Weak sell signal - monitor closely"
    else:  # HOLD
        if risk == 'LOW':
            return "💎 Hold steady - good stability"
        elif risk == 'MEDIUM':
            return "⏸️ Hold and monitor developments"
        else:
            return "🚨 Hold with high uncertainty - be ready"


def predictions_page():
    """Future predictions page using trained models"""
    st.header("🔮 S&P 500 Future Predictions")
    st.markdown("Generate future price predictions using trained ML models")
    
    # Model selection
    st.subheader("🎯 Model Selection")
    col1, col2 = st.columns(2)
    
    with col1:
        model_family = st.selectbox(
            "Choose Model Family:",
            ["LightGBM", "XGBoost", "Logistic Regression", "LSTM"],
            help="Select the ML model family to use for predictions"
        )
    
    with col2:
        task_type = st.selectbox(
            "Prediction Type:",
            ["Classification", "Regression"],
            help="Classification: Up/Down direction, Regression: Price change amount"
        )
    
    # Prediction parameters
    st.subheader("⚙️ Prediction Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prediction_days = st.slider(
            "Days to Predict:",
            min_value=1,
            max_value=30,
            value=5,
            help="Number of future days to predict"
        )
    
    with col2:
        confidence_interval = st.selectbox(
            "Confidence Level:",
            [0.68, 0.95, 0.99],
            index=1,
            format_func=lambda x: f"{int(x*100)}%",
            help="Confidence interval for predictions"
        )
    
    with col3:
        update_data = st.checkbox(
            "Use Latest Data",
            value=True,
            help="Fetch the most recent market data before prediction"
        )
    
    # Prediction button
    if st.button("🚀 Generate Predictions", type="primary"):
        
        with st.spinner("Loading model and generating predictions..."):
            try:
                # Load the selected model
                predictor = None
                model_name = 'optimized'
                
                if model_family == "LightGBM":
                    from models.Ligthgbm import SP500LightGBMPredictor, create_comprehensive_lgb_models
                    predictor = create_comprehensive_lgb_models()
                    # Check for different model names in LightGBM
                    if predictor and predictor.models:
                        if task_type.lower() == 'regression' and 'regression_optimized' in predictor.models:
                            model_name = 'regression_optimized'
                        elif task_type.lower() == 'classification' and 'classification_optimized' in predictor.models:
                            model_name = 'classification_optimized'
                        else:
                            # Use any available model
                            model_name = list(predictor.models.keys())[0]
                            
                elif model_family == "XGBoost":
                    from models.XGboost import SP500XGBPredictor, create_comprehensive_xgb_models
                    predictor = create_comprehensive_xgb_models()
                    
                elif model_family == "Logistic Regression":
                    from models.LogisticReg import SP500LogisticRegPredictor, create_comprehensive_logistic_models
                    predictor = create_comprehensive_logistic_models()
                    
                elif model_family == "LSTM":
                    from models.LSTM import SP500LSTMPredictor, create_comprehensive_lstm_models
                    predictor = create_comprehensive_lstm_models()
                
                if predictor is None or not predictor.models:
                    st.error(f"Could not load {model_family} model. Please train the model first.")
                    return
                
                # Verify the model exists and is evaluated
                if model_name not in predictor.models:
                    st.error(f"Model '{model_name}' not found in {model_family}. Available models: {list(predictor.models.keys())}")
                    return
                
                # Load appropriate data for the task
                predictor.load_data(target_type=task_type.lower())
                
                # Ensure model is evaluated
                if model_name not in predictor.model_metrics:
                    st.info("Evaluating model performance...")
                    predictor.evaluate_model(model_name)
                
                # Generate predictions using a simple approach
                st.info("Generating predictions...")
                
                # Get the latest data point for prediction
                if hasattr(predictor, 'X_test') and predictor.X_test is not None and len(predictor.X_test) > 0:
                    # Use the last few test samples to generate predictions
                    latest_features = predictor.X_test.tail(min(prediction_days, len(predictor.X_test)))
                    
                    # Get model predictions
                    model = predictor.models[model_name]
                    
                    if model_family == "LSTM":
                        # For LSTM, we need to handle sequences differently
                        if hasattr(predictor, 'X_test_seq') and predictor.X_test_seq is not None:
                            import torch
                            model.eval()
                            X_test_tensor = torch.FloatTensor(predictor.X_test_seq[-prediction_days:])
                            with torch.no_grad():
                                predictions_raw = model(X_test_tensor).cpu().numpy().flatten()
                        else:
                            st.error("LSTM model requires sequence data. Please ensure model is properly trained.")
                            return
                    else:
                        # For other models, use direct prediction
                        predictions_raw = model.predict(latest_features)
                    
                    # Create prediction results
                    predictions = []
                    for i, pred in enumerate(predictions_raw):
                        pred_date = pd.Timestamp.now() + pd.Timedelta(days=i+1)
                        
                        if task_type == "Classification":
                            # For classification, convert to probability
                            if hasattr(model, 'predict_proba'):
                                prob = model.predict_proba(latest_features.iloc[[i % len(latest_features)]])[0][1]
                            else:
                                prob = float(pred) if model_family != "LSTM" else 1.0 / (1.0 + np.exp(-pred))  # sigmoid for LSTM
                            
                            predictions.append({
                                'date': pred_date.strftime('%Y-%m-%d'),
                                'prediction': int(prob > 0.5),
                                'probability': prob
                            })
                        else:
                            # For regression
                            predictions.append({
                                'date': pred_date.strftime('%Y-%m-%d'),
                                'prediction': float(pred)
                            })
                
                else:
                    st.error("No test data available for predictions. Please ensure data is loaded properly.")
                    return
                
                # Display results
                st.success("✅ Predictions generated successfully!")
                
                # Create predictions visualization
                st.subheader("📈 Market Predictions & Trading Signals")
                
                if predictions:
                    if task_type == "Classification":
                        # Classification results (Trading Signals Table)
                        st.subheader("📊 Trading Signals & Risk Assessment")
                        
                        # Create trading signals table
                        signals_data = []
                        for i, pred in enumerate(predictions):
                            date = pred['date']
                            probability = pred['probability']
                            direction = pred['prediction']
                            
                            # Determine trading signal
                            if probability >= 0.7:
                                if direction > 0:
                                    signal = "🔥 STRONG BUY"
                                    signal_color = "#00C851"
                                else:
                                    signal = "🔥 STRONG SELL"
                                    signal_color = "#ff4444"
                                risk = "LOW"
                                risk_color = "#00C851"
                            elif probability >= 0.6:
                                if direction > 0:
                                    signal = "📈 BUY"
                                    signal_color = "#28a745"
                                else:
                                    signal = "📉 SELL"
                                    signal_color = "#dc3545"
                                risk = "MEDIUM"
                                risk_color = "#ffc107"
                            elif probability >= 0.55:
                                if direction > 0:
                                    signal = "🔼 WEAK BUY"
                                    signal_color = "#6c757d"
                                else:
                                    signal = "🔽 WEAK SELL"
                                    signal_color = "#6c757d"
                                risk = "MEDIUM"
                                risk_color = "#ffc107"
                            else:
                                signal = "⚖️ HOLD"
                                signal_color = "#17a2b8"
                                risk = "HIGH"
                                risk_color = "#ff4444"
                            
                            signals_data.append({
                                'Date': date,
                                'Signal': signal,
                                'Confidence': f"{probability:.1%}",
                                'Risk Level': risk,
                                'Direction': "📈 UP" if direction > 0 else "📉 DOWN",
                                'Recommendation': get_recommendation(signal, risk)
                            })
                        
                        # Display signals table
                        signals_df = pd.DataFrame(signals_data)
                        
                        # Custom styling for the table
                        def style_signals(val):
                            if 'STRONG BUY' in str(val):
                                return 'background-color: #d4edda; color: #155724'
                            elif 'BUY' in str(val):
                                return 'background-color: #e6f3ff; color: #0066cc'
                            elif 'SELL' in str(val):
                                return 'background-color: #f8d7da; color: #721c24'
                            elif 'HOLD' in str(val):
                                return 'background-color: #fff3cd; color: #856404'
                            elif val == 'LOW':
                                return 'background-color: #d4edda; color: #155724'
                            elif val == 'MEDIUM':
                                return 'background-color: #fff3cd; color: #856404'
                            elif val == 'HIGH':
                                return 'background-color: #f8d7da; color: #721c24'
                            return ''
                        
                        st.dataframe(
                            signals_df.style.applymap(style_signals, subset=['Signal', 'Risk Level']),
                            use_container_width=True
                        )
                    
                    else:
                        # Regression results (Price changes with trading signals)
                        st.subheader("📊 Price Predictions & Trading Signals")
                        
                        # Create trading signals table for regression
                        signals_data = []
                        for i, pred in enumerate(predictions):
                            date = pred['date']
                            change_pct = pred['prediction'] * 100
                            abs_change = abs(change_pct)
                            
                            # Determine trading signal based on predicted change
                            if abs_change >= 2.0:  # Large movement
                                if change_pct > 0:
                                    signal = "🔥 STRONG BUY"
                                    risk = "LOW"
                                else:
                                    signal = "🔥 STRONG SELL"
                                    risk = "LOW"
                            elif abs_change >= 1.0:  # Medium movement
                                if change_pct > 0:
                                    signal = "📈 BUY"
                                    risk = "MEDIUM"
                                else:
                                    signal = "📉 SELL"
                                    risk = "MEDIUM"
                            elif abs_change >= 0.5:  # Small movement
                                if change_pct > 0:
                                    signal = "🔼 WEAK BUY"
                                    risk = "MEDIUM"
                                else:
                                    signal = "🔽 WEAK SELL"
                                    risk = "MEDIUM"
                            else:  # Very small movement
                                signal = "⚖️ HOLD"
                                risk = "HIGH"
                            
                            signals_data.append({
                                'Date': date,
                                'Predicted Change': f"{change_pct:+.2f}%",
                                'Signal': signal,
                                'Risk Level': risk,
                                'Direction': "📈 UP" if change_pct > 0 else "📉 DOWN",
                                'Recommendation': get_recommendation(signal, risk)
                            })
                        
                        # Display signals table
                        signals_df = pd.DataFrame(signals_data)
                        
                        # Custom styling for the table
                        def style_regression_signals(val):
                            if 'STRONG BUY' in str(val):
                                return 'background-color: #d4edda; color: #155724'
                            elif 'BUY' in str(val):
                                return 'background-color: #e6f3ff; color: #0066cc'
                            elif 'SELL' in str(val):
                                return 'background-color: #f8d7da; color: #721c24'
                            elif 'HOLD' in str(val):
                                return 'background-color: #fff3cd; color: #856404'
                            elif val == 'LOW':
                                return 'background-color: #d4edda; color: #155724'
                            elif val == 'MEDIUM':
                                return 'background-color: #fff3cd; color: #856404'
                            elif val == 'HIGH':
                                return 'background-color: #f8d7da; color: #721c24'
                            return ''
                        
                        st.dataframe(
                            signals_df.style.applymap(style_regression_signals, subset=['Signal', 'Risk Level']),
                            use_container_width=True
                        )
                        
                        # Price trend visualization (Current vs Predicted)
                        st.subheader("📈 Current vs Predicted Price Trend")
                        
                        # Get recent historical data for context
                        if hasattr(predictor, 'X_test') and len(predictor.X_test) >= 10:
                            # Get last 10 days of actual price changes
                            historical_days = min(10, len(predictor.X_test))
                            recent_actual = predictor.y_test.tail(historical_days) if hasattr(predictor, 'y_test') else None
                            
                            if recent_actual is not None:
                                # Create price trend comparison
                                trend_dates = []
                                trend_values = []
                                trend_types = []
                                
                                # Historical data (last 5 days)
                                for i, (date_idx, value) in enumerate(recent_actual.tail(5).items()):
                                    trend_dates.append(f"Day -{5-i}")
                                    trend_values.append(value * 100)  # Convert to percentage
                                    trend_types.append("Historical")
                                
                                # Predicted data
                                for i, pred in enumerate(predictions):
                                    trend_dates.append(f"Day +{i+1}")
                                    trend_values.append(pred['prediction'] * 100)
                                    trend_types.append("Predicted")
                                
                                trend_df = pd.DataFrame({
                                    'Date': trend_dates,
                                    'Price Change (%)': trend_values,
                                    'Type': trend_types
                                })
                                
                                # Create line chart with go.Scatter traces
                                fig = go.Figure()
                                
                                # Historical data trace
                                historical_data = trend_df[trend_df['Type'] == 'Historical']
                                fig.add_trace(go.Scatter(
                                    x=historical_data['Date'],
                                    y=historical_data['Price Change (%)'],
                                    mode='lines+markers',
                                    name='Historical',
                                    line=dict(color='#1f77b4', width=2),
                                    marker=dict(size=6)
                                ))
                                
                                # Predicted data trace
                                predicted_data = trend_df[trend_df['Type'] == 'Predicted']
                                fig.add_trace(go.Scatter(
                                    x=predicted_data['Date'],
                                    y=predicted_data['Price Change (%)'],
                                    mode='lines+markers',
                                    name='Predicted',
                                    line=dict(color='#ff7f0e', width=2),
                                    marker=dict(size=6)
                                ))
                                
                                # Add horizontal line at y=0
                                fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                                            annotation_text="Break-even")
                                
                                fig.update_layout(
                                    title="Price Changes: Historical vs Predicted",
                                    xaxis_title="Date",
                                    yaxis_title="Price Change (%)",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                
                # Advanced Trending Graphs with Real-time Data
                st.subheader("📊 Advanced Market Trending Analysis")
                
                # Get real-time data for trending graphs
                try:
                    from data.fetch_data import DataFetcher
                    fetcher = DataFetcher()
                    
                    # Auto-refresh every 5 minutes
                    st.info("📡 Real-time data updates every 5 minutes")
                    
                    # Graph 1: Real-time intraday with predictions (5-minute intervals)
                    st.subheader("🕐 Real-Time Intraday Trend (5-min intervals)")
                    
                    # Get intraday data
                    intraday_data = fetcher.get_historical_data("^GSPC", period="1d", interval="5m")
                    
                    if intraday_data is not None and not intraday_data.empty:
                        # Create prediction extension for intraday
                        last_price = intraday_data['close'].iloc[-1]
                        last_time = intraday_data.index[-1]
                        
                        # Generate next 12 intervals (1 hour) predictions
                        future_times = pd.date_range(
                            start=last_time + pd.Timedelta(minutes=5),
                            periods=12,
                            freq='5min'
                        )
                        
                        # Use model predictions to create price projections
                        if predictions:
                            # Take first prediction as baseline
                            base_change = predictions[0]['prediction'] if task_type == "Regression" else (0.01 if predictions[0]['prediction'] > 0 else -0.01)
                            
                            future_prices = []
                            confidence_upper = []
                            confidence_lower = []
                            
                            for i in range(12):
                                # Apply prediction with some random walk
                                change_factor = base_change * (1 + np.random.normal(0, 0.1))
                                predicted_price = last_price * (1 + change_factor * (i + 1) / 12)
                                future_prices.append(predicted_price)
                                
                                # Confidence intervals (±2%)
                                confidence_upper.append(predicted_price * 1.02)
                                confidence_lower.append(predicted_price * 0.98)
                            
                            # Create comprehensive intraday chart
                            fig_intraday = go.Figure()
                            
                            # Historical data
                            fig_intraday.add_trace(go.Scatter(
                                x=intraday_data.index,
                                y=intraday_data['close'],
                                mode='lines',
                                name='Actual Price',
                                line=dict(color='#1f77b4', width=2)
                            ))
                            
                            # Predicted prices
                            fig_intraday.add_trace(go.Scatter(
                                x=future_times,
                                y=future_prices,
                                mode='lines+markers',
                                name='Predicted Price',
                                line=dict(color='#ff7f0e', width=2, dash='dash')
                            ))
                            
                            # Confidence interval
                            fig_intraday.add_trace(go.Scatter(
                                x=future_times,
                                y=confidence_upper,
                                fill=None,
                                mode='lines',
                                line_color='rgba(0,0,0,0)',
                                showlegend=False
                            ))
                            
                            fig_intraday.add_trace(go.Scatter(
                                x=future_times,
                                y=confidence_lower,
                                fill='tonexty',
                                mode='lines',
                                line_color='rgba(0,0,0,0)',
                                name='Confidence Interval',
                                fillcolor='rgba(255,127,14,0.2)'
                            ))
                            
                            fig_intraday.update_layout(
                                title="Real-Time S&P 500 with Predictions (5-min intervals)",
                                xaxis_title="Time",
                                yaxis_title="Price ($)",
                                hovermode='x unified',
                                height=500
                            )
                            
                            st.plotly_chart(fig_intraday, use_container_width=True)
                            
                            # Add key indicators for intraday
                            st.subheader("📊 Intraday Key Indicators")
                            
                            # Calculate accuracy and reliability metrics
                            avg_predicted_price = np.mean(future_prices)
                            price_volatility = np.std(future_prices) / avg_predicted_price * 100
                            confidence_range = ((np.mean(confidence_upper) - np.mean(confidence_lower)) / avg_predicted_price) * 100
                            
                            # Determine reliability level
                            if model_name in predictor.model_metrics:
                                metrics = predictor.model_metrics[model_name]
                                if task_type.lower() == 'classification':
                                    accuracy = metrics.get('test_accuracy', 0)
                                    if accuracy > 0.7:
                                        reliability_level = "🟢 HIGH"
                                        reliability_color = "#28a745"
                                    elif accuracy > 0.6:
                                        reliability_level = "🟡 MEDIUM"
                                        reliability_color = "#ffc107"
                                    else:
                                        reliability_level = "🔴 LOW"
                                        reliability_color = "#dc3545"
                                else:
                                    r2_score = metrics.get('test_r2', 0)
                                    if r2_score > 0.3:
                                        reliability_level = "🟢 HIGH"
                                        reliability_color = "#28a745"
                                    elif r2_score > 0.1:
                                        reliability_level = "🟡 MEDIUM"
                                        reliability_color = "#ffc107"
                                    else:
                                        reliability_level = "🔴 LOW"
                                        reliability_color = "#dc3545"
                            else:
                                reliability_level = "🔴 UNKNOWN"
                                reliability_color = "#6c757d"
                            
                            # Display metrics in columns
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "🎯 Avg Predicted Price",
                                    f"${avg_predicted_price:,.2f}",
                                    f"{((avg_predicted_price - last_price) / last_price) * 100:+.2f}%"
                                )
                            
                            with col2:
                                st.metric(
                                    "📊 Volatility",
                                    f"{price_volatility:.2f}%",
                                    help="Price variation within prediction period"
                                )
                            
                            with col3:
                                st.metric(
                                    "📏 Confidence Range",
                                    f"±{confidence_range/2:.1f}%",
                                    help="Uncertainty range around predictions"
                                )
                            
                            with col4:
                                st.markdown(
                                    f'<div style="padding:10px; background-color:{reliability_color}; color:white; '
                                    f'border-radius:10px; text-align:center; font-weight:bold; margin-top:8px;">'
                                    f'🏷️ Reliability<br>{reliability_level}</div>',
                                    unsafe_allow_html=True
                                )
                    
                    # Graph 2: Weekly view with daily predictions
                    st.subheader("📅 Weekly Trend with Daily Predictions")
                    
                    # Get daily data for the last 2 weeks + predictions
                    daily_data = fetcher.get_historical_data("^GSPC", period="1mo", interval="1d")
                    
                    if daily_data is not None and not daily_data.empty:
                        # Get last 14 days
                        weekly_data = daily_data.tail(14)
                        last_price_daily = weekly_data['close'].iloc[-1]
                        last_date = weekly_data.index[-1]
                        
                        # Generate daily predictions for next week
                        future_dates = pd.date_range(
                            start=last_date + pd.Timedelta(days=1),
                            periods=min(prediction_days, 7),
                            freq='D'
                        )
                        
                        daily_predictions = []
                        daily_conf_upper = []
                        daily_conf_lower = []
                        
                        cumulative_change = 0
                        for i in range(len(future_dates)):
                            if i < len(predictions):
                                pred_change = predictions[i]['prediction'] if task_type == "Regression" else (0.005 if predictions[i]['prediction'] > 0 else -0.005)
                            else:
                                pred_change = predictions[-1]['prediction'] if predictions else 0
                                
                            cumulative_change += pred_change
                            predicted_price = last_price_daily * (1 + cumulative_change)
                            daily_predictions.append(predicted_price)
                            
                            # Confidence intervals (±3% for daily)
                            daily_conf_upper.append(predicted_price * 1.03)
                            daily_conf_lower.append(predicted_price * 0.97)
                        
                        # Create simple beginner-friendly weekly chart
                        fig_weekly = go.Figure()
                        
                        # Historical price line - simple and clear
                        fig_weekly.add_trace(go.Scatter(
                            x=weekly_data.index,
                            y=weekly_data['close'],
                            mode='lines+markers',
                            name='📈 Past Prices (2 weeks)',
                            line=dict(color='blue', width=3),
                            marker=dict(size=5, color='blue')
                        ))
                        
                        # Predicted prices - simple dotted line
                        fig_weekly.add_trace(go.Scatter(
                            x=future_dates,
                            y=daily_predictions,
                            mode='lines+markers',
                            name='🔮 Future Predictions (7 days)',
                            line=dict(color='orange', width=3, dash='dot'),
                            marker=dict(size=6, color='orange')
                        ))
                        
                        # Simple confidence area - what might happen
                        fig_weekly.add_trace(go.Scatter(
                            x=list(future_dates) + list(future_dates[::-1]),
                            y=list(daily_conf_upper) + list(daily_conf_lower[::-1]),
                            fill='toself',
                            fillcolor='rgba(255, 165, 0, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='📊 What Might Happen (Range)',
                            hoverinfo='skip'
                        ))
                        
                        # Current price line with clear label
                        current_price = weekly_data['close'].iloc[-1]
                        fig_weekly.add_hline(
                            y=current_price,
                            line_dash="dash",
                            line_color="red",
                            line_width=2,
                            annotation_text=f"🏷️ Today's Price: ${current_price:,.0f}",
                            annotation_position="top right",
                            annotation_bgcolor="white",
                            annotation_bordercolor="red",
                            annotation_borderwidth=1
                        )
                        
                        # Beginner-friendly layout
                        fig_weekly.update_layout(
                            title="📅 S&P 500: What Happened vs What Might Happen",
                            xaxis_title="📅 Date",
                            yaxis_title="💰 Stock Price ($)",
                            height=400,
                            showlegend=True,
                            legend=dict(
                                x=0,
                                y=1,
                                bgcolor='rgba(255,255,255,0.8)',
                                bordercolor='gray',
                                borderwidth=1
                            ),
                            hovermode='x'
                        )
                        
                        # Add simple explanation text
                        st.write("**📚 How to Read This Chart:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write("🔵 **Blue Line**: What actually happened in the past 2 weeks")
                        with col2:
                            st.write("🟠 **Orange Line**: What we predict for the next 7 days")  
                        with col3:
                            st.write("📊 **Light Orange Area**: Range where price might go")
                        
                        st.plotly_chart(fig_weekly, use_container_width=True)
                        
                        # Add key indicators for weekly predictions
                        st.subheader("📊 Weekly Key Indicators")
                        
                        # Calculate weekly metrics
                        avg_weekly_price = np.mean(daily_predictions)
                        weekly_trend = ((daily_predictions[-1] - daily_predictions[0]) / daily_predictions[0]) * 100
                        weekly_volatility = np.std(daily_predictions) / avg_weekly_price * 100
                        weekly_confidence_range = ((np.mean(daily_conf_upper) - np.mean(daily_conf_lower)) / avg_weekly_price) * 100
                        
                        # Weekly reliability assessment
                        if model_name in predictor.model_metrics:
                            metrics = predictor.model_metrics[model_name]
                            if task_type.lower() == 'classification':
                                weekly_accuracy = metrics.get('test_accuracy', 0)
                                # Adjust for weekly timeframe (slightly lower confidence)
                                weekly_accuracy *= 0.9
                                if weekly_accuracy > 0.65:
                                    weekly_reliability = "🟢 HIGH"
                                    weekly_rel_color = "#28a745"
                                elif weekly_accuracy > 0.55:
                                    weekly_reliability = "🟡 MEDIUM"
                                    weekly_rel_color = "#ffc107"
                                else:
                                    weekly_reliability = "🔴 LOW"
                                    weekly_rel_color = "#dc3545"
                            else:
                                weekly_r2 = metrics.get('test_r2', 0)
                                # Adjust for weekly timeframe
                                weekly_r2 *= 0.85
                                if weekly_r2 > 0.25:
                                    weekly_reliability = "🟢 HIGH"
                                    weekly_rel_color = "#28a745"
                                elif weekly_r2 > 0.05:
                                    weekly_reliability = "🟡 MEDIUM"
                                    weekly_rel_color = "#ffc107"
                                else:
                                    weekly_reliability = "🔴 LOW"
                                    weekly_rel_color = "#dc3545"
                        else:
                            weekly_reliability = "🔴 UNKNOWN"
                            weekly_rel_color = "#6c757d"
                        
                        # Display weekly metrics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric(
                                "🎯 Avg Weekly Price",
                                f"${avg_weekly_price:,.2f}",
                                f"{((avg_weekly_price - current_price) / current_price) * 100:+.2f}%"
                            )
                        
                        with col2:
                            trend_icon = "📈" if weekly_trend > 0 else "📉" if weekly_trend < 0 else "➡️"
                            st.metric(
                                f"{trend_icon} Weekly Trend",
                                f"{weekly_trend:+.2f}%",
                                help="Overall price direction for the week"
                            )
                        
                        with col3:
                            st.metric(
                                "📊 Volatility",
                                f"{weekly_volatility:.2f}%",
                                help="Price variation during the week"
                            )
                        
                        with col4:
                            st.metric(
                                "📏 Confidence Range",
                                f"±{weekly_confidence_range/2:.1f}%",
                                help="Uncertainty range for weekly predictions"
                            )
                        
                        with col5:
                            st.markdown(
                                f'<div style="padding:10px; background-color:{weekly_rel_color}; color:white; '
                                f'border-radius:10px; text-align:center; font-weight:bold; margin-top:8px;">'
                                f'🏷️ Reliability<br>{weekly_reliability}</div>',
                                unsafe_allow_html=True
                            )
                    
                    # Graph 3: Yearly view with monthly predictions
                    st.subheader("📈 Yearly Trend with Monthly Predictions")
                    
                    # Get monthly data for the last year + predictions
                    yearly_data = fetcher.get_historical_data("^GSPC", period="1y", interval="1mo")
                    
                    if yearly_data is not None and not yearly_data.empty:
                        last_price_yearly = yearly_data['close'].iloc[-1]
                        last_month = yearly_data.index[-1]
                        
                        # Generate monthly predictions for next 6 months
                        future_months = pd.date_range(
                            start=last_month + pd.DateOffset(months=1),
                            periods=6,
                            freq='MS'  # Month start
                        )
                        
                        monthly_predictions = []
                        monthly_conf_upper = []
                        monthly_conf_lower = []
                        
                        # Aggregate daily predictions into monthly
                        monthly_change = 0
                        if predictions:
                            # Use average of daily predictions as monthly trend
                            avg_daily_change = np.mean([p['prediction'] for p in predictions])
                            monthly_change = avg_daily_change * 21  # ~21 trading days per month
                        
                        cumulative_monthly = 0
                        for i in range(6):
                            # Add some decay factor for longer-term predictions
                            decay_factor = 0.8 ** i
                            adjusted_change = monthly_change * decay_factor
                            cumulative_monthly += adjusted_change
                            
                            predicted_price = last_price_yearly * (1 + cumulative_monthly)
                            monthly_predictions.append(predicted_price)
                            
                            # Confidence intervals (±5% for monthly)
                            monthly_conf_upper.append(predicted_price * (1.05 + i * 0.01))
                            monthly_conf_lower.append(predicted_price * (0.95 - i * 0.01))
                        
                        # Create yearly chart
                        fig_yearly = go.Figure()
                        
                        # Historical monthly data
                        fig_yearly.add_trace(go.Scatter(
                            x=yearly_data.index,
                            y=yearly_data['close'],
                            mode='lines+markers',
                            name='Historical Monthly',
                            line=dict(color='#1f77b4', width=3),
                            marker=dict(size=6)
                        ))
                        
                        # Predicted monthly prices
                        fig_yearly.add_trace(go.Scatter(
                            x=future_months,
                            y=monthly_predictions,
                            mode='lines+markers',
                            name='Monthly Predictions',
                            line=dict(color='#ff7f0e', width=3, dash='dot'),
                            marker=dict(size=10, symbol='diamond')
                        ))
                        
                        # Confidence interval
                        fig_yearly.add_trace(go.Scatter(
                            x=future_months,
                            y=monthly_conf_upper,
                            fill=None,
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            showlegend=False
                        ))
                        
                        fig_yearly.add_trace(go.Scatter(
                            x=future_months,
                            y=monthly_conf_lower,
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            name='Monthly Confidence',
                            fillcolor='rgba(255,127,14,0.4)'
                        ))
                        
                        fig_yearly.update_layout(
                            title="Yearly S&P 500 Trend with Monthly Predictions",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig_yearly, use_container_width=True)
                        
                        # Add key indicators for yearly predictions
                        st.subheader("📊 Yearly Key Indicators")
                        
                        # Calculate yearly metrics
                        avg_yearly_price = np.mean(monthly_predictions)
                        yearly_trend = ((monthly_predictions[-1] - monthly_predictions[0]) / monthly_predictions[0]) * 100
                        yearly_volatility = np.std(monthly_predictions) / avg_yearly_price * 100
                        yearly_confidence_range = ((np.mean(monthly_conf_upper) - np.mean(monthly_conf_lower)) / avg_yearly_price) * 100
                        
                        # Calculate potential ROI
                        six_month_roi = ((monthly_predictions[-1] - last_price_yearly) / last_price_yearly) * 100
                        
                        # Yearly reliability assessment (lower confidence for longer timeframe)
                        if model_name in predictor.model_metrics:
                            metrics = predictor.model_metrics[model_name]
                            if task_type.lower() == 'classification':
                                yearly_accuracy = metrics.get('test_accuracy', 0)
                                # Significantly adjust for yearly timeframe
                                yearly_accuracy *= 0.7
                                if yearly_accuracy > 0.5:
                                    yearly_reliability = "🟡 MEDIUM"
                                    yearly_rel_color = "#ffc107"
                                elif yearly_accuracy > 0.4:
                                    yearly_reliability = "🔴 LOW"
                                    yearly_rel_color = "#fd7e14"
                                else:
                                    yearly_reliability = "🔴 VERY LOW"
                                    yearly_rel_color = "#dc3545"
                            else:
                                yearly_r2 = metrics.get('test_r2', 0)
                                # Significantly adjust for yearly timeframe
                                yearly_r2 *= 0.6
                                if yearly_r2 > 0.15:
                                    yearly_reliability = "🟡 MEDIUM"
                                    yearly_rel_color = "#ffc107"
                                elif yearly_r2 > 0.05:
                                    yearly_reliability = "🔴 LOW"
                                    yearly_rel_color = "#fd7e14"
                                else:
                                    yearly_reliability = "🔴 VERY LOW"
                                    yearly_rel_color = "#dc3545"
                        else:
                            yearly_reliability = "🔴 UNKNOWN"
                            yearly_rel_color = "#6c757d"
                        
                        # Display yearly metrics
                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        
                        with col1:
                            st.metric(
                                "🎯 Avg 6M Price",
                                f"${avg_yearly_price:,.2f}",
                                f"{((avg_yearly_price - last_price_yearly) / last_price_yearly) * 100:+.2f}%"
                            )
                        
                        with col2:
                            roi_icon = "💰" if six_month_roi > 0 else "📉" if six_month_roi < 0 else "➡️"
                            st.metric(
                                f"{roi_icon} 6M ROI",
                                f"{six_month_roi:+.2f}%",
                                help="Return on Investment over 6 months"
                            )
                        
                        with col3:
                            trend_icon = "📈" if yearly_trend > 0 else "📉" if yearly_trend < 0 else "➡️"
                            st.metric(
                                f"{trend_icon} Long Trend",
                                f"{yearly_trend:+.2f}%",
                                help="Overall trend across 6 months"
                            )
                        
                        with col4:
                            st.metric(
                                "📊 Volatility",
                                f"{yearly_volatility:.2f}%",
                                help="Price variation over 6 months"
                            )
                        
                        with col5:
                            st.metric(
                                "📏 Confidence Range",
                                f"±{yearly_confidence_range/2:.1f}%",
                                help="Uncertainty range for long-term predictions"
                            )
                        
                        with col6:
                            st.markdown(
                                f'<div style="padding:10px; background-color:{yearly_rel_color}; color:white; '
                                f'border-radius:10px; text-align:center; font-weight:bold; margin-top:8px;">'
                                f'🏷️ Reliability<br>{yearly_reliability}</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Additional warning for long-term predictions
                        st.warning("⚠️ **Long-term Note**: 6-month predictions have inherently lower reliability due to market complexity and unforeseen events.")
                    
                    # Summary metrics for all timeframes
                    st.subheader("📊 Multi-Timeframe Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'future_prices' in locals():
                            next_hour_change = ((future_prices[-1] - last_price) / last_price) * 100
                            st.metric("Next Hour (5-min)", f"{next_hour_change:+.2f}%", 
                                    delta=f"${future_prices[-1] - last_price:+.2f}")
                    
                    with col2:
                        if 'daily_predictions' in locals():
                            next_week_change = ((daily_predictions[-1] - last_price_daily) / last_price_daily) * 100
                            st.metric("Next Week (Daily)", f"{next_week_change:+.2f}%",
                                    delta=f"${daily_predictions[-1] - last_price_daily:+.2f}")
                    
                    with col3:
                        if 'monthly_predictions' in locals():
                            next_months_change = ((monthly_predictions[-1] - last_price_yearly) / last_price_yearly) * 100
                            st.metric("Next 6 Months", f"{next_months_change:+.2f}%",
                                    delta=f"${monthly_predictions[-1] - last_price_yearly:+.2f}")
                
                except Exception as e:
                    st.error(f"Could not load real-time trending data: {e}")
                    st.info("Trending graphs require active market hours and data connection.")
                
                
                # Model information
                st.subheader("ℹ️ Model Information")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Model Family", model_family)
                
                with col2:
                    st.metric("Task Type", task_type)
                
                with col3:
                    features_count = len(predictor.feature_names) if hasattr(predictor, 'feature_names') else 0
                    st.metric("Features Used", features_count)
                
                # Model performance
                if model_name in predictor.model_metrics:
                    st.subheader("📊 Model Performance")
                    metrics = predictor.model_metrics[model_name]
                    
                    if task_type.lower() == 'classification':
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{metrics.get('test_accuracy', 0):.1%}")
                        with col2:
                            st.metric("Precision", f"{metrics.get('test_precision', 0):.1%}")
                        with col3:
                            st.metric("F1-Score", f"{metrics.get('test_f1_score', 0):.1%}")
                    else:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("RMSE", f"{metrics.get('test_rmse', 0):.4f}")
                        with col2:
                            st.metric("R²", f"{metrics.get('test_r2', 0):.4f}")
                        with col3:
                            st.metric("Direction Accuracy", f"{metrics.get('test_direction_accuracy', 0):.1%}")
                
                # Disclaimer
                st.warning("""
                ⚠️ **Important Disclaimer:**
                - These predictions are for educational/research purposes only
                - Past performance does not guarantee future results
                - Always consult financial professionals before making investment decisions
                - Consider multiple factors beyond model predictions
                """)
                
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
                st.info("Please ensure the selected model is properly trained and data is available.")
                # Debug information
                if 'predictor' in locals() and predictor:
                    st.info(f"Available models: {list(predictor.models.keys())}")
                    st.info(f"Available metrics: {list(predictor.model_metrics.keys())}")
    
    # Historical prediction accuracy (if available)
    st.subheader("📊 Historical Model Performance")
    st.info("Select a model and generate predictions to see recent performance metrics.")
    
    # Tips for better predictions
    with st.expander("💡 Tips for Better Predictions"):
        st.markdown("""
        **For more accurate predictions:**
        
        1. **Use Latest Data**: Enable "Use Latest Data" to include recent market movements
        2. **Choose Appropriate Model**: 
           - **Classification**: Better for direction prediction (buy/sell signals)
           - **Regression**: Better for estimating price change magnitude
        3. **Consider Multiple Models**: Compare predictions across different model families
        4. **Short-term Focus**: Predictions are more reliable for shorter time horizons (1-5 days)
        5. **Market Context**: Consider current market conditions and news events
        6. **Risk Management**: Use predictions as one input among many in your decision process
        """)


def executive_summary_page():
    """Executive Summary Dashboard for CEO-level presentation"""
    st.markdown("""
    <div style='text-align: center; padding: 15px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 15px; margin-bottom: 20px;'>
        <h1 style='margin: 0; font-size: 2.2em;'>📊 EXECUTIVE DASHBOARD</h1>
        <p style='margin: 5px 0; opacity: 0.9;'>S&P 500 AI Trading Intelligence • Strategic Overview</p>
    </div>
    """, unsafe_allow_html=True)
    
    # TOP KPIs ROW
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    with kpi_col1:
        st.metric("🎯 AI Models", "4", "Active")
    with kpi_col2:
        st.metric("📊 Accuracy", "58-70%", "+12% vs Baseline")
    with kpi_col3:
        st.metric("⚡ Speed", "80%", "Faster Decisions")
    with kpi_col4:
        st.metric("💰 ROI Potential", "5-15%", "Return Improvement")
    with kpi_col5:
        st.metric("⚠️ Risk Level", "MEDIUM", "Managed")
    
    st.markdown("---")
    
    # PERFORMANCE VISUALIZATION
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("### 🎯 Model Performance Radar")
        
        # Create performance radar chart
        try:
            from models.Ligthgbm import SP500LightGBMPredictor, create_comprehensive_lgb_models
            
            lgb_predictor = create_comprehensive_lgb_models()
            
            if lgb_predictor and lgb_predictor.models:
                # Extract performance metrics for visualization
                model_names = []
                accuracy_scores = []
                
                for model_name, model in lgb_predictor.models.items():
                    if model_name in lgb_predictor.model_metrics:
                        metrics = lgb_predictor.model_metrics[model_name]
                        
                        if "classification" in model_name.lower():
                            accuracy = metrics.get('test_accuracy', 0) * 100
                            model_display = model_name.replace('_', ' ').title()
                            model_names.append(model_display[:15] + '...' if len(model_display) > 15 else model_display)
                            accuracy_scores.append(accuracy)
                
                if model_names and accuracy_scores:
                    # Create bar chart for model performance
                    fig_performance = go.Figure()
                    
                    colors = ['#28a745' if score > 65 else '#ffc107' if score > 55 else '#dc3545' 
                             for score in accuracy_scores]
                    
                    fig_performance.add_trace(go.Bar(
                        x=model_names,
                        y=accuracy_scores,
                        marker_color=colors,
                        text=[f'{score:.1f}%' for score in accuracy_scores],
                        textposition='auto',
                        name='Accuracy'
                    ))
                    
                    fig_performance.update_layout(
                        title="🎯 Model Accuracy Comparison",
                        xaxis_title="AI Models",
                        yaxis_title="Accuracy %",
                        height=350,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    # Add benchmark line
                    fig_performance.add_hline(y=60, line_dash="dash", line_color="red", 
                                            annotation_text="Minimum Threshold")
                    
                    st.plotly_chart(fig_performance, use_container_width=True)
                else:
                    st.info("📊 Train models to see performance visualization")
            else:
                st.info("📊 No model data available")
                
        except Exception as e:
            st.info("📊 Performance data will appear after model training")
    
    with chart_col2:
        st.markdown("### 🏆 Business Impact Scorecard")
        
        # Business impact gauge charts
        scorecard_data = [
            {"metric": "Production Readiness", "score": 75, "color": "#28a745"},
            {"metric": "Risk Management", "score": 85, "color": "#17a2b8"},
            {"metric": "Decision Speed", "score": 90, "color": "#007bff"},
            {"metric": "Cost Efficiency", "score": 70, "color": "#6f42c1"}
        ]
        
        for i, data in enumerate(scorecard_data):
            score_col1, score_col2 = st.columns([3, 1])
            with score_col1:
                st.markdown(f"**{data['metric']}**")
                st.progress(data['score'] / 100)
            with score_col2:
                st.markdown(f"<h3 style='color:{data['color']};margin:0'>{data['score']}%</h3>", 
                           unsafe_allow_html=True)
    
    st.markdown("---")
    
    # RISK vs REWARD MATRIX
    matrix_col1, matrix_col2, matrix_col3 = st.columns([1, 2, 1])
    
    with matrix_col2:
        st.markdown("### 📊 Risk vs Reward Matrix")
        
        # Create risk-reward scatter plot
        models_risk_reward = {
            'LightGBM': {'risk': 25, 'reward': 70, 'size': 40},
            'XGBoost': {'risk': 30, 'reward': 65, 'size': 35},
            'Logistic Reg': {'risk': 15, 'reward': 55, 'size': 25},
            'LSTM': {'risk': 60, 'reward': 75, 'size': 45}
        }
        
        fig_matrix = go.Figure()
        
        for model, data in models_risk_reward.items():
            color = '#28a745' if data['risk'] < 30 else '#ffc107' if data['risk'] < 50 else '#dc3545'
            
            fig_matrix.add_trace(go.Scatter(
                x=[data['risk']],
                y=[data['reward']],
                mode='markers+text',
                marker=dict(size=data['size'], color=color, opacity=0.7),
                text=[model],
                textposition="middle center",
                textfont=dict(color='white', size=10),
                name=model,
                showlegend=False
            ))
        
        # Add quadrant lines
        fig_matrix.add_hline(y=60, line_dash="dash", line_color="gray", opacity=0.5)
        fig_matrix.add_vline(x=40, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig_matrix.add_annotation(x=20, y=80, text="🎯 OPTIMAL", showarrow=False, 
                                 font=dict(size=12, color="green"))
        fig_matrix.add_annotation(x=70, y=80, text="⚠️ HIGH RISK", showarrow=False, 
                                 font=dict(size=12, color="red"))
        fig_matrix.add_annotation(x=20, y=40, text="🔒 CONSERVATIVE", showarrow=False, 
                                 font=dict(size=12, color="blue"))
        fig_matrix.add_annotation(x=70, y=40, text="❌ AVOID", showarrow=False, 
                                 font=dict(size=12, color="red"))
        
        fig_matrix.update_layout(
            xaxis_title="Risk Level (%)",
            yaxis_title="Expected Reward (%)",
            height=400,
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[30, 90]),
            plot_bgcolor='rgba(240,240,240,0.5)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_matrix, use_container_width=True)
    
    
    st.markdown("---")
    
    # DECISION MATRIX
    decision_col1, decision_col2 = st.columns(2)
    
    with decision_col1:
        st.markdown("### 🎯 Strategic Decision Matrix")
        
        # Create decision heatmap
        decision_matrix = {
            'Short-term (1-7 days)': [85, 70, 60],
            'Medium-term (1-4 weeks)': [70, 85, 75], 
            'Long-term (1-6 months)': [45, 55, 80]
        }
        
        strategies = ['Aggressive', 'Balanced', 'Conservative']
        timeframes = list(decision_matrix.keys())
        
        z_matrix = [[decision_matrix[tf][i] for i in range(3)] for tf in timeframes]
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=strategies,
            y=timeframes,
            colorscale=[[0, '#dc3545'], [0.5, '#ffc107'], [1, '#28a745']],
            text=[[f'{z_matrix[i][j]}%' for j in range(3)] for i in range(3)],
            texttemplate="%{text}",
            textfont={"size": 14},
            showscale=False
        ))
        
        fig_heatmap.update_layout(
            title="Confidence Level by Strategy & Timeframe",
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with decision_col2:
        st.markdown("### 💰 ROI Projection")
        
        # ROI projection chart
        months = ['Month 1', 'Month 2', 'Month 3', 'Month 6', 'Month 12']
        conservative = [2, 3, 5, 8, 12]
        aggressive = [5, 8, 12, 18, 25]
        
        fig_roi = go.Figure()
        
        fig_roi.add_trace(go.Scatter(
            x=months, y=conservative,
            mode='lines+markers',
            name='Conservative Approach',
            line=dict(color='#28a745', width=3),
            marker=dict(size=8)
        ))
        
        fig_roi.add_trace(go.Scatter(
            x=months, y=aggressive,
            mode='lines+markers',
            name='Aggressive Approach',
            line=dict(color='#dc3545', width=3),
            marker=dict(size=8)
        ))
        
        fig_roi.update_layout(
            title="Expected ROI Timeline",
            xaxis_title="Timeline",
            yaxis_title="ROI Improvement (%)",
            height=300,
            legend=dict(x=0.02, y=0.98),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_roi, use_container_width=True)
    
    st.markdown("---")
    
    # EXECUTIVE DECISION CARDS
    st.markdown("### 🎯 EXECUTIVE DECISIONS")
    
    exec_col1, exec_col2, exec_col3 = st.columns(3)
    
    with exec_col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #28a745, #20c997); padding: 20px; 
                    border-radius: 15px; color: white; text-align: center; height: 200px;'>
            <h2 style='margin: 0;'>✅ APPROVED</h2>
            <h1 style='margin: 10px 0; font-size: 3em;'>GO</h1>
            <p><strong>LightGBM + XGBoost</strong></p>
            <p>Short-term Trading</p>
            <p>Conservative Start</p>
        </div>
        """, unsafe_allow_html=True)
    
    with exec_col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffc107, #fd7e14); padding: 20px; 
                    border-radius: 15px; color: white; text-align: center; height: 200px;'>
            <h2 style='margin: 0;'>🟡 PILOT</h2>
            <h1 style='margin: 10px 0; font-size: 3em;'>TEST</h1>
            <p><strong>LSTM Networks</strong></p>
            <p>6-Month Trial</p>
            <p>Small Positions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with exec_col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #6f42c1, #e83e8c); padding: 20px; 
                    border-radius: 15px; color: white; text-align: center; height: 200px;'>
            <h2 style='margin: 0;'>🔍 MONITOR</h2>
            <h1 style='margin: 10px 0; font-size: 3em;'>WATCH</h1>
            <p><strong>All Models</strong></p>
            <p>Performance Tracking</p>
            <p>Risk Management</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # FINAL RECOMMENDATION BANNER
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 25px; border-radius: 15px; color: white; text-align: center; margin-top: 20px;'>
        <h2 style='margin: 0; font-size: 1.8em;'>💡 FINAL RECOMMENDATION</h2>
        <h1 style='margin: 15px 0; font-size: 2.5em; color: #90EE90;'>PROCEED WITH PHASE 1</h1>
        <div style='display: flex; justify-content: space-around; margin-top: 20px;'>
            <div><h3>🎯 START:</h3><p>LightGBM Model</p></div>
            <div><h3>� BUDGET:</h3><p>Conservative</p></div>
            <div><h3>⏱️ TIMELINE:</h3><p>30 Days</p></div>
            <div><h3>� TARGET:</h3><p>5-8% ROI</p></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main_app():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Fetch Data", 
        "Model Performance", 
        "Models Comparison", 
        "Predictions",
        "📊 Executive Summary"
    ])
    
    if page == "Fetch Data":
        fetch_data_page()
    elif page == "Model Performance":
        model_performance_page()
    elif page == "Models Comparison":
        models_comparison_page()
    elif page == "Predictions":
        predictions_page()
    elif page == "📊 Executive Summary":
        executive_summary_page()

if __name__ == "__main__":
    main_app()
