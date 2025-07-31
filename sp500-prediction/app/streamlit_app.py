"""
Streamlit Dashboard for S&P 500 Prediction System
Interactive web dashboard for visualizing predictions and market data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import sys
import os
from datetime import datetime, timedelta

# Add src directory to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from predict import SP500Predictor
    from data_collection import SP500DataCollector
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="S&P 500 Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-up {
        color: #00ff00;
        font-weight: bold;
    }
    .prediction-down {
        color: #ff0000;
        font-weight: bold;
    }
    .sidebar-content {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_market_data(period="3mo"):
    """Get market data with caching"""
    try:
        # Use absolute paths for data
        base_dir = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(base_dir, "data")
        
        collector = SP500DataCollector(data_dir=data_dir)
        sp500_data = collector.fetch_sp500_data(period=period)
        vix_data = collector.fetch_vix_data(period=period)
        return sp500_data, vix_data
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        return None, None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_prediction():
    """Get prediction with caching"""
    try:
        # Use absolute paths for models and data
        base_dir = os.path.dirname(os.path.dirname(__file__))
        models_dir = os.path.join(base_dir, "models")
        data_dir = os.path.join(base_dir, "data")
        
        predictor = SP500Predictor(models_dir=models_dir, data_dir=data_dir)
        prediction = predictor.make_prediction()
        return prediction
    except Exception as e:
        st.error(f"Error generating prediction: {e}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_future_predictions(days_ahead=5):
    """Get predictions for multiple days ahead"""
    try:
        # Use absolute paths for models and data
        base_dir = os.path.dirname(os.path.dirname(__file__))
        models_dir = os.path.join(base_dir, "models")
        data_dir = os.path.join(base_dir, "data")
        
        predictor = SP500Predictor(models_dir=models_dir, data_dir=data_dir)
        
        # Load models and get latest data
        if not predictor.load_models():
            return None
            
        sp500_data, vix_data = predictor.get_latest_market_data()
        if sp500_data is None or vix_data is None:
            return None
        
        # Ensure timezone-naive dates for consistent processing
        if 'date' in sp500_data.columns:
            sp500_data['date'] = pd.to_datetime(sp500_data['date'])
            if hasattr(sp500_data['date'].iloc[0], 'tz') and sp500_data['date'].iloc[0].tz is not None:
                sp500_data['date'] = sp500_data['date'].dt.tz_localize(None)
        
        if 'date' in vix_data.columns:
            vix_data['date'] = pd.to_datetime(vix_data['date'])
            if hasattr(vix_data['date'].iloc[0], 'tz') and vix_data['date'].iloc[0].tz is not None:
                vix_data['date'] = vix_data['date'].dt.tz_localize(None)
            
        predictions_list = []
        current_data = sp500_data.copy()
        
        for day in range(1, days_ahead + 1):
            # Engineer features for current data
            features_data = predictor.engineer_features_for_prediction(current_data, vix_data)
            if features_data is None:
                break
                
            # Get the latest record for prediction
            latest_data = features_data.iloc[-1:].copy()
            
            # Select only the features used in training
            feature_columns = []
            for feature in predictor.feature_names:
                if feature in latest_data.columns:
                    feature_columns.append(feature)
            
            # Prepare feature vector
            X = latest_data[feature_columns].copy()
            
            # Handle missing features
            if len(feature_columns) < len(predictor.feature_names):
                missing_features = set(predictor.feature_names) - set(feature_columns)
                for feature in missing_features:
                    X[feature] = 0
            
            # Reorder columns to match training order
            X = X[predictor.feature_names]
            X = X.fillna(0).replace([np.inf, -np.inf], 0)
            
            # Scale features
            X_scaled = predictor.scaler.transform(X.values)
            
            # Make predictions with all models
            predictions = {}
            for name, model in predictor.models.items():
                pred_proba = model.predict_proba(X_scaled)[0]
                pred_class = model.predict(X_scaled)[0]
                
                predictions[name] = {
                    'prediction': int(pred_class),
                    'probability_down': float(pred_proba[0]),
                    'probability_up': float(pred_proba[1]),
                    'confidence': float(max(pred_proba))
                }
            
            # Ensemble prediction
            ensemble_votes = [pred['prediction'] for pred in predictions.values()]
            ensemble_prediction = int(np.round(np.mean(ensemble_votes)))
            ensemble_confidence = np.mean([pred['confidence'] for pred in predictions.values()])
            
            # Create future date (skip weekends)
            last_date = pd.to_datetime(current_data['date'].iloc[-1])
            
            # Ensure we're working with a naive datetime (remove timezone if present)
            if hasattr(last_date, 'tz') and last_date.tz is not None:
                last_date = last_date.tz_localize(None)
            
            # For each prediction day, calculate the future date
            future_date = last_date + pd.Timedelta(days=day)
            
            # Skip weekends
            while future_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                future_date += pd.Timedelta(days=1)
            
            # Estimate future price based on prediction and current volatility
            current_price = float(current_data['close'].iloc[-1])
            volatility = current_data['close'].pct_change().std() * np.sqrt(252)  # Annualized volatility
            
            # Simple price estimation: apply random walk with bias based on prediction
            if ensemble_prediction == 1:  # Up prediction
                price_change = np.random.normal(0.001, volatility/252)  # Slight upward bias
            else:  # Down prediction
                price_change = np.random.normal(-0.001, volatility/252)  # Slight downward bias
            
            estimated_price = current_price * (1 + price_change)
            
            prediction_result = {
                'day': day,
                'date': future_date,
                'estimated_price': estimated_price,
                'direction': 'UP' if ensemble_prediction == 1 else 'DOWN',
                'ensemble_prediction': ensemble_prediction,
                'ensemble_confidence': float(ensemble_confidence),
                'individual_predictions': predictions
            }
            
            predictions_list.append(prediction_result)
            
            # Update current_data for next iteration (simulate adding the predicted day)
            new_row = current_data.iloc[-1:].copy()
            new_row['date'] = future_date
            new_row['close'] = estimated_price
            new_row['open'] = current_price
            new_row['high'] = max(current_price, estimated_price) * 1.01
            new_row['low'] = min(current_price, estimated_price) * 0.99
            new_row['volume'] = current_data['volume'].iloc[-1]
            
            # Ensure the new row date is timezone-naive to match current_data
            if hasattr(new_row['date'].iloc[0], 'tz') and new_row['date'].iloc[0].tz is not None:
                new_row['date'] = new_row['date'].dt.tz_localize(None)
            
            current_data = pd.concat([current_data, new_row], ignore_index=True)
        
        return predictions_list
        
    except Exception as e:
        st.error(f"Error generating future predictions: {e}")
        return None

def create_price_chart(data, future_predictions=None, title="S&P 500 Price Chart"):
    """Create interactive price chart with future predictions"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data['date'],
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name="S&P 500",
        increasing_line_color='green',
        decreasing_line_color='red'
    ))
    
    # Add moving averages
    if 'sma_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['close'].rolling(window=20).mean(),
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=1)
        ))
    
    if 'sma_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['close'].rolling(window=50).mean(),
            mode='lines',
            name='SMA 50',
            line=dict(color='blue', width=1)
        ))
    
    # Add future predictions if provided
    if future_predictions:
        # Extract prediction data
        future_dates = [pred['date'] for pred in future_predictions]
        future_prices = [pred['estimated_price'] for pred in future_predictions]
        future_directions = [pred['direction'] for pred in future_predictions]
        future_confidences = [pred['ensemble_confidence'] for pred in future_predictions]
        
        # Add prediction line
        current_price = data['close'].iloc[-1]
        current_date = data['date'].iloc[-1]
        
        # Connect current price to future predictions
        prediction_dates = [current_date] + future_dates
        prediction_prices = [current_price] + future_prices
        
        fig.add_trace(go.Scatter(
            x=prediction_dates,
            y=prediction_prices,
            mode='lines+markers',
            name='Predicted Path',
            line=dict(color='purple', width=3, dash='dash'),
            marker=dict(size=8, color='purple')
        ))
        
        # Add prediction markers with directional colors and confidence
        for i, pred in enumerate(future_predictions):
            color = 'green' if pred['direction'] == 'UP' else 'red'
            symbol = 'triangle-up' if pred['direction'] == 'UP' else 'triangle-down'
            
            fig.add_trace(go.Scatter(
                x=[pred['date']],
                y=[pred['estimated_price']],
                mode='markers',
                name=f"Day {pred['day']} ({pred['direction']})",
                marker=dict(
                    size=15,
                    color=color,
                    symbol=symbol,
                    line=dict(width=2, color='white')
                ),
                hovertemplate=f"<b>Day {pred['day']} Prediction</b><br>" +
                             f"Date: {pred['date'].strftime('%Y-%m-%d')}<br>" +
                             f"Direction: {pred['direction']}<br>" +
                             f"Confidence: {pred['ensemble_confidence']:.1%}<br>" +
                             f"Est. Price: ${pred['estimated_price']:.2f}<extra></extra>",
                showlegend=False
            ))
        
        # Add confidence bands
        upper_band = [price * (1 + 0.02) for price in prediction_prices[1:]]  # 2% upper band
        lower_band = [price * (1 - 0.02) for price in prediction_prices[1:]]  # 2% lower band
        
        fig.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=upper_band + lower_band[::-1],
            fill='toself',
            fillcolor='rgba(128, 0, 128, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Prediction Uncertainty',
            hoverinfo="skip"
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_future_predictions_timeline(future_predictions):
    """Create a timeline chart for future predictions"""
    if not future_predictions:
        return None
    
    fig = go.Figure()
    
    # Extract data
    days = [pred['day'] for pred in future_predictions]
    # Use full date format to show the actual year (2025)
    dates = [pred['date'].strftime('%Y-%m-%d') for pred in future_predictions]
    directions = [pred['direction'] for pred in future_predictions]
    confidences = [pred['ensemble_confidence'] for pred in future_predictions]
    prices = [pred['estimated_price'] for pred in future_predictions]
    
    # Create bar chart for confidence levels
    colors = ['green' if direction == 'UP' else 'red' for direction in directions]
    
    fig.add_trace(go.Bar(
        x=dates,
        y=confidences,
        marker_color=colors,
        name='Prediction Confidence',
        text=[f"{direction}<br>{conf:.1%}" for direction, conf in zip(directions, confidences)],
        textposition='auto',
        hovertemplate="<b>%{x}</b><br>" +
                     "Confidence: %{y:.1%}<br>" +
                     "Est. Price: $%{customdata:.2f}<extra></extra>",
        customdata=prices
    ))
    
    # Get the current year from the first prediction
    current_year = future_predictions[0]['date'].year if future_predictions else 2025
    
    fig.update_layout(
        title=f"5-Day Prediction Timeline ({current_year})",
        xaxis_title="Date",
        yaxis_title="Confidence Level",
        template="plotly_white",
        height=400,
        yaxis=dict(tickformat='.0%'),
        showlegend=False,
        xaxis=dict(
            tickangle=45,  # Rotate dates for better readability
            type='category'  # Ensure dates are treated as categories
        )
    )
    
    return fig

def create_prediction_summary_table(future_predictions):
    """Create a summary table for future predictions"""
    if not future_predictions:
        return None
    
    summary_data = []
    for pred in future_predictions:
        summary_data.append({
            'Day': f"Day {pred['day']}",
            'Date': pred['date'].strftime('%Y-%m-%d'),
            'Direction': pred['direction'],
            'Confidence': f"{pred['ensemble_confidence']:.1%}",
            'Est. Price': f"${pred['estimated_price']:.2f}",
            'Models Agreement': f"{sum([p['prediction'] for p in pred['individual_predictions'].values()])}/3"
        })
    
    return pd.DataFrame(summary_data)

def create_technical_indicators_chart(data):
    """Create technical indicators chart"""
    # Calculate RSI and MACD if not present
    if 'rsi' not in data.columns:
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
    
    if 'macd' not in data.columns:
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        data['macd'] = ema_12 - ema_26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('RSI', 'MACD'),
        vertical_spacing=0.15,
        shared_xaxes=True
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['rsi'], mode='lines', name='RSI', line=dict(color='purple')),
        row=1, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    
    # MACD
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['macd'], mode='lines', name='MACD', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['macd_signal'], mode='lines', name='Signal', line=dict(color='red')),
        row=2, col=1
    )
    
    fig.update_layout(
        height=400,
        template="plotly_white",
        showlegend=True
    )
    
    return fig

def create_volume_chart(data):
    """Create volume chart"""
    fig = go.Figure()
    
    # Volume bars
    colors = ['green' if close >= open else 'red' for close, open in zip(data['close'], data['open'])]
    
    fig.add_trace(go.Bar(
        x=data['date'],
        y=data['volume'],
        marker_color=colors,
        name='Volume',
        opacity=0.7
    ))
    
    # Volume moving average
    volume_ma = data['volume'].rolling(window=20).mean()
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=volume_ma,
        mode='lines',
        name='Volume MA(20)',
        line=dict(color='orange', width=2)
    ))
    
    fig.update_layout(
        title="Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        template="plotly_white",
        height=300
    )
    
    return fig

def main():
    """Main dashboard function"""
    
    # Header
    st.title("📈 S&P 500 Prediction Dashboard")
    st.markdown("Real-time predictions and market analysis for the S&P 500 index")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Time period selection
        period = st.selectbox(
            "📅 Select Time Period",
            options=["1mo", "3mo", "6mo", "1y"],
            index=1
        )
        
        # Future predictions toggle
        show_predictions = st.checkbox("🔮 Show Future Predictions", value=True)
        
        # Number of prediction days
        if show_predictions:
            prediction_days = st.slider(
                "📊 Prediction Days",
                min_value=1,
                max_value=5,
                value=5,
                help="Number of days to predict ahead"
            )
        else:
            prediction_days = 5  # Default value when predictions are disabled
        
        # Model selection for prediction
        model_choice = st.selectbox(
            "🤖 Select Model",
            options=["Ensemble", "Random Forest", "Gradient Boosting", "Logistic Regression"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📊 Dashboard Info")
        st.info(
            "This dashboard provides real-time S&P 500 predictions using machine learning models. "
            "Data is updated every 5 minutes during market hours."
        )
    
    # Main content
    col1, col2, col3 = st.columns([2, 1, 1])
    
    # Get prediction
    prediction = get_prediction()
    
    # Get future predictions if enabled
    future_predictions = None
    if show_predictions:
        future_predictions = get_future_predictions(prediction_days)
    
    if prediction:
        with col1:
            st.subheader("🎯 Today's Prediction")
            
            prediction_text = prediction['prediction_text']
            confidence = prediction['ensemble_confidence']
            
            if prediction_text == "UP":
                st.markdown(f"<h2 class='prediction-up'>📈 {prediction_text}</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 class='prediction-down'>📉 {prediction_text}</h2>", unsafe_allow_html=True)
            
            st.metric(
                label="Confidence Level",
                value=f"{confidence:.1%}",
                delta=None
            )
        
        with col2:
            st.subheader("💰 Current Price")
            st.metric(
                label="S&P 500",
                value=f"${prediction['current_price']:.2f}",
                delta=None
            )
            
            st.metric(
                label="Last Updated",
                value=prediction['date'],
                delta=None
            )
        
        with col3:
            st.subheader("🤖 Model Votes")
            for model_name, pred in prediction['individual_predictions'].items():
                direction = "📈" if pred['prediction'] == 1 else "📉"
                st.write(f"{direction} {model_name.replace('_', ' ').title()}: {pred['confidence']:.1%}")
    else:
        st.error("Unable to generate prediction. Please check if models are trained.")
    
    # Future Predictions Section
    if show_predictions and future_predictions:
        st.markdown("---")
        st.subheader("🔮 Future Predictions")
        
        # Create columns for future predictions display
        pred_col1, pred_col2 = st.columns([2, 1])
        
        with pred_col1:
            # Future predictions timeline
            timeline_chart = create_future_predictions_timeline(future_predictions)
            if timeline_chart:
                st.plotly_chart(timeline_chart, use_container_width=True)
        
        with pred_col2:
            # Summary metrics
            st.subheader("📊 Prediction Summary")
            
            # Overall trend
            up_count = sum(1 for pred in future_predictions if pred['direction'] == 'UP')
            down_count = len(future_predictions) - up_count
            
            if up_count > down_count:
                st.markdown("**Overall Trend: 📈 BULLISH**")
            elif down_count > up_count:
                st.markdown("**Overall Trend: 📉 BEARISH**")
            else:
                st.markdown("**Overall Trend: ➡️ NEUTRAL**")
            
            # Average confidence
            avg_confidence = np.mean([pred['ensemble_confidence'] for pred in future_predictions])
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            # Price range
            prices = [pred['estimated_price'] for pred in future_predictions]
            st.metric("Est. Price Range", f"${min(prices):.2f} - ${max(prices):.2f}")
        
        # Detailed prediction table
        st.subheader("📋 Detailed Predictions")
        summary_table = create_prediction_summary_table(future_predictions)
        if summary_table is not None:
            st.dataframe(summary_table, use_container_width=True)
    
    st.markdown("---")
    
    # Get market data
    sp500_data, vix_data = get_market_data(period)
    
    if sp500_data is not None:
        # Price Chart with Future Predictions
        st.subheader("📊 Price Chart with Predictions")
        
        # Update chart title based on predictions
        chart_title = "S&P 500 Price Chart"
        if show_predictions and future_predictions:
            chart_title += f" with {len(future_predictions)}-Day Predictions"
        
        price_chart = create_price_chart(sp500_data, future_predictions, chart_title)
        st.plotly_chart(price_chart, use_container_width=True)
        
        # Technical Indicators and Volume
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Technical Indicators")
            tech_chart = create_technical_indicators_chart(sp500_data.copy())
            st.plotly_chart(tech_chart, use_container_width=True)
        
        with col2:
            st.subheader("📊 Trading Volume")
            volume_chart = create_volume_chart(sp500_data)
            st.plotly_chart(volume_chart, use_container_width=True)
        
        # Market Statistics
        st.subheader("📋 Market Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = sp500_data['close'].iloc[-1]
        previous_close = sp500_data['close'].iloc[-2]
        daily_change = current_price - previous_close
        daily_change_pct = (daily_change / previous_close) * 100
        
        with col1:
            st.metric(
                label="Daily Change",
                value=f"${daily_change:.2f}",
                delta=f"{daily_change_pct:.2f}%"
            )
        
        with col2:
            st.metric(
                label="52W High",
                value=f"${sp500_data['high'].max():.2f}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="52W Low",
                value=f"${sp500_data['low'].min():.2f}",
                delta=None
            )
        
        with col4:
            avg_volume = sp500_data['volume'].mean()
            st.metric(
                label="Avg Volume",
                value=f"{avg_volume/1e6:.1f}M",
                delta=None
            )
        
        # VIX Data
        if vix_data is not None:
            st.subheader("😰 VIX (Fear Index)")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=vix_data['date'],
                y=vix_data['vix_close'],
                mode='lines',
                name='VIX',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="VIX (Volatility Index)",
                xaxis_title="Date",
                yaxis_title="VIX Level",
                template="plotly_white",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Raw Data
        with st.expander("📊 View Raw Data"):
            st.write("**S&P 500 Data (Last 10 days)**")
            st.dataframe(sp500_data.tail(10))
            
            if vix_data is not None:
                st.write("**VIX Data (Last 10 days)**")
                st.dataframe(vix_data.tail(10))
    
    else:
        st.error("Unable to fetch market data. Please check your internet connection.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Disclaimer:** This dashboard is for educational purposes only. "
        "Do not use these predictions for actual trading decisions. "
        "Always consult with a financial advisor before making investment decisions."
    )

if __name__ == "__main__":
    main()
