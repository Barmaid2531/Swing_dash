# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

st.set_page_config(page_title="Advanced Swing Trading Dashboard", layout="wide")

# ---------------------------
# Helper Functions
# ---------------------------
@st.cache_data(ttl=60)
def get_omxspi_symbols():
    # Example list; ideally fetch dynamically from a reliable source
    return ["ERIC-B.ST","VOLV-B.ST","HM-B.ST","SEB-A.ST","TELIA.ST"]

@st.cache_data(ttl=60)
def fetch_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df.dropna(inplace=True)
    return df

def compute_indicators(df):
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta>0,0)).rolling(14).mean()
    loss = (-delta.where(delta<0,0)).rolling(14).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100/(1+RS))
    
    EMA12 = df['Close'].ewm(span=12, adjust=False).mean()
    EMA26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = EMA12 - EMA26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Upper'] = df['BB_Mid'] + 2*df['Close'].rolling(20).std()
    df['BB_Lower'] = df['BB_Mid'] - 2*df['Close'].rolling(20).std()
    
    return df

def generate_signals(df):
    df['Buy'] = ((df['Close'] > df['EMA10']) & 
                 (df['RSI'] < 40) & 
                 (df['MACD'] > df['Signal']))
    df['Sell'] = ((df['RSI'] > 65) | 
                  (df['MACD'] < df['Signal']))
    return df

def prepare_ml_features(df):
    df_ml = df.copy()
    df_ml['Pct_Change1'] = df_ml['Close'].pct_change(1)
    df_ml['Pct_Change3'] = df_ml['Close'].pct_change(3)
    df_ml['Pct_Change5'] = df_ml['Close'].pct_change(5)
    df_ml = df_ml.dropna()
    df_ml['Target'] = np.where(df_ml['Close'].shift(-5)/df_ml['Close'] -1 >= 0.02, 1, 0)
    features = ['Close', 'SMA10', 'SMA50', 'EMA10', 'EMA50', 'RSI', 'MACD', 'Signal',
                'BB_Upper','BB_Lower','Pct_Change1','Pct_Change3','Pct_Change5']
    X = df_ml[features]
    y = df_ml['Target']
    return X, y, df_ml

def backtest_signals(df):
    df_bt = df.copy()
    df_bt['Position'] = 0
    df_bt['Position'] = np.where(df_bt['Buy'],1,df_bt['Position'])
    df_bt['Position'] = np.where(df_bt['Sell'],0,df_bt['Position'])
    df_bt['Daily_Return'] = df_bt['Close'].pct_change()
    df_bt['Strategy_Return'] = df_bt['Position'].shift(1) * df_bt['Daily_Return']
    df_bt['Cumulative_Return'] = (1+df_bt['Strategy_Return']).cumprod()
    return df_bt

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("Settings")
asset_type = st.sidebar.selectbox("Asset Type", ["OMXSPI Stock", "Global ETF"])

if asset_type=="OMXSPI Stock":
    symbol_list = get_omxspi_symbols()
else:
    # Example global ETFs; extend as needed
    symbol_list = ["SPY","QQQ","VTI","IWM","EFA","EEM"]

symbol_input = st.sidebar.selectbox("Select Symbol", symbol_list)
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())
refresh_interval = st.sidebar.number_input("Refresh interval (minutes)", min_value=1, max_value=60, value=5)

# ---------------------------
# Fetch Data
# ---------------------------
data = fetch_data(symbol_input, start_date, end_date)
if data.empty:
    st.error("No data found. Check symbol.")
    st.stop()

# ---------------------------
# Indicators and Signals
# ---------------------------
data = compute_indicators(data)
data = generate_signals(data)
X, y, df_ml = prepare_ml_features(data)

# ML Model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
latest_prob = rf.predict_proba(X_scaled[-1].reshape(1,-1))[0][1]

# Backtest
bt_df = backtest_signals(data)

# ---------------------------
# Dashboard
# ---------------------------
st.title(f"Advanced Swing Trading Dashboard: {symbol_input}")

# Latest Indicators
st.subheader("Latest Indicators & Signals")
latest = data.iloc[-1][['Close','SMA10','SMA50','EMA10','EMA50','RSI','MACD','Signal','BB_Upper','BB_Lower','Buy','Sell']]
st.dataframe(latest.to_frame().T.style.applymap(lambda x: 'background-color: lightgreen' if x==True else '', subset=['Buy'])
                                 .applymap(lambda x: 'background-color: tomato' if x==True else '', subset=['Sell']))

# ML Prediction
st.subheader("ML Short-Term Gain Prediction (Next 3-5 Days)")
st.metric(label="Probability of Gain", value=f"{latest_prob*100:.1f}%")

# Price Chart with Indicators & Buy/Sell Markers
st.subheader("Price Chart with Indicators")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price"))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA10'], line=dict(color='blue', width=1), name='SMA10'))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], line=dict(color='purple', width=1), name='SMA50'))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA10'], line=dict(color='green', width=1), name='EMA10'))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA50'], line=dict(color='orange', width=1), name='EMA50'))
fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], line=dict(color='red', width=1, dash='dash'), name='BB Upper'))
fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], line=dict(color='red', width=1, dash='dash'), name='BB Lower'))
# Add Buy/Sell markers
buys = data[data['Buy']]
sells = data[data['Sell']]
fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy'))
fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell'))
fig.update_layout(height=600, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# Signals Table
st.subheader("Recent Buy/Sell Signals")
signals_df = data[['Buy','Sell']].copy()
signals_df = signals_df[(signals_df['Buy']==True) | (signals_df['Sell']==True)]
st.dataframe(signals_df.tail(20))

# Backtest Cumulative Returns
st.subheader("Strategy Backtest: Cumulative Returns")
fig_bt = go.Figure()
fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Cumulative_Return'], mode='lines', name='Strategy Cumulative Return'))
fig_bt.add_trace(go.Scatter(x=bt_df.index, y=(1+bt_df['Daily_Return']).cumprod(), mode='lines', name='Buy & Hold'))
st.plotly_chart(fig_bt, use_container_width=True)

# Alerts: Combined Signals
st.subheader("Alerts")
if latest['Buy'] and latest_prob>0.6:
    st.success(f"Strong Buy Signal! Technical + ML aligned. Probability of gain: {latest_prob*100:.1f}%")
elif latest['Sell'] and latest_prob<0.4:
    st.error(f"Strong Sell Signal! Technical + ML aligned. Probability of gain: {latest_prob*100:.1f}%")
else:
    st.info("No strong combined signals at the moment.")

st.info(f"Dashboard refreshes every {refresh_interval} minutes with live data.")
