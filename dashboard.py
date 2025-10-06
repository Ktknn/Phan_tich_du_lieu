import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import altair as alt
from model import *
import matplotlib.pyplot as plt
import ta
import plotly.graph_objects as go

# -----------------Page Config-------------------------------
st.set_page_config(layout="wide")


# -----------------------------------------------------------

# Fetch stock data based on the ticker, period, and interval
def fetch_stock_data(ticker, period, interval):
    end_date = datetime.now()
    if period == '1wk':
        start_date = end_date - timedelta(days=7)
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    else:
        data = yf.download(ticker, period=period, interval=interval)
    return data


# Process data to ensure it is timezone-aware and has the correct format
def process_data(data):
    data.rename(columns={'Date': 'Datetime'}, inplace=True)
    return data


# Calculate basic metrics from the stock data
def calculate_metrics(data):
    last_close = data['Close'].iloc[-1][-1]
    prev_close = data['Close'].iloc[0][-1]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    high = data['High'].max()[-1]
    low = data['Low'].min()[-1]
    volume = data['Volume'].sum()[-1]
    return last_close, change, pct_change, high, low, volume


# Sidebar for user input parameters
st.sidebar.header('Ticker & Model Selection')
ticker = st.sidebar.selectbox('Ticker', ["AAPL", "CRM", "EPAM", "GOOGL", "HPQ", "KLAC", "MSI", "NVDA", "TDY", "WDC"])
time_period = st.sidebar.selectbox('Time Period', ['1wk', '1mo', '3m', '1y'])
model_name = st.sidebar.selectbox('Model Name', ['Moving Average', 'Linear Regression', "K-Nearest Neighbors", "LSTM"])
selected_data = st.sidebar.button('Data visualization')

# Mapping of time periods to data intervals
interval_mapping = {
    '1wk': 7,
    '1mo': 30,
    '3m': 30 * 3,
    '1y': 365
}

model_name_mapping = {
    "Moving Average": "MA",
    "Linear Regression": "LR",
    "K-Nearest Neighbors": "KN",
    "LSTM": "LSTM"
}
# Set up Streamlit page layout
st.title('Stock Prediction Dashboard')
st.markdown(f"## {ticker}")
if True:
    data = fetch_stock_data(ticker, "1wk", "30m")
    data = process_data(data)

    last_close, change, pct_change, high, low, volume = calculate_metrics(data)

    # Display main metrics
    st.metric(label=f"{ticker} Last Price", value=f"{last_close:.2f} USD", delta=f"{change:.2f} ({pct_change:.2f}%)")

    col1, col2, col3 = st.columns(3)
    col1.metric("High", f"{high:.2f} USD")
    col2.metric("Low", f"{low:.2f} USD")
    col3.metric("Volume", f"{volume:,}")

# Sidebar section for real-time stock prices of selected symbols
st.sidebar.header('Real-Time Stock Prices')
stock_symbols = ["AAPL", "CRM", "EPAM", "GOOGL", "HPQ", "KLAC", "MSI", "NVDA", "TDY", "WDC"]
for symbol in stock_symbols:
    real_time_data = fetch_stock_data(symbol, '1d', '1m')
    if not real_time_data.empty:
        real_time_data = process_data(real_time_data)
        last_price = real_time_data['Close'].iloc[-1][-1]
        change = last_price - real_time_data['Open'].iloc[0][-1]
        pct_change = (change / real_time_data['Open'].iloc[0][-1]) * 100
        st.sidebar.metric(f"{symbol}", f"{last_price:.2f} USD", f"{change:.2f} ({pct_change:.2f}%)")

col1, col2 = st.columns([3, 1])
with col1:
    # --------------Load Model--------------------
    model_match = {
        "Moving Average": MAModel(ticker, model_dir=r"models", data_dir=r"data"),
        "Linear Regression": LRModel(ticker, model_dir="models", data_dir="data"),
        "K-Nearest Neighbors": KNNModel(ticker, model_dir="models", data_dir="data"),
        "LSTM": LSTMModel(ticker, model_dir="models", data_dir="data"),
    }
    model = model_match[model_name]
    predictions = model.stock_prediction(interval_mapping[time_period]+3)

    # --------------Predict Chart-----------------
    df = pd.read_csv(f"./data/{ticker}.csv")
    df = df[["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]]

    df['Date'] = pd.to_datetime(df['Date'])  # Chuyển về kiểu datetime
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    new_dates = df["Date"][-interval_mapping[time_period]:].to_list()

    df["Type"] = "Actual"
    # Tạo danh sách ngày mới
    df['Date'] = df['Date'].dt.strftime('%Y/%m/%d')

    new_dates += [(last_date + timedelta(days=i)).strftime('%Y/%m/%d') for i in
                 range(1, 3 + 1)]

    # Tạo DataFrame mới cho dự đoán
    predictions_df = pd.DataFrame({
        "Date": new_dates,
        "Adj Close": predictions,
        "Type": "Prediction"
    })

    # Gộp dữ liệu
    df = pd.concat([df, predictions_df], ignore_index=True)

    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('Date:T', title='Date', axis=alt.Axis(labelAngle=-45, format='%b %Y')),  # Trục x
        y=alt.Y('Adj Close:Q', title='Price (USD)'),
        color=alt.Color('Type:N', title='Type',
                        scale=alt.Scale(domain=['Actual', 'Prediction'],
                                        range=['blue', 'orange'])),  # Màu sắc khác nhau
        tooltip=['Date:T', 'Type:N', 'Adj Close:Q']  # Trục y
    ).properties(
        title="Stock Price Over Time",
        width=700,  # Chiều rộng
        height=500  # Chiều cao
    ).configure_title(
        fontSize=20,  # Kích thước tiêu đề
        anchor='start',
        color='gray'
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

# ----------------------------------------------------------------------------
with col2:
    st.html('<span class="column_indicator"></span>')
    st.subheader("Model Metrics")
    l, r = st.columns(2)
    with l:
        st.html('<span class="low_indicator"></span>')
        st.metric("Modle Name", model_name_mapping[model_name])
        st.metric("RMSE", round(model.model_static["rmse"], 2))
    with r:
        st.html('<span class="high_indicator"></span>')
        st.metric("MAE", round(model.model_static["mae"], 2))
        st.metric("R2", round(model.model_static["r2"], 2))


# if selected_data:
#     st.title('Data visualization')
#     # Tạo figure lớn cho 4 biểu đồ
#     fig, axs = plt.subplots(2, 2, figsize=(8, 6))
#
#     # Biểu đồ giá đóng cửa
#     axs[0, 0].plot(df['Close'], color='blue')
#     axs[0, 0].set_title('Closing Price Over Time')
#     axs[0, 0].set_xlabel('Date')
#     axs[0, 0].set_ylabel('Close Price')
#
#         # Biểu đồ khối lượng giao dịch
#     axs[0, 1].bar(df.index, df['Volume'], color='orange')
#     axs[0, 1].set_title('Trading Volume Over Time')
#     axs[0, 1].set_xlabel('Date')
#     axs[0, 1].set_ylabel('Volume')
#
#     # Tính và vẽ trung bình động
#     ma100 = df['Close'].rolling(window=100).mean()
#     ma200 = df['Close'].rolling(window=200).mean()
#
#     # Biểu đồ với cả hai trung bình động
#     axs[1, 0].plot(df['Close'], 'r', label="Daily Closing Price")
#     axs[1, 0].plot(ma100, 'g', label="100-Day Moving Average")
#     axs[1, 0].plot(ma200, 'b', label="200-Day Moving Average")
#     axs[1, 0].set_title('Closing Price with Moving Averages')
#     axs[1, 0].set_xlabel('Date')
#     axs[1, 0].set_ylabel('Close Price')
#     axs[1, 0].legend()
#
#     # Thêm biểu đồ tỷ lệ phần trăm thay đổi giá
#     df['Pct Change'] = df['Close'].pct_change() * 100  # Tính tỷ lệ phần trăm thay đổi
#     axs[1, 1].plot(df['Pct Change'], label='Percentage Change', color='purple')
#     axs[1, 1].axhline(0, color='black', linewidth=0.8, linestyle='--')  # Dòng ngang tại 0%
#     axs[1, 1].set_title('Percentage Change in Closing Price Over Time')
#     axs[1, 1].set_xlabel('Date')
#     axs[1, 1].set_ylabel('Percentage Change (%)')
#     axs[1, 1].legend()
#
#     # Điều chỉnh khoảng cách giữa các biểu đồ
#     plt.tight_layout()
#
#     # Hiển thị biểu đồ trong Streamlit
#     st.pyplot(fig)
#

def create_candlestick_chart(data, ticker):
    data = process_data(data)
    # Plot the stock price chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data['Datetime'],
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close']))

    # Add selected technical indicators to the chart
    fig.add_trace(go.Scatter(x=data['Datetime'], y=data['SMA_20'], name='SMA 20'))
    fig.add_trace(go.Scatter(x=data['Datetime'], y=data['EMA_20'], name='EMA 20'))


    # Format graph
    fig.update_layout(title=f'{ticker} 1 YEAR Chart',
                      xaxis_title='Time',
                      yaxis_title='Price (USD)',
                      height=600)
    st.plotly_chart(fig, use_container_width=True)
    return fig


def add_technical_indicators(data):
    """
    Thêm SMA_20 và EMA_20 vào DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame chứa các cột 'Close'.

    Returns:
        pd.DataFrame: DataFrame với SMA_20 và EMA_20.
    """
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    return data


plot_data = pd.read_csv(f"./data/{ticker}.csv")
plot_data = plot_data[-365:]
plot_data = add_technical_indicators(plot_data)
create_candlestick_chart(plot_data, ticker)
