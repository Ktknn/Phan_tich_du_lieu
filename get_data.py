import yfinance as yf
import pandas as pd
import os
import datetime as dt

tickers = ["AAPL", "CRM", "EPAM", "GOOGL", "HPQ", "KLAC", "MSI", "NVDA", "TDY", "WDC"]
start = dt.datetime.today() - dt.timedelta(5 * 365)
end = dt.datetime.today()

# Tạo thư mục lưu trữ (nếu chưa có)
base_dir = os.getcwd()  # Thư mục hiện tại
data_dir = os.path.join(base_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

for i in tickers:
    data = yf.download(i, start, end)
    data.reset_index(inplace=True)
    csv_file_path = os.path.join(data_dir, f'{i}.csv')  # Đường dẫn lưu file CSV
    data.to_csv(csv_file_path, index=True)

print(f"Data has been saved to {data_dir}")