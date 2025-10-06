import datetime as dt
import json
import os
import pickle

import pandas as pd
from keras.models import load_model

from utils import *

class MAModel:
    def __init__(self, stock_name, data_dir="..\data", model_dir="..\models"):
        self.stock_name = stock_name
        self.data_dir = data_dir
        self.model_dir = os.path.join(model_dir, self.stock_name)
        self.model_static = None
        self.model = None

        self._load_model()

    def _load_model(self):
        model_path = os.path.join(self.model_dir, "MA.pkl")
        model_static_path = os.path.join(self.model_dir, "model_loss.json")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            self.model = model

        with open(model_static_path, 'r') as f:
            model_static = json.load(f)
            self.model_static = model_static["MA"]

    def stock_prediction(self, time_range=1):
        return self.model.predict([0 for _ in range(time_range)])


class LRModel:
    def __init__(self, stock_name, data_dir="..\data", model_dir="..\models"):
        self.stock_name = stock_name
        self.data_dir = os.path.join(data_dir, self.stock_name+".csv")
        self.model_dir = os.path.join(model_dir, self.stock_name)
        self.model = None
        self.model_static = None

        self._load_model()

    def _load_model(self):
        model_path = os.path.join(self.model_dir, "LR.pkl")
        model_static_path = os.path.join(self.model_dir, "model_loss.json")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            self.model = model

        with open(model_static_path, 'r') as f:
            model_static = json.load(f)
            self.model_static = model_static["LR"]

    def _process_df(self, df):
        df['Date'] = pd.to_datetime(df['Date'])

        # Remove timezone information before conversion
        df['Date'] = df['Date'].dt.tz_localize(None).astype('datetime64[ns]')

        # Perform other operations after conversion
        df["Year"] = df['Date'].dt.year
        df["Month"] = df['Date'].dt.month
        df["Day"] = df['Date'].dt.day
        df["DayOfWeek"] = df['Date'].dt.dayofweek
        df["DayOfYear"] = df['Date'].dt.dayofyear

        df.drop('Date', axis=1, inplace=True)

        return df

    def stock_prediction(self, time_range=1):
        stock_df = pd.read_csv(self.data_dir)
        date = stock_df["Date"]
        df = pd.DataFrame(columns=["Date"])
        df["Date"] = date[-time_range:]
        df = self._process_df(df)

        preds = self.model.predict(df)
        return preds


class KNNModel:
    def __init__(self, stock_name, data_dir="..\data", model_dir="..\models"):
        self.stock_name = stock_name
        self.data_dir = os.path.join(data_dir, self.stock_name+".csv")
        self.model_dir = os.path.join(model_dir, self.stock_name)
        self.model = None
        self.model_static = None

        self._load_model()

    def _load_model(self):
        model_path = os.path.join(self.model_dir, "KNN.pkl")
        model_static_path = os.path.join(self.model_dir, "model_loss.json")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            self.model = model

        with open(model_static_path, 'r') as f:
            model_static = json.load(f)
            self.model_static = model_static["KNN"]

    def _process_df(self, df):
        df['Date'] = pd.to_datetime(df['Date'])
        # Remove timezone information before conversion
        df['Date'] = df['Date'].dt.tz_localize(None).astype('datetime64[ns]')

        # Perform other operations after conversion
        df["Year"] = df['Date'].dt.year
        df["Month"] = df['Date'].dt.month
        df["Day"] = df['Date'].dt.day
        df["DayOfWeek"] = df['Date'].dt.dayofweek
        df["DayOfYear"] = df['Date'].dt.dayofyear

        df.drop('Date', axis=1, inplace=True)

        return df

    def stock_prediction(self, time_range=1):
        date = pd.read_csv(self.data_dir)
        date = date["Date"]
        df = pd.DataFrame(columns=["Date"])
        df["Date"] = date[-time_range:]
        df = self._process_df(df)

        preds = self.model.predict(df)
        return preds


class LSTMModel:
    def __init__(self, stock_name, data_dir="..\data", model_dir="..\models"):
        self.stock_name = stock_name
        self.data_dir = data_dir
        self.model_dir = os.path.join(model_dir, self.stock_name)
        self.model = None
        self.scaler = None
        self.model_static = None
        self.stock_history = None

        self._load_model()
        self._load_stock_data()

    def _load_stock_data(self):
        data_path = os.path.join(self.data_dir, f"{self.stock_name}.csv")
        stock_his = pd.read_csv(data_path)
        print(stock_his.columns)
        stock_his = stock_his[["Date", "Adj Close"]]
        stock_his.index = stock_his["Date"]
        stock_his.drop("Date", axis=1, inplace=True)

        self.stock_history = stock_his

    def _load_model(self):
        model_path = os.path.join(self.model_dir, "LSTM.h5")
        scaler_path = os.path.join(self.model_dir, "LSTM_scaler.pkl")
        model_static_path = os.path.join(self.model_dir, "model_loss.json")

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            self.scaler = scaler

        with open(model_static_path, 'r') as f:
            model_static = json.load(f)
            self.model_static = model_static["LSTM"]

        self.model = load_model(model_path)

    def stock_prediction(self, time_range=1):
        time_range -= 3
        PAST_RANGE = 80

        # Lấy dữ liệu đầu vào cho lần dự đoán đầu tiên
        inputs = self.stock_history[len(self.stock_history) - time_range - PAST_RANGE:].values
        inputs = inputs.reshape(-1, 1)
        inputs = self.scaler.transform(inputs)

        # Chuẩn bị dữ liệu đầu vào cho mô hình
        data_input = []
        for i in range(PAST_RANGE, inputs.shape[0]):
            data_input.append(inputs[i - PAST_RANGE:i, 0])
        data_input = np.array(data_input)

        # Dự đoán giá cổ phiếu
        data_input = np.reshape(data_input, (data_input.shape[0], data_input.shape[1], 1))
        closing_price = self.model.predict(data_input)
        preds = self.scaler.inverse_transform(closing_price)
        preds = preds.tolist()
        preds = [i[0] for i in preds]

        # Thêm kết quả dự đoán vào stock_history
        last_date = self.stock_history.index[-1]
        next_date = pd.to_datetime(last_date) + pd.Timedelta(days=1)
        predicted_data = pd.DataFrame({"Adj Close": preds[-1:]}, index=[next_date])
        self.stock_history = pd.concat([self.stock_history, predicted_data])

        inputs = self.stock_history[len(self.stock_history) - PAST_RANGE:].values
        inputs = inputs.reshape(-1, 1)
        inputs = self.scaler.transform(inputs)
        predicted_future_prices = []

        # Dự đoán cho các ngày tiếp theo
        for i in range(3):  # Dự đoán cho 90 ngày
            input_data = inputs.reshape(1, 80, 1)
            future_price = self.model.predict(input_data)
            predicted_future_prices.append(future_price[0][0])
            inputs = np.append(input_data[:, 1:, :], future_price.reshape(1, 1, 1), axis=1)

        predicted_future_prices = self.scaler.inverse_transform(np.array(predicted_future_prices).reshape(-1, 1))
        predicted_future_prices = [i[0] for i in predicted_future_prices.tolist()]
        preds+= predicted_future_prices
        return preds


