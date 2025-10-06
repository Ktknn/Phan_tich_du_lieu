import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class MovingAveragePredictor(BaseEstimator, RegressorMixin):
    def __init__(self, window_size=246):
        self.window_size = window_size  # Kích thước cửa sổ trung bình động
        self.history = None  # Lưu trữ dữ liệu để dự đoán

    def fit(self, X, y=None):
        self.history = X # Lưu dữ liệu huấn luyện dưới dạng mảng 1D
        return self

    def predict(self, X):
        history = self.history
        print(history.shape)
        preds = []
        for i in range(len(X)):
            moving_avg = (np.sum(history[len(history)-self.window_size+i:]) + sum(preds)) / self.window_size
            preds.append(moving_avg)
        return np.array(preds)