import numpy as np
import pandas as pd

class NpDataloader:

    def __init__(self, csv_path="nasdaq_nyse_combined.csv"):
        self.prices = pd.read_csv(csv_path, index_col="date", parse_dates=True).sort_index()
        self.returns = self.prices.pct_change().dropna()

def load_data(end_date, T) -> np.ndarray: 
    window = self.returns.loc[:end_date].tail(T);
    return window.values
