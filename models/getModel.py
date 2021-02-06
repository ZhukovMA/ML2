import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

MOBILE_PATH = "datasets/"


def load_data(books_path=MOBILE_PATH):
    csv_path = os.path.join(books_path, "mobile_train.csv")
    return pd.read_csv(csv_path, error_bad_lines=False)


mobile_data = load_data()

target = 'price_range'
X = mobile_data.drop(labels=target, axis=1)
db = DBSCAN(eps=400, min_samples=3)

db.fit(X)
mobile_data = mobile_data.drop(np.where(db.labels_ == -1)[0])

mobile_data.to_csv('clearDataset.csv', index=False)
