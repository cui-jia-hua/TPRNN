import os
import numpy as np
import pandas as pd
import os
import torch
from utils.timefeatures import time_features

def get_time_feature(df, timeenc, freq):
    if timeenc == 0:
        df['month'] = df.date.apply(lambda row: row.month, 1)
        df['day'] = df.date.apply(lambda row: row.day, 1)
        df['weekday'] = df.date.apply(lambda row: row.weekday(), 1)
        df['hour'] = df.date.apply(lambda row: row.hour, 1)
        # 加入分钟级编码
        if freq in ['s','t']:
            df['minute'] = df.date.apply(lambda row: row.minute, 1)
            df['minute'] = df.minute.map(lambda x: x // 15)
        features = df.drop(['date'], 1).values
    elif timeenc == 1:
        data_stamp = time_features(pd.to_datetime(df['date'].values), freq=freq)
        features = data_stamp.transpose(1, 0)

    return features