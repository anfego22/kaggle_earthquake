import pymongo
import pandas as pd
import numpy as np


def get_sample_point(n):
    conection = pymongo.MongoClient()
    db = conection.earthquake
    cursor = db.train.aggregate([{
        '$sample': {'size': n}}])
    ids = [el['_id'] for el in cursor]
    conection.close()
    return ids


def data_reduction_array(x, y=[]):
    if len(y) != 0:
        return [x.mean(), x.std(), x.max(), x.min(),
                y[-1]]
    else:
        return [x.mean(), x.std(), x.max(), x.min()]


def data_reduction_dict(x, y=[]):
    if len(y) != 0:
        return {'mean': x.mean(), 'sd': x.std(),
                'max': x.max(), 'min': x.min(),
                'time_to_failure': y[-1]}
    else:
        return {'mean': x.mean(), 'sd': x.std(),
                'max': x.max(), 'min': x.min()}


def data_reduction(x, y=[], array=False):
    if array:
        return data_reduction_array(x, y)
    else:
        return data_reduction_dict(x, y)


def preprocess_data(cursor, scaler, new_features=False):
    if new_features:
        return preprocess_data_newfeatures(cursor, scaler)
    else:
        return preprocess_data_simple(cursor, scaler)


def preprocess_data_simple(cursor, scaler):
    data = pd.DataFrame([el for el in cursor])
    data['acoustic_data'] = scaler.fit_transform(data['acoustic_data'].
                                                 values.reshape(-1, 1))
    features = data[data.columns[data.columns != 'time_to_failure']].values
    features = features[~np.isnan(features).any(axis=1)]
    labels = data['time_to_failure'].values[-1]
    return (features, labels)


def preprocess_data_newfeatures(cursor, scaler):
    data = pd.DataFrame([el for el in cursor])
    features = data[data.columns[data.columns != 'time_to_failure']].values
    new_features = data_reduction_array(features)
    features = features[~np.isnan(features).any(axis=1)]
    labels = data['time_to_failure'].values[-1]
    return (features, new_features, labels)
