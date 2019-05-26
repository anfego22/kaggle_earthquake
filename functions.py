import pymongo
import pandas as pd


def get_sample_point(n):
    conection = pymongo.MongoClient()
    db = conection.earthquake
    cursor = db.train.aggregate([{
        '$sample': {'size': n}}])
    ids = [el['_id'] for el in cursor]
    conection.close()
    return ids


def preprocess_data(cursor, scaler):
    data = pd.DataFrame([el for el in cursor])
    data['acoustic_data'] = scaler.fit_transform(data['acoustic_data'].
                                                 values.reshape(-1, 1))
    features = data[data.columns[data.columns != 'time_to_failure']].values
    features = features[~np.isnan(features).any(axis=1)]
    labels = data['time_to_failure'].values[-1]
    return (features, labels)


def data_reduction(x, y):
    return {'mean': x.mean(), 'sd': x.std(),
            'max': x.max(), 'min': x.min(),
            'time_to_failure': y[-1]}
