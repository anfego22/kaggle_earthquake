import tensorflow.keras as k
import tensorflow as tf
import pandas as pd
import pymongo
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

PATH = '/home/anfego/Documents/Kaggle/Earthquake/Data/Test/'
BATCH_SIZE = 10
WINDOW = 1400
N = 150000
FEATURES = 6
MODEL_NAME = 'initial_model'
new_model = True


def get_sample_point():
    conection = pymongo.MongoClient()
    db = conection.earthquake
    cursor = db.train.aggregate([{
        '$sample': {'size': BATCH_SIZE}}])
    ids = [el['_id'] for el in cursor]
    conection.close()
    return ids


def gen_features(features):
    features['mean'] = features['acoustic_data'].rolling(WINDOW).mean()
    features['sd'] = features['acoustic_data'].rolling(WINDOW).std()
    features['max'] = features['acoustic_data'].rolling(WINDOW).max()
    features['min'] = features['acoustic_data'].rolling(WINDOW).min()
    features['abs'] = features['acoustic_data'].abs()


def preprocess_data(cursor, scaler):
    data = pd.DataFrame([el for el in cursor])
    data['acoustic_data'] = scaler.fit_transform(data['acoustic_data'].\
                                                 values.reshape(-1, 1))
    gen_features(data)
    features = data[data.columns[data.columns != 'time_to_failure']].values
    features = features[~np.isnan(features).any(axis=1)]
    labels = data['time_to_failure'].values[-1]
    return (features, labels)


def get_data():
    while True:
        conection = pymongo.MongoClient()
        db = conection.earthquake
        ids = get_sample_point()
        features_batch = []
        label_batch = []
        scaler = StandardScaler()
        for _id in ids:
            docs = db.train.find({'_id': {'$lte': _id}}, {'_id': 0}).\
                   sort([('_id', -1)]).limit(N)
            features, labels = preprocess_data(docs, scaler)
            while features.shape[0] != (N - WINDOW + 1):
                cursor = db.train.aggregate([{
                    '$sample': {'size': BATCH_SIZE}}])
                ids = [el['_id'] for el in cursor]
                docs = db.train.find({'_id': {'$lte': _id}}, {'_id': 0}).\
                    sort([('_id', -1)]).limit(N)
                features, labels = preprocess_data(docs, scaler)
            features_batch.append(np.expand_dims(features, axis=0))
            label_batch.append(labels)
        features_batch = np.concatenate(features_batch, axis=0)
        label_batch = np.expand_dims(np.array(label_batch), axis=1)
        conection.close()
        yield (features_batch, label_batch)


# Looks it take a constant time no matter how many documents you pull
ds = tf.data.Dataset.from_generator(get_data,
                                    output_types=(tf.float32, tf.float32),
                                    output_shapes=((BATCH_SIZE, N - WINDOW + 1,
                                                    FEATURES),
                                                   (BATCH_SIZE, 1)))


# Model
if new_model:
    model = k.Sequential([k.layers.InputLayer(input_shape=(N - WINDOW + 1,
                                                           FEATURES)),
                          k.layers.Conv1D(10, kernel_size= 100, strides=1,
                                          activation='relu'),
                          k.layers.Conv1D(10, kernel_size=100, strides=100,
                                          activation='relu'),
                          k.layers.Conv1D(100, kernel_size=100, strides=100),
                          k.layers.Flatten(),
                          k.layers.Dense(64, activation='relu'),
                          k.layers.Dropout(0.1),
                          k.layers.Dense(10, activation='relu'),
                          k.layers.BatchNormalization(),
                          k.layers.Dense(1, activation='linear')])
model.summary()
model.compile(optimizer=k.optimizers.RMSprop(1e-3), loss='mae')
model.fit(ds, epochs=1, steps_per_epoch=528)


submission = []
scaler = StandardScaler()
for f in os.listdir(PATH):
    res = {}
    data = pd.read_csv(PATH + f, sep=',')
    data['acoustic_data'] = scaler.fit_transform(data['acoustic_data'].\
                                                 values.reshape(-1, 1))
    gen_features(data)
    features = data[data.columns[data.columns != 'time_to_failure']].values
    features = features[~np.isnan(features).any(axis=1)]
    features = np.expand_dims(features, axis=0)
    res['seg_id'] = f[:-4]
    res['time_to_failure'] = model.predict(features)[0][0]
    submission.append(res)

submission_pd = pd.DataFrame(submission)
submission_pd.to_csv('Submission/second_submission.csv', index=False)
