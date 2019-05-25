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
EPOCH = 32
N = 150000
MODEL_NAME = 'initial_model'
FEATURES = 1


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


def get_data():
    while True:
        conection = pymongo.MongoClient()
        db = conection.earthquake
        features_batch = []
        global ids
        label_batch = []
        scaler = StandardScaler()
        for _ in range(BATCH_SIZE):
            _id = ids.pop()
            docs = db.train.find({'_id': {'$lte': _id}}, {'_id': 0}).\
                   sort([('_id', -1)]).limit(N)
            features, labels = preprocess_data(docs, scaler)
            while features.shape[0] != N:
                cursor = db.train.aggregate([{
                    '$sample': {'size': 1}}])
                new_id = [el['_id'] for el in cursor]
                docs = db.train.find({'_id': {'$lte': new_id}}, {'_id': 0}).\
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
                                    output_shapes=((BATCH_SIZE, N,
                                                    FEATURES),
                                                   (BATCH_SIZE, 1)))


# Model
model = k.Sequential([k.layers.InputLayer(input_shape=(N, 1)),
                      k.layers.Conv1D(10, kernel_size= 100, strides=1,
                                      activation='relu'),
                      k.layers.Conv1D(10, kernel_size=100, strides=100,
                                      activation='relu'),
                      k.layers.Conv1D(100, kernel_size=100, strides=100),
                      k.layers.Flatten(),
                      k.layers.Dense(10, activation='relu'),
                      k.layers.Dense(1, activation='linear')])
model.compile(optimizer='adam', loss='mae')
model.summary()
for i in range(10):
    ids = get_sample_point(BATCH_SIZE*EPOCH)
    model.fit(ds, epochs=1, steps_per_epoch=EPOCH)


submission = []
scaler = StandardScaler()
for f in os.listdir(PATH):
    res = {}
    data = pd.read_csv(PATH + f, sep=',')
    data['acoustic_data'] = scaler.fit_transform(data['acoustic_data'].\
                                                 values.reshape(-1, 1))
    features = data[data.columns[data.columns != 'time_to_failure']].values
    features = features[~np.isnan(features).any(axis=1)]
    features = np.expand_dims(features, axis=0)
    res['seg_id'] = f[:-4]
    res['time_to_failure'] = model.predict(features)[0][0]
    submission.append(res)

submission_pd = pd.DataFrame(submission)
submission_pd.to_csv('Submission/third_submission.csv', index=False)
