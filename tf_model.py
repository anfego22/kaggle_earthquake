import pickle
import tensorflow.keras as k
import tensorflow as tf
import pandas as pd
import pymongo
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import functions as fn

PATH = '/home/anfego/Documents/Kaggle/Earthquake/Data/Test/'
BATCH_SIZE = 10
EPOCH = 32
N = 150000
AGG_FEAT = 4
FEATURES = 1

y_mean = 5.6816
y_std = 3.6345


with open('augmented_scaler', 'rb') as f:
    scaler_new = pickle.load(f)


def get_data():
    while True:
        conection = pymongo.MongoClient()
        db = conection.earthquake
        features_batch = []
        global ids, y_mean, y_std
        label_batch = []
        scaler = StandardScaler()
        for _ in range(BATCH_SIZE):
            _id = ids.pop()
            docs = db.train.find({'_id': {'$lte': _id}}, {'_id': 0}).\
                sort([('_id', -1)]).limit(N)
            features, labels = fn.preprocess_data(
                docs, scaler, False)
            while features.shape[0] != N:
                cursor = db.train.aggregate([{
                    '$sample': {'size': 1}}])
                new_id = [el['_id'] for el in cursor]
                docs = db.train.find({'_id': {'$lte': new_id}}, {'_id': 0}).\
                    sort([('_id', -1)]).limit(N)
                features, labels = fn.preprocess_data(
                    docs, scaler, False)
            labels = (labels - y_mean)/y_std
            features_batch.append(np.expand_dims(features, axis=0))
            label_batch.append(labels)
        features_batch = np.concatenate(features_batch, axis=0)
        label_batch = np.expand_dims(np.array(label_batch), axis=1)
        conection.close()
        yield features_batch, label_batch


# Looks it take a constant time no matter how many documents you pull
ds = tf.data.Dataset.from_generator(get_data,
                                    output_types=(tf.float64, tf.float32),
                                    output_shapes=((BATCH_SIZE, N, FEATURES),
                                                   (BATCH_SIZE, 1)))

raw_data = k.layers.Input(shape=(N, FEATURES))
h = k.layers.Conv1D(10, kernel_size=100, strides=1,
                    activation='relu', padding='same')(raw_data)
h = k.layers.Conv1D(10, kernel_size=100, strides=100,
                        activation='relu', padding='same')(h)
h = k.layers.Conv1D(100, kernel_size=100, strides=100,
                    activation='relu', padding='same')(h)
h = k.layers.Flatten()(h)
h = k.layers.Dense(10, activation='relu')(h)
predictions = k.layers.Dense(1, activation='linear')(h)
model = k.Model(raw_data, predictions)
model.compile(optimizer='adam', loss='mae')
model.summary()

for i in range(128):
    ids = fn.get_sample_point(BATCH_SIZE*EPOCH)
    j = 0
    mae = 0
    try:
        for el in ds.take(EPOCH):
            input1, labs = el
            print('EPOCH {}'.format(i), 'BATCH {}'.format(j))
            new_mae = model.train_on_batch(x=input1, y=labs)
            j += 1
            mae = (BATCH_SIZE*(j-1)*mae + BATCH_SIZE*new_mae)/(j*BATCH_SIZE)
            print('MAE {}'.format(mae))
    except:
        continue
    if j % 8 == 0:
        model.save('cnn_first_mode_v2.h5')

submission = []
scaler = StandardScaler()
for f in os.listdir(PATH):
    res = {}
    data = pd.read_csv(PATH + f, sep=',')
    data['acoustic_data'] = scaler.fit_transform(data['acoustic_data'].
                                                 values.reshape(-1, 1))
    features = data[data.columns[data.columns != 'time_to_failure']].values
    features = features[~np.isnan(features).any(axis=1)]
    features = np.expand_dims(features, axis=0)
    res['seg_id'] = f[:-4]
    res['time_to_failure'] = model.predict(features)[0][0]
    res['time_to_failure'] = res['time_to_failure']*y_std + y_mean
    submission.append(res)

submission_pd = pd.DataFrame(submission)
submission_pd.to_csv('Submission/nine_submission.csv', index=False)
