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

with open('augmented_scaler', 'rb') as f:
    scaler_new = pickle.load(f)


def get_data():
    while True:
        conection = pymongo.MongoClient()
        db = conection.earthquake
        features_batch = []
        new_features_batch = []
        global ids
        label_batch = []
        scaler = StandardScaler()
        global scaler_new
        for _ in range(BATCH_SIZE):
            _id = ids.pop()
            docs = db.train.find({'_id': {'$lte': _id}}, {'_id': 0}).\
                   sort([('_id', -1)]).limit(N)
            features, new_features, labels = fn.preprocess_data(
                docs, scaler, True)
            while features.shape[0] != N:
                cursor = db.train.aggregate([{
                    '$sample': {'size': 1}}])
                new_id = [el['_id'] for el in cursor]
                docs = db.train.find({'_id': {'$lte': new_id}}, {'_id': 0}).\
                    sort([('_id', -1)]).limit(N)
                features, new_features, labels = fn.preprocess_data(
                    docs, scaler, True)
            new_features = np.expand_dims(new_features, axis=0)
            new_features = scaler_new.transform(new_features)
            new_features_batch.append(new_features)
            features_batch.append(np.expand_dims(features, axis=0))
            label_batch.append(labels)
        features_batch = np.concatenate(features_batch, axis=0)
        new_features_batch = np.concatenate(new_features_batch, axis=0)
        label_batch = np.expand_dims(np.array(label_batch), axis=1)
        conection.close()
        yield features_batch, new_features_batch, label_batch


# Looks it take a constant time no matter how many documents you pull
ds = tf.data.Dataset.from_generator(get_data,
                                    output_types=(tf.float64, tf.float64,
                                                  tf.float32),
                                    output_shapes=((BATCH_SIZE, N, FEATURES),
                                                   (BATCH_SIZE, AGG_FEAT),
                                                   (BATCH_SIZE, 1)))

model = k.models.load_model('cnn_two_inputs.h5')

if not model:
    raw_data = k.layers.Input(shape=(N, FEATURES))
    h = k.layers.Conv1D(16, kernel_size=500, strides=1,
                        activation='relu', padding='same',
                        kernel_regularizer=k.regularizers.l2(0.01))(raw_data)
    h = k.layers.Conv1D(5, kernel_size=512, strides=1024,
                        activation='relu', padding='same',
                        kernel_regularizer=k.regularizers.l2(0.01))(h)
    h = k.layers.Conv1D(100, kernel_size=100, strides=10,
                        activation='relu', padding='same',
                        kernel_regularizer=k.regularizers.l2(0.01))(h)
    h = k.layers.Flatten()(h)
    h = k.layers.Dense(10, activation='relu')(h)
    new_features = k.layers.Input(shape=(AGG_FEAT))
    h = k.layers.concatenate([h, new_features])
    predictions = k.layers.Dense(1, activation='linear')(h)
    model = k.Model([raw_data, new_features], predictions)
    model.compile(optimizer='adam', loss='mae')
model.summary()

for i in range(1):
    ids = fn.get_sample_point(BATCH_SIZE*EPOCH)
    j = 0
    mae = 0
    try:
        for el in ds.take(EPOCH):
            input1, input2, labs = el
            print('EPOCH {}'.format(i), 'BATCH {}'.format(j))
            new_mae = model.train_on_batch(x=[input1, input2], y=labs)
            j += 1
            mae = (BATCH_SIZE*(j-1)*mae + BATCH_SIZE*new_mae)/(j*BATCH_SIZE)
            print('MAE {}'.format(mae))
    except:
        continue

model.save('cnn_two_inputs.h5')

submission = []
scaler = StandardScaler()
for f in os.listdir(PATH):
    res = {}
    data = pd.read_csv(PATH + f, sep=',')
    new_features = fn.data_reduction(data.values, array=True)
    new_features = scaler_new.transform(np.expand_dims(new_features, axis=0))
    data['acoustic_data'] = scaler.fit_transform(data['acoustic_data'].
                                                 values.reshape(-1, 1))
    features = data[data.columns[data.columns != 'time_to_failure']].values
    features = features[~np.isnan(features).any(axis=1)]
    features = np.expand_dims(features, axis=0)
    res['seg_id'] = f[:-4]
    res['time_to_failure'] = model.predict([features, new_features])[0][0]
    submission.append(res)

submission_pd = pd.DataFrame(submission)
submission_pd.to_csv('Submission/seven_submission.csv', index=False)
