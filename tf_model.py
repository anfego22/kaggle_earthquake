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
N = 150000
MODEL_NAME = 'initial_model'
new_model = True

json_file = open(MODEL_NAME + 'arch.json', 'r')
model_json = json_file.read()
json_file.close()
model = k.models.model_from_json(model_json)
model.load_weights(MODEL_NAME + 'weight.index')
model.build(input_shape=(None, N, 1))
model.summary()


def get_sample_point():
    conection = pymongo.MongoClient()
    db = conection.earthquake
    cursor = db.train.aggregate([{
        '$sample': {'size': BATCH_SIZE}}])
    ids = [el['_id'] for el in cursor]
    conection.close()
    return ids


def preprocess_data(cursor, scaler):
    sample_doc = np.array([[np.int32(el['acoustic_data']),
                            el['time_to_failure']] for el in cursor])
    features = sample_doc[:, 0].reshape((-1, 1))
    scaler.fit(features)
    features = scaler.transform(features)
    labels = sample_doc[-1, 1]
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
            while features.shape[0] != N:
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
                                    output_shapes=((BATCH_SIZE, N, 1),
                                                   (BATCH_SIZE, 1)))


# Model
if new_model:
    model = k.Sequential([k.layers.InputLayer(input_shape=(N, 1)),
                          k.layers.Conv1D(10, kernel_size= 100, strides=1,
                                          activation='relu'),
                          k.layers.Conv1D(10, kernel_size=100, strides=100,
                                          activation='relu'),
                          k.layers.Conv1D(100, kernel_size=100, strides=100),
                          k.layers.Flatten(),
                          k.layers.Dense(10, activation='relu'),
                          k.layers.Dense(1, activation='linear')])

model.compile(optimizer=k.optimizers.RMSprop(1e-3), loss='mae')
model.fit(ds, epochs=1, steps_per_epoch=128)


submission = []
for f in os.listdir(PATH):
    res = {}
    data = np.genfromtxt(PATH + f, delimiter=',', skip_header=True)
    data = data.reshape((-1, 1))
    data = scaler.fit_transform(data)
    data = np.expand_dims(data, axis=0)
    res['seg_id'] = f[:-4]
    res['time_to_failure'] = model.predict(data)[0][0]
    submission.append(res)

submission_pd = pd.DataFrame(submission)
submission_pd['time_to_failure'] = submission_pd['time_to_failure'].\
                                   apply(lambda x: x[0][0])
submission_pd.to_csv('Submission/first_submission.csv', index=False)

model_json = model.to_json()
with open(MODEL_NAME + 'arch.json', 'w') as f:
    f.write(model_json)
model.save_weights(MODEL_NAME + 'weight', save_format='tf')

data = get_data()
feat, lab = next(data)

inputs = k.layers.InputLayer(input_shape=(N, 1))
extraction = k.Model(inputs=inputs, outputs=model.layers[0].output)
hidden_one = extraction.predict(feat[0, :, :])
rows = hidden_one.shape[1]

x = range(rows)
x2 = range(150000)
plt.plot(x2, feat[0, :, 0], color='r')
plt.plot(x, hidden_one[0, :, 1])
plt.show()
