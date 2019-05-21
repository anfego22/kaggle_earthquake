import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras as k
from tensorflow.keras.models import load_model
import numpy as np
from pymongo import MongoClient
import matplotlib.pyplot as plt


N = 150000
BATCH_SIZE = 10

db = MongoClient(connect=False)
load_model('convolutional_nn_v1.h5')

# https://towardsdatascience.com/how-to-quickly-build-a-tensorflow-training-pipeline-15e9ae4d78a0
# https://stackoverflow.com/questions/48769142/tensorflow-how-to-use-dataset-from-generator-in-estimator
# https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
# Use tf.Dataset.from_generator()


class mongo_sequence(k.utils.Sequence):
    def __init__(self, batch_size=10, n=150000, epoch_size=20):
        self.batch_size = batch_size
        self.n = 150000
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        db = MongoClient()
        cursor = db.earthquake.train.aggregate([{
            '$sample': {'size': self.batch_size}}])
        features_batch = []
        label_batch = []
        scaler = StandardScaler(copy=False)
        for el in cursor:
            _id = el['_id']
            docs = db.earthquake.train.find(
                {'_id': {'$lte': _id}}, {'_id': 0}).sort(
                    [('_id', -1)]).limit(self.n)
            sample_doc = pd.DataFrame([el for el in docs])
            label_batch.append(sample_doc.time_to_failure.values[-1])
            features = sample_doc['acoustic_data'].values.reshape((-1, 1))
            scaler.fit(features)
            features = scaler.transform(features)
            features_batch.append(np.expand_dims(features, axis=0))
        features_batch = np.concatenate(features_batch, axis=0)
        db.close()
        return (features_batch, np.array(label_batch))


generator = mongo_sequence(batch_size=10, epoch_size=128)
if not model:
    model = k.Sequential(
        [k.layers.InputLayer(input_shape=(150000, 1)),
         k.layers.Conv1D(12, 500, 1, activation='relu'),
         k.layers.Conv1D(20, 100, 1000, activation='relu'),
         k.layers.Flatten(),
         k.layers.Dense(100, activation='relu'),
         k.layers.BatchNormalization(),
         k.layers.Dense(32, activation='relu'),
         k.layers.BatchNormalization(),
         k.layers.Dense(1, activation='linear')])
    model.compile(optimizer='adam', loss='mae')

model.fit_generator(generator, steps_per_epoch=128, epochs=10,
                    shuffle=False, use_multiprocessing=False)
model.save('convolutional_nn_v1.h5')

feat, lab = generator.__getitem__(0)
extraction = k.Model(inputs=model.input, outputs=model.layers[0].output)
hidden_one = extraction.predict(feat)
rows = hidden_one.shape[1]

x = range(rows)
x2 = range(150000)
plt.plot(x2, feat[0, :, 0], color='r')
plt.plot(x, hidden_one[0, :, 1])
plt.show()
