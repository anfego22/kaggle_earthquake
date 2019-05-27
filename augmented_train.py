import pickle
import pymongo
import pandas as pd
import functions as fn
from sklearn.preprocessing import StandardScaler

N = 150000

ids = fn.get_sample_point(15000)

conection = pymongo.MongoClient()
db = conection.earthquake
augmented_data = []
for _ in range(len(ids)):
    _id = ids.pop()
    docs = db.train.find({'_id': {'$lte': _id}}, {'_id': 0}).\
           sort([('_id', -1)]).limit(N)
    data = pd.DataFrame([el for el in docs])
    while data.shape[0] != N:
        cursor = db.train.aggregate([{
            '$sample': {'size': 1}}])
        _id = [el['_id'] for el in cursor]
        docs = db.train.find({'_id': {'$lte': _id[0]}}, {'_id': 0}).\
               sort([('_id', -1)]).limit(N)
        data = pd.DataFrame([el for el in docs])
    augmented_doc = fn.data_reduction(data.values[:, 0],
                                      data.values[:, 1])
    augmented_doc['created_from'] = _id
    augmented_data.append()

db.augmented_train.insert_many(augmented_data, ordered=False)
conection.close()


cursor = db.augmented_train.find({}, {'_id': 0})
aug_data = pd.DataFrame([el for el in cursor])
features = aug_data[['mean', 'sd', 'max', 'min']].values
scaler = StandardScaler()
scaler.fit(features)

with open('augmented_scaler', 'wb') as f:
    pickle.dump(scaler, f)
