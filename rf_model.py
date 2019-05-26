import os
import pymongo
import pandas as pd
import functions as fn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

PATH = '/home/anfego/Documents/Kaggle/Earthquake/Data/Test/'


conection = pymongo.MongoClient()
db = conection.earthquake
cursor = db.augmented_train.find({}, {'_id': 0})
augmented_data = [el for el in cursor]
augmented_data = pd.DataFrame(augmented_data)
X = augmented_data[['max', 'mean', 'min', 'sd']].values
y = augmented_data['time_to_failure'].values
scaler = StandardScaler()
rf = RandomForestRegressor(n_estimators=120, n_jobs=-1,
                           min_samples_leaf=1,
                           max_features = "auto", max_depth=15)
kf = KFold(5)
fold_ = 0
for trn_, val_ in kf.split(X):
    print("Current Fold: {}".format(fold_))
    trn_x, trn_y = X[trn_], y[trn_]
    val_x, val_y = X[val_], y[val_]
    trn_x = scaler.fit_transform(trn_x)
    rf.fit(trn_x, trn_y)
    val_x = scaler.fit_transform(val_x)
    rf_pred = rf.predict(val_x)
    print('RF MAE: {}'.format(mean_absolute_error(val_y, rf_pred)))
    fold_ += 1

submission = []
for f in os.listdir(PATH):
    res = {}
    data = pd.read_csv(PATH + f, sep=',')
    x = [fn.data_reduction(data.values)]
    x = pd.DataFrame(x).values
    x = scaler.transform(x)
    res['seg_id'] = f[:-4]
    res['time_to_failure'] = rf.predict(x)[0]
    submission.append(res)

submission_pd = pd.DataFrame(submission)
submission_pd.to_csv('Submission/fifth_submission.csv', index=False)
